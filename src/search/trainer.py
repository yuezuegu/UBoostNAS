import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
import wandb

import src.search as search
import src.search.layers
import src.search.model

from src.utils import Stats, accuracy, plot_alphas

from src.hardware.sim_binder import run_csim
from src.hardware.precompile import precompile_model, calc_no_ops

# HW-aware loss function with latency and utilization
class UtilLoss(nn.Module):
	def __init__(self, lat_coef, util_coef):
		super(UtilLoss, self).__init__()
		self.lat_coef = lat_coef
		self.util_coef = util_coef

	def forward(self, logits, targets, latency, util):
		loss = F.cross_entropy(logits, targets) + self.lat_coef * latency - self.util_coef * util
		assert not torch.any(torch.isnan(loss)), "loss value is NaN"
		return loss

class Trainer:
    def __init__(self, device, hw_model, parser_args) -> None:
        self.args = parser_args
        self.device = device
        self.hw_model = hw_model

    def init_microsearch(self,input_dims,no_classes):
        self.ms_model = search.Model(
            module_dict=None,
            nasmode=search.dmask.NASMODE.microsearch,
            hw_model=self.hw_model,
            device=self.device,
            input_dims=input_dims, 
            no_classes=no_classes,
            args = self.args
            ).to(device=self.device)

        self.tau = self.ms_model.tau

        self.ms_weight_optimizer = torch.optim.SGD(
            params=self.ms_model.parameters(),
            lr=self.args.search_sgd_init_lr,
            momentum=self.args.search_sgd_momentum,
            weight_decay=self.args.search_sgd_weight_decay,
        )

        # NAS is optimized via Adam for the remaining 20% of epochs
        self.ms_arch_optimizer = torch.optim.Adam(
            params=self.ms_model.get_arch_parameters_as_list(),
            lr=self.args.adam_init_lr,
            weight_decay=self.args.adam_weight_decay,
        )     

        self.ms_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.ms_weight_optimizer, 
            step_size=self.args.search_lr_scheduler_step, 
            gamma=self.args.search_lr_scheduler_gamma)
        
        self.ms_loss_fn = UtilLoss(self.args.lat_coef, self.args.util_coef).to(self.device)
        self.ms_stats = Stats()

        self.ms_model.plot_model(self.args.save_dir + '/ms_model')

    def init_channel_search(self,input_dims,no_classes):
        self.cs_model = self.ms_model.convert_to_channel_search(input_dims,no_classes)
        self.ms_model = None
        torch.cuda.empty_cache()

        self.cs_weight_optimizer = torch.optim.SGD(
            params=self.cs_model.parameters(),
            lr=self.args.search_sgd_init_lr,
            momentum=self.args.search_sgd_momentum,
            weight_decay=self.args.search_sgd_weight_decay,
        )

        # NAS is optimized via Adam for the remaining 20% of epochs
        self.cs_arch_optimizer = torch.optim.Adam(
            params=self.cs_model.get_arch_parameters_as_list(),
            lr=self.args.adam_init_lr,
            weight_decay=self.args.adam_weight_decay,
        )     

        self.cs_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.cs_weight_optimizer, 
            step_size=self.args.search_lr_scheduler_step, 
            gamma=self.args.search_lr_scheduler_gamma)

        self.cs_loss_fn = UtilLoss(self.args.lat_coef, self.args.util_coef).to(self.device)
        self.cs_stats = Stats()

        self.cs_model.plot_model(self.args.save_dir + '/cs_model')

    def microarch_search(self, train_dataloader, valid_dataloader, epochs):
        for epoch in range(epochs):
            wandb.log({"epoch": epoch})
            logging.info(f"microarch search epoch: {epoch}")
            self.ms_stats.set_epoch(epoch)
            epoch_start = time.time()

            self.run_epoch(self.ms_model, self.ms_loss_fn, train_dataloader, self.ms_weight_optimizer, self.ms_arch_optimizer, self.ms_stats, epoch)

            self.infer(self.ms_model, valid_dataloader, self.ms_loss_fn, self.ms_stats)

            self.run_csim(self.ms_model, self.ms_stats)

            self.ms_stats.log('microarch_search', ['train_top1', 'valid_top1'])

            # update lr
            self.ms_lr_scheduler.step()

            self.ms_model.anneal_tau()

            alphas = self.ms_model.get_arch_parameters()
            for layer in alphas:
                logging.info("{}: {}".format(layer, alphas[layer]))

            self.ms_stats.export_json(self.args.save_dir + '/ms_stats.json')

            epoch_elapsed = time.time() - epoch_start
            logging.info("end of epoch elapsed time: {} s".format(epoch_elapsed))

        self.ms_model.export_arch_state(self.args.save_dir + '/ms_arch_state.json')

    def channel_search(self, train_dataloader, valid_dataloader, epochs):
        for epoch in range(epochs):
            wandb.log({"epoch": epoch})
            logging.info(f"channel search epoch: {epoch}")
            self.cs_stats.set_epoch(epoch)
            epoch_start = time.time()

            self.run_epoch(self.cs_model, self.cs_loss_fn, train_dataloader, self.cs_weight_optimizer, self.cs_arch_optimizer, self.cs_stats, epoch)

            self.infer(self.cs_model, valid_dataloader, self.cs_loss_fn, self.cs_stats)

            self.run_csim(self.cs_model, self.cs_stats)

            self.cs_stats.log('channel_search', ['train_top1', 'valid_top1'])

            # update lr
            self.cs_lr_scheduler.step()

            self.cs_model.anneal_tau()

            alphas = self.cs_model.get_arch_parameters()
            for layer in alphas:
                logging.info("{}: {}".format(layer, alphas[layer]))

            self.cs_stats.export_json(self.args.save_dir + '/cs_stats.json')

            epoch_elapsed = time.time() - epoch_start
            logging.info("end of epoch elapsed time: {} s".format(epoch_elapsed))

        self.cs_model.export_arch_state(self.args.save_dir + '/cs_arch_state.json')

    def run_epoch(self, model, loss_fn, train_dataloader, weight_optimizer, arch_optimizer, stats, epoch):
        model.train()
        for step, batch in enumerate(train_dataloader):
            iter_per_epoch = len(train_dataloader)

            x, y = batch

            x = x.requires_grad_(False).to(self.device)
            y = y.requires_grad_(False).to(self.device)

            if arch_optimizer is None or step < iter_per_epoch * self.args.weight_vs_arch:
                # optimize weights of current architecture
                self.train_weights(model, x, y, loss_fn, weight_optimizer, stats, self.args.search_weight_grad_clip)
                #logging.info("training weights")
            else:
                # NAS optimization
                self.train_arch(model, x, y, loss_fn, arch_optimizer, stats)
                #logging.info("training architecture")

            if (step+1) % self.args.report_freq == 0:
                stats.log('search', ['train_top1'])

                model.log_eff_channels()

    def train_weights(self, model, x, y, loss_fn, nn_optimizer, stats, grad_clip=5):
        model.train()

        nn_optimizer.zero_grad()
        logits, runtime, util = model(x)

        loss = loss_fn(logits, y, runtime, util)

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        nn_optimizer.step()

        prec1, prec5 = accuracy(logits, y, topk=(1, 5))
        n = x.size(0)
        stats.update_avg("train_loss", n, loss)
        stats.update_avg("train_top1", n, prec1)
        stats.update_avg("train_top5", n, prec5)
        stats.set_value("runtime_model", runtime)
        stats.set_value("util_model", util)

    def train_arch(self, model, x, y, loss_fn, arch_optimizer, stats, grad_clip=5):
        model.train()

        arch_optimizer.zero_grad()

        logits, runtime, util = model(x)

        loss = loss_fn(logits, y, runtime, util)

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.get_arch_parameters().values(), grad_clip)

        arch_optimizer.step()

        prec1, prec5 = accuracy(logits, y, topk=(1, 5))
        n = x.size(0)
        stats.update_avg("train_loss", n, loss)
        stats.update_avg("train_top1", n, prec1)
        stats.update_avg("train_top5", n, prec5)
        stats.set_value("runtime_model", runtime)
        stats.set_value("util_model", util)

    def infer(self, model, data_queue, loss_fn, stats):
        model.eval()

        for step, (x, y) in enumerate(data_queue):
            x = x.requires_grad_(False).to(self.device)
            y = y.requires_grad_(False).to(self.device)

            logits, runtime, util = model(x)

            loss = loss_fn(logits, y, runtime, util)

            prec1, prec5 = accuracy(logits, y, topk=(1, 5))
            n = x.size(0)
            stats.update_avg("valid_loss", n, loss)
            stats.update_avg("valid_top1", n, prec1)
            stats.update_avg("valid_top5", n, prec5)

    def run_csim(self, model, stats):
        layers = precompile_model(model.convert_to_keras(), array_size=self.args.array_size, partition_size=None)
        no_ops = calc_no_ops(layers)

        json_out = {"args":self.args.__dict__, "model1":{"order":list(layers.keys()), "layers":layers, "no_repeat":1, "no_ops":no_ops}}

        sim_res = run_csim(json_out)

        csim_runtime = sim_res['no_cycles'] * 1e-9 * 1e3
        throughput = sim_res['no_ops'] / (csim_runtime/1e3)
        csim_util = throughput / (2*self.hw_model.peak_throughput())    

        stats.set_value('runtime_csim',csim_runtime)
        stats.set_value('util_csim',csim_util)
        stats.set_value('no_ops',sim_res['no_ops'])

        logging.info("runtime_csim: {} ms".format(csim_runtime))

