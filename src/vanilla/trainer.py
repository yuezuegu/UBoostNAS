import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

import logging
import wandb

import src
from src.utils import Stats, accuracy
from src.hardware.sim_binder import run_csim
from src.hardware.precompile import precompile_model, calc_no_ops


import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


# Loss function with only accuracy for final training
class AccLoss(nn.Module):
	def __init__(self):
		super(AccLoss, self).__init__()

	def forward(self, logits, targets):
		loss = F.cross_entropy(logits, targets)
		assert not torch.any(torch.isnan(loss)), "loss value is NaN"
		return loss

class Trainer:
    def __init__(self, device, parser_args) -> None:
        self.args = parser_args
        self.device = device

        pl_logger = WandbLogger(
            project='eccv-experiments', 
            entity='dmaskingnas', 
            config=self.args,
            mode='online' if self.args.wandb_enable else 'disabled'
        )

        self.pl_trainer = pl.Trainer(
            max_epochs=self.args.ft_no_epoch,
            # precision=32,
            precision=16,
            gpus=-1,
            strategy="dp",
            logger=[pl_logger],
            default_root_dir=self.args.save_dir,
            gradient_clip_val=self.args.train_weight_grad_clip,
            progress_bar_refresh_rate=100 #0 to disable progress bar
            #profiler="simple"
        )

    def init_final_train(self, cs_model, input_dims, no_classes):
        self.ft_model = cs_model.convert_to_vanilla(input_dims, no_classes)

        self.pl_model = src.vanilla.PLWrapper(
            model=self.ft_model,
            config=None,
            parser_args=self.args
        )

        self.ft_stats = Stats()

        self.ft_model.plot_model(self.args.save_dir + '/ft_model')
        
        self.run_csim(self.ft_model, self.ft_stats)

    def train_final(self, train_dataloader, valid_dataloader):
        self.pl_trainer.fit(model=self.pl_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    def test_final(self, test_dataloader):
        self.pl_trainer.test(
            model=self.pl_model,
            test_dataloaders=test_dataloader
        )

    def run_csim(self, model, stats):
        layers = precompile_model(model.convert_to_keras(), array_size=self.args.array_size, partition_size=None)
        no_ops = calc_no_ops(layers)

        json_out = {"args":self.args.__dict__, "model1":{"order":list(layers.keys()), "layers":layers, "no_repeat":1, "no_ops":no_ops}}

        sim_res = run_csim(json_out)

        csim_runtime = sim_res['no_cycles'] / self.args.clk_freq * 1e3
        throughput = sim_res['no_ops'] / (csim_runtime/1e3)
        csim_util = throughput / (2*self.args.array_size[0]*self.args.array_size[1]*self.args.clk_freq)     

        stats.set_value('runtime_csim',csim_runtime)
        stats.set_value('util_csim',csim_util)
        stats.set_value('no_ops',sim_res['no_ops'])

        stats.log("csim", ["runtime_csim", "util_csim", "no_ops"])

        logging.info("runtime_csim: {} ms".format(csim_runtime))
