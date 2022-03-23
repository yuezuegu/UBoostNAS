from copy import deepcopy

import tensorflow as tf

import numpy as np
import json 

import torch
import torch.nn as nn

import logging 

import wandb

from tensorflow.keras.utils import plot_model

from src.hardware.sim_binder import run_csim
from src.hardware.precompile import precompile_model 

import src.search as search
import src.vanilla as vanilla
from src.search.dmask import NASMODE

class Model(nn.Module):
    def __init__(self, module_dict, nasmode, device, hw_model, input_dims, no_classes, args) -> None:
        super().__init__()
        self.device = device
        self.hw_model = hw_model
        self.nasmode = nasmode
        self.input_dims = input_dims
        self.no_classes = no_classes
        self.args = args

        self.tau = torch.tensor(args.init_tau, dtype=torch.float).requires_grad_(False)

        if module_dict is None:
            self.arch_state = {}
            multiplier = self.input_dims[0]

            self.module_dict = nn.ModuleDict()

            self.module_dict["stem"] = vanilla.layers.Stem(in_channels=self.input_dims[2], out_channels=128)

            multiplier = multiplier / 2
            mult_prev = multiplier
            mult_prevprev = multiplier

            ch_prev = self.module_dict["stem"].out_channels

            arch = None
            out_maxpool = self.args.ms_stacks
            for i in range(len(out_maxpool)):
                in_maxpool = True if mult_prev != mult_prevprev else False

                midstage = search.Cell(
                    layer_name="midstage"+str(i+1),
                    block_dict=None, 
                    in_channels=ch_prev,
                    out_channels=ch_prev,
                    out_range=None,
                    arch_state=arch,
                    nasmode=NASMODE.microsearch,
                    in_maxpool=in_maxpool,
                    out_maxpool=out_maxpool[i],
                    hw_model=hw_model, 
                    device=device)

                if out_maxpool[i]:
                    multiplier = multiplier / 2

                mult_prevprev = mult_prev
                mult_prev = multiplier

                ch_prev = midstage.out_channels

                self.module_dict["midstage"+str(i+1)] = midstage

                if arch is None:
                    arch = self.module_dict["midstage"+str(i+1)].arch_state
                    self.arch_state["midstage"+str(i+1)] = arch

            self.module_dict["classifier"] = vanilla.layers.Classifier(in_channels=ch_prev, out_channels=self.no_classes, multiplier=multiplier)

        else:
            self.arch_state = {}
            for k in module_dict:
                if hasattr(module_dict[k], "arch_state"):
                    self.arch_state[k] = module_dict[k].arch_state
            
            self.module_dict = module_dict

    def forward(self, x):
        self.update_alpha_gumbel()

        runtime = 0
        util = 0
        
        x_curr = x
        x_curr = self.module_dict["stem"](x_curr)

        ch_prev = self.module_dict["stem"].out_channels
        ch_prevprev = self.module_dict["stem"].out_channels
        x_prev = x_curr
        x_prevprev = x_curr

        for layer_name, layer in self.module_dict.items():
            if isinstance(layer, search.cell.Cell):
                x_curr, ch_curr, layer_runtime, layer_util = layer(x_prev, x_prevprev, ch_prev, ch_prevprev)

                ch_prevprev = ch_prev
                ch_prev = ch_curr
                x_prevprev = x_prev
                x_prev = x_curr
            
                if layer_runtime > 0:
                    runtime += layer_runtime
                    util += layer_util

        x_curr = self.module_dict["classifier"](x_curr)

        return x_curr, runtime, util

    def update_alpha_gumbel(self):
        for k in self.arch_state:
            self.arch_state[k].update_alpha_gumbel(self.tau)

    def get_arch_parameters(self):
        arch_params = {}
        for name, module in self.module_dict.items():
            alpha = None
            if hasattr(module,'get_arch_parameters'):
                alpha = module.get_arch_parameters()
            if alpha is not None:
                arch_params[name] = alpha
        return arch_params

    def get_arch_parameters_as_list(self):
        alpha_list = []
        alphas = self.get_arch_parameters()
        for layer in alphas:
            if isinstance(alphas[layer], dict):
                alpha_list += list(alphas[layer].values())
            else:
                alpha_list.append(alphas[layer])
                
        return alpha_list

    def get_soft_eff_channels(self):
        eff_channels = {}
        for name, module in self.module_dict.items():
            if hasattr(module,'nasmode'):
                if module.nasmode == NASMODE.channelsearch:
                    eff_channels[name] = module.get_soft_eff_channels()
        return eff_channels

    def get_hard_eff_channels(self):
        eff_channels = {}
        for name, module in self.module_dict.items():
            if hasattr(module,'nasmode'):
                if module.nasmode == NASMODE.channelsearch:
                    eff_channels[name] = module.get_hard_eff_channels()
        return eff_channels



    def convert_to_channel_search(self, input_dims, no_classes):
        self.input_dims = input_dims
        self.no_classes = no_classes

        multiplier = self.input_dims[0]
        channel_range = self.args.channel_range
        
        new_dict = nn.ModuleDict()

        new_dict["stem"] = vanilla.layers.Stem(in_channels=self.input_dims[2], out_channels=128)

        multiplier = multiplier / 2
        mult_prev = multiplier
        mult_prevprev = multiplier

        out_channels = new_dict["stem"].out_channels

        ch_prev = out_channels
        ch_prevprev = out_channels

        out_maxpool = self.args.cs_stacks
        for i in range(len(out_maxpool)):

            in_maxpool = True if mult_prev != mult_prevprev else False

            midstage = self.module_dict["midstage1"].convert_to_channel_search(ch_prev, ch_prevprev, channel_range, in_maxpool, out_maxpool[i])
            new_dict.add_module("midstage"+str(i+1), midstage)

            if out_maxpool[i]:
                multiplier = multiplier / 2

            mult_prevprev = mult_prev
            mult_prev = multiplier

            ch_prevprev = ch_prev
            ch_prev = midstage.out_channels

        new_dict["classifier"] = vanilla.layers.Classifier(in_channels=ch_prev, out_channels=self.no_classes, multiplier=multiplier)

        return search.model.Model(new_dict, NASMODE.channelsearch, self.device, self.hw_model, self.input_dims, self.no_classes, self.args).to(device=self.device)
    
    def convert_to_vanilla(self, input_dims, no_classes):
        self.input_dims = input_dims
        self.no_classes = no_classes

        multiplier = self.input_dims[0]
        new_dict = nn.ModuleDict()

        new_dict["stem"] = vanilla.layers.Stem(in_channels=self.input_dims[2], out_channels=128)

        multiplier = multiplier / 2
        mult_prev = multiplier
        mult_prevprev = multiplier

        out_channels = new_dict["stem"].out_channels

        ch_prev = out_channels
        ch_prevprev = out_channels

        out_maxpool = self.args.cs_stacks
        for i in range(len(out_maxpool)):
            in_maxpool = True if mult_prev != mult_prevprev else False

            midstage = self.module_dict["midstage"+str(i+1)].convert_to_vanilla(ch_prev, ch_prevprev, in_maxpool, out_maxpool[i])
            new_dict.add_module("midstage"+str(i+1), midstage)

            if out_maxpool[i]:
                multiplier = multiplier / 2

            mult_prevprev = mult_prev
            mult_prev = multiplier

            ch_prevprev = ch_prev
            ch_prev = midstage.out_channels

        new_dict["classifier"] = vanilla.layers.Classifier(in_channels=ch_prev, out_channels=self.no_classes, multiplier=multiplier)
    
        return vanilla.Model(new_dict, self.input_dims, self.no_classes, self.args)

    def convert_to_keras(self):
        input_layer = tf.keras.Input(shape=self.input_dims, batch_size=1, name="input_1")
        x = input_layer

        x = self.module_dict["stem"].convert_keras(x, layer_name="stem")

        x_prev = x
        x_prevprev = x 
        for k, module in self.module_dict.items():
            if isinstance(module, search.cell.Cell):
                x = module.convert_keras(x_prev, x_prevprev, layer_name=k)

                x_prevprev = x_prev
                x_prev = x

        x = self.module_dict["classifier"].convert_keras(x, layer_name="classifier")

        model = tf.keras.models.Model(inputs=input_layer, outputs=x)
        return model

    def anneal_tau(self):
        # anneal tau (tau_anneal_rate < 1)
        if self.tau > self.args.min_tau:
            self.tau *= self.args.tau_anneal_rate

        logging.info("Tau of gumbel_softmax: {}".format(self.tau))
        wandb.log({"tau": self.tau})

    def log_eff_channels(self):
        if self.nasmode == NASMODE.channelsearch:
            eff_channels = self.get_soft_eff_channels()
            for layer in eff_channels:
                wandb.log({f"soft_eff_channels/{layer}": eff_channels[layer]})

            eff_channels = self.get_hard_eff_channels()
            for layer in eff_channels:
                wandb.log({f"hard_eff_channels/{layer}": eff_channels[layer]})

    def plot_model(self, filename):
        s = ""
        for k, module in self.module_dict.items():
            _s = "{}".format(module.print_layer())
            s += "\n" + _s
            logging.info(_s)
        
        with open(filename+".txt", "w") as f:
            f.write(s)

        plot_model(self.convert_to_keras(), to_file=filename+".png", show_shapes=True, show_layer_names=True)

    def load_arch_state(self, filename):
        with open(filename, "r") as f:
            out = json.load(f)
        for layername in out:
            self.arch_state[layername].load_arch_state(out[layername])
            self.module_dict[layername].set_arch_parameters(self.arch_state[layername])

    def export_arch_state(self, filename):
        out = {}
        for layername in self.arch_state:
            arch = self.arch_state[layername]
            out[layername] = arch.export_arch_state()
            
        with open(filename, "w") as f:
            json.dump(out, f)


