

import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class NASMODE:
    microsearch = 0
    channelsearch = 1
    vanilla = 2

class Arch_State:
    def __init__(self, device, input_file = None) -> None:
        self.device = device
        self.layer_alphas = {}
        self.alpha_gumbels = {}
        self.channel_alphas = {}
        self.channel_masks = {}
        self.ind2ch = {}
        self.out_range = {}

        if input_file is not None:
            self.load_arch_state(input_file)

    def add_layer_alphas(self, block_name, no_cells):
        alphas = torch.randn(no_cells)
        self.layer_alphas[block_name] = alphas.to(self.device).requires_grad_(True)

    def update_alpha_gumbel(self, tau):
        for block_name in self.layer_alphas:
            self.alpha_gumbels[block_name] = F.gumbel_softmax(self.layer_alphas[block_name], tau=tau.item(), hard=False, eps=1e-10, dim=-1)

        for block_name in self.channel_alphas:
            self.alpha_gumbels[block_name] = F.gumbel_softmax(self.channel_alphas[block_name], tau=tau.item(), hard=False, eps=1e-10, dim=-1)

    def get_layer_alpha(self, name):
        if name in self.layer_alphas:
            return self.layer_alphas[name]
        return None

    def get_alpha_gumbel(self, name):
        if name in self.alpha_gumbels:
            return self.alpha_gumbels[name]
        return None
           
    def get_dmask_vars(self, name):
        alpha_gumbel, masks, soft_eff_ch = None, None, None
        if name in self.alpha_gumbels:
            alpha_gumbel = alpha_gumbel = self.alpha_gumbels[name]
        if name in self.ind2ch:
            soft_eff_ch = torch.sum(alpha_gumbel * self.ind2ch[name])
        if name in self.channel_masks:
            masks = self.channel_masks[name]
        return alpha_gumbel, soft_eff_ch, masks 

    def add_channel_alphas(self, block_name, out_range):
        step = out_range["step"]
        start = int(np.ceil(out_range["start"] / step) * step)
        stop = int(np.floor(out_range["stop"] / step) * step)
        no_steps = int((stop - start) / step + 1)

        mean = np.random.randint(low=0,high=no_steps)
        sd = 1

        x = np.arange(no_steps)
        var = sd**2
        alphas = 1*np.exp(-np.power(x-mean,2)/(2*var)) / ((2*math.pi*var)**.5)
        alphas = torch.Tensor(alphas)

        ch_range = np.linspace(start=start, stop=stop, num=no_steps, dtype=int, endpoint=True)

        channel_masks = []
        ind2ch = []
        for ch in ch_range:
            mask = [0 for j in range(out_range["stop"])]
            mask[0:ch] = [1 for j in range(ch)]
            channel_masks.append(mask)
            ind2ch.append(ch)

        self.channel_alphas[block_name] = alphas.to(self.device).requires_grad_(True)
        self.channel_masks[block_name] = torch.tensor(channel_masks, dtype=torch.float).to(self.device)
        self.ind2ch[block_name] = torch.tensor(ind2ch).to(self.device)
        self.out_range[block_name] = out_range

    def load_arch_state(self, arch_state):
        for block_name in arch_state:
            if "layer_alphas" in arch_state[block_name]:
                self.layer_alphas[block_name] = torch.tensor(arch_state[block_name]["layer_alphas"]).to(self.device).requires_grad_(True)
            if "channel_alphas" in arch_state[block_name]:
                self.channel_alphas[block_name] = torch.tensor(arch_state[block_name]["channel_alphas"]).to(self.device).requires_grad_(True)
                self.out_range[block_name] = out_range = arch_state[block_name]["out_range"]

                step = out_range["step"]
                start = int(np.ceil(out_range["start"] / step) * step)
                stop = int(np.floor(out_range["stop"] / step) * step)
                no_steps = int((stop - start) / step + 1)

                ch_range = np.linspace(start=start, stop=stop, num=no_steps, dtype=int, endpoint=True)

                channel_masks = []
                ind2ch = []
                for ch in ch_range:
                    mask = [0 for j in range(stop)]
                    mask[0:ch] = [1 for j in range(ch)]
                    channel_masks.append(mask)
                    ind2ch.append(ch)

                self.channel_masks[block_name] = torch.tensor(channel_masks, dtype=torch.float).to(self.device)
                self.ind2ch[block_name] = torch.tensor(ind2ch).to(self.device)

    def export_arch_state(self):
        out = {}
        for block_name in self.layer_alphas:
            out[block_name] = {"layer_alphas": self.layer_alphas[block_name].tolist()}
        for block_name in self.channel_alphas:
            out[block_name] = {"channel_alphas": self.channel_alphas[block_name].tolist(),
                                "out_range": self.out_range[block_name]}
        return out

class Dmask(nn.Module):
    def __init__(self, layer_name, in_channels, out_channels, out_range, nasmode=NASMODE.vanilla, hw_model=None, device=None):
        super().__init__()

        self.layer_name = layer_name
        self.device = device
        self.out_range = out_range
        self.nasmode = nasmode

        self.in_channels = in_channels

        if out_channels is None: 
            assert out_range is not None, 'both out_channels and out_range cannot be None'
            self.out_channels = self.out_range["stop"]
        else:
            assert out_range is None, 'either out_channels or out_range can be None'
            self.out_channels = out_channels

        self.soft_eff_channels = None

        self.alpha = None

        self.hw_model = hw_model

        self.bn = None
        self.relu = None 

    def forward(self, x, in_channels, alpha_gumbel=None, soft_eff_channels=None, masks=None):
        if self.nasmode == NASMODE.channelsearch:
            self.soft_eff_channels = soft_eff_channels
            out_channels = soft_eff_channels
            masks = torch.sum(torch.multiply(alpha_gumbel, masks.transpose(0, 1)), dim=1).reshape([1, -1] + [1 for i in range(2, len(x.shape))])

            x = torch.multiply(masks, self.cell(x))
        else:
            out_channels = self.out_channels
            x = self.cell(x)

        if self.bn is not None:
            x = self.bn(x)
   
        if self.relu is not None:
            x = self.relu(x)

        runtime, util = self.hw_model.runtime(self.cell, in_channels, out_channels, x.shape)

        return x, out_channels, runtime, util

    def get_arch_parameters(self):
        return self.alpha

    def set_arch_parameters(self, alpha):
        self.alpha = alpha

    def get_hard_eff_channels(self):
        if self.nasmode == NASMODE.channelsearch:
            step = self.out_range["step"]
            start = int(np.ceil(self.out_range["start"] / step) * step)
            stop = int(np.floor(self.out_range["stop"] / step) * step)
            no_steps = int((stop - start) / step + 1)

            ch_range = np.linspace(start=start, stop=stop, num=no_steps, dtype=int, endpoint=True)
            return ch_range[torch.argmax(self.alpha)].item()
        else:
            return self.out_channels

    def get_soft_eff_channels(self):
        return self.soft_eff_channels if self.nasmode == NASMODE.channelsearch else self.out_channels

    def convert_keras(self):
        raise NotImplementedError

    def print_layer(self):
        s = self.layer_type
        s += str(self.kernel_size) if hasattr(self, 'kernel_size') else ""
        s += "\t" + str(self.in_channels) + "->" + str(self.get_hard_eff_channels())
        return s

if __name__=="__main__":
    import matplotlib.pyplot as plt

    sd = 1
    mean = 10
    no_steps = 25
    
    x = np.arange(no_steps)
    var = sd**2
    norm_pdf = np.exp(-np.power(x-mean,2)/(2*var)) / ((2*math.pi*var)**.5)
    
    plt.plot(norm_pdf)
    plt.savefig("results/initialization.png")
