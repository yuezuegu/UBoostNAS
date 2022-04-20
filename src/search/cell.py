
import torch.nn as nn

import src.search as search
import src.vanilla as vanilla
from src.search.dmask import Arch_State, NASMODE

class Cell(nn.Module):
    def __init__(self, layer_name, block_dict, in_channels, out_channels, out_range, arch_state, nasmode, in_maxpool, out_maxpool, hw_model, device):
        super().__init__()

        self.layer_name = layer_name
        self.nasmode = nasmode
        self.device = device
        self.hw_model = hw_model
        self.out_range = out_range
        self.in_maxpool = in_maxpool 
        self.out_maxpool = out_maxpool 

        self.in_channels = in_channels
        if out_channels is None: 
            assert out_range is not None, 'both out_channels and out_range cannot be None'
            self.out_channels = out_range["stop"]
        else:
            assert out_range is None, 'either out_channels or out_range can be None'
            self.out_channels = out_channels

        if block_dict is None:
            self.block_dict = nn.ModuleDict()

            self.block_dict["xp_pre"] = search.layers.Conv2d(layer_name=layer_name+"xp_pre", kernel_size=[3,3], in_channels=in_channels, out_channels=in_channels, nasmode=nasmode, hw_model=hw_model, device=device)
            if in_maxpool:
                self.block_dict["xpp_pre"] = search.layers.Conv2d(layer_name=layer_name+"xpp_pre", kernel_size=[3,3], in_channels=in_channels, out_channels=in_channels, stride=2, nasmode=nasmode, hw_model=hw_model, device=device)
            else:
                self.block_dict["xpp_pre"] = search.layers.Conv2d(layer_name=layer_name+"xpp_pre", kernel_size=[3,3], in_channels=in_channels, out_channels=in_channels, stride=1, nasmode=nasmode, hw_model=hw_model, device=device)

            self.block_dict["xp_t1"] = search.layers.Lmask(layer_name+"xp_t1", self.in_channels, nasmode, hw_model, device)
            self.block_dict["t1_t2"] = search.layers.Lmask(layer_name+"t1_t2", self.in_channels, nasmode, hw_model, device)
            self.block_dict["xp_t2"] = search.layers.Lmask(layer_name+"xp_t2", self.in_channels, nasmode, hw_model, device)
            self.block_dict["t2_t3"] = search.layers.Lmask(layer_name+"t2_t3", self.in_channels, nasmode, hw_model, device)
            self.block_dict["t1_t3"] = search.layers.Lmask(layer_name+"t1_t3", self.in_channels, nasmode, hw_model, device)
            self.block_dict["xp_t3"] = search.layers.Lmask(layer_name+"xp_t3", self.in_channels, nasmode, hw_model, device)

            self.block_dict["xpp_t1"] = search.layers.Lmask(layer_name+"xpp_t1", self.in_channels, nasmode, hw_model, device)
            self.block_dict["xpp_t2"] = search.layers.Lmask(layer_name+"xpp_t2", self.in_channels, nasmode, hw_model, device)
            self.block_dict["xpp_t3"] = search.layers.Lmask(layer_name+"xpp_t3", self.in_channels, nasmode, hw_model, device)

            self.block_dict["t1_add"] = search.layers.Add(layer_name+"t1_add", [self.block_dict[b].out_channels for b in ["xp_t1", "xpp_t1"]], hw_model, device)
            self.block_dict["t2_add"] = search.layers.Add(layer_name+"t2_add", [self.block_dict[b].out_channels for b in ["t1_t2", "xp_t2", "xpp_t2"]], hw_model, device)
            self.block_dict["t3_add"] = search.layers.Add(layer_name+"t3_add", [self.block_dict[b].out_channels for b in ["t2_t3", "t1_t3", "xp_t3", "xpp_t3"]], hw_model, device)

            if out_maxpool:
                self.block_dict["out_maxpool"] = search.layers.MaxPool(layer_name+"out_maxpool", self.block_dict["t3_add"].out_channels, kernel_size=2, stride=2, hw_model=hw_model, device=device)
        else:
            self.block_dict = block_dict

        if arch_state is None:
            self.arch_state = Arch_State(self.device)
            self.create_layer_alphas()
        else:
            self.arch_state = arch_state
        self.set_arch_parameters(self.arch_state)

    def forward(self, x_p, x_pp, x_p_ch, x_pp_ch):
        def gumbel(name):
            return self.arch_state.get_dmask_vars(name)
        # yb, ch_b, tb, ub = res
        
        all_results = {}

        all_results["res_xp_pre"] = x_p, x_p_ch, _, _ = self.block_dict["xp_pre"](x_p, x_p_ch, *gumbel("xp_pre"))
        all_results["res_xpp_pre"] = x_pp, x_pp_ch, _, _ = self.block_dict["xpp_pre"](x_pp, x_pp_ch, *gumbel("xpp_pre"))

        all_results["res_xp_t1"] = res_xp_t1 = self.block_dict["xp_t1"](x_p, x_p_ch, *gumbel("xp_t1"))
        all_results["res_xp_t2"] = res_xp_t2 = self.block_dict["xp_t2"](x_p, x_p_ch, *gumbel("xp_t2"))
        all_results["res_xp_t3"] = res_xp_t3 = self.block_dict["xp_t3"](x_p, x_p_ch, *gumbel("xp_t3"))
        all_results["res_xpp_t1"] = res_xpp_t1 = self.block_dict["xpp_t1"](x_pp, x_pp_ch, *gumbel("xpp_t1"))
        all_results["res_xpp_t2"] = res_xpp_t2 = self.block_dict["xpp_t2"](x_pp, x_pp_ch, *gumbel("xpp_t2"))
        all_results["res_xpp_t3"] = res_xpp_t3 = self.block_dict["xpp_t3"](x_pp, x_pp_ch, *gumbel("xpp_t3"))

        all_results["res_t1"] = res_t1 = self.block_dict["t1_add"]([res_xp_t1[0], res_xpp_t1[0]], [res_xp_t1[1], res_xpp_t1[1]])

        all_results["res_t1_t2"] = res_t1_t2 = self.block_dict["t1_t2"](res_t1[0], res_t1[1], *gumbel("t1_t2"))
        all_results["res_t1_t3"] = res_t1_t3 = self.block_dict["t1_t3"](res_t1[0], res_t1[1], *gumbel("t1_t3"))

        all_results["res_t2"] = res_t2 = self.block_dict["t2_add"]([res_t1_t2[0], res_xp_t2[0], res_xpp_t2[0]], [res_t1_t2[1], res_xp_t2[1], res_xpp_t2[1]])
        all_results["res_t2_t3"] = res_t2_t3 = self.block_dict["t2_t3"](res_t2[0], res_t2[1], *gumbel("t2_t3"))

        all_results["res_t3"] = res_t3 = self.block_dict["t3_add"]([res_t2_t3[0], res_t1_t3[0], res_xp_t3[0], res_xpp_t3[0]], [res_t2_t3[1], res_t1_t3[1], res_xp_t3[1], res_xpp_t3[1]]) 

        y, out_ch, _, _  = res_t3

        if self.out_maxpool:
            y, out_ch, _, _ = self.block_dict["out_maxpool"](y, out_ch, None)

        total_runtime = 0
        avg_util = 0 
        
        cnt = 0
        for k in all_results:
            t, u = all_results[k][2], all_results[k][3]
            if t > 0:
                total_runtime += t
                avg_util += u
                cnt += 1
        
        if cnt > 0:
            avg_util = avg_util / cnt

        return y, out_ch, total_runtime, avg_util

    def create_layer_alphas(self):
        for block_name, block in self.block_dict.items():
            if isinstance(block, search.layers.Lmask):
                if block.nasmode == NASMODE.microsearch:
                    self.arch_state.add_layer_alphas(block_name, len(block.cells))

    def create_channel_alphas(self, block_dict, out_range):
        for block_name, block in block_dict.items():
            if isinstance(block, search.layers.ConvBase) or isinstance(block, search.layers.SeparableConv) or isinstance(block, search.layers.Linear):
                if block.nasmode == NASMODE.channelsearch:
                    self.arch_state.add_channel_alphas(block_name, out_range)

    def get_arch_parameters(self):
        alphas = {}
        for name, block in self.block_dict.items():
            alpha = block.get_arch_parameters()
            if alpha is not None:
                alphas[name] = alpha
        return alphas

    def set_arch_parameters(self, arch_state):
        for name in arch_state.layer_alphas:
            self.block_dict[name].set_arch_parameters(arch_state.layer_alphas[name])

        for name in arch_state.channel_alphas:
            self.block_dict[name].set_arch_parameters(arch_state.channel_alphas[name])

    def convert_to_channel_search(self, x_p_ch, x_pp_ch, channel_range, in_maxpool, out_maxpool):
        new_dict = nn.ModuleDict()

        new_dict["xp_pre"] = search.layers.Conv2d(layer_name=self.layer_name+"xp_pre", kernel_size=[3,3], in_channels=x_p_ch, out_channels=None, out_range=channel_range, nasmode=NASMODE.channelsearch, hw_model=self.hw_model, device=self.device)
        if in_maxpool:
            new_dict["xpp_pre"] = search.layers.Conv2d(layer_name=self.layer_name+"xpp_pre", kernel_size=[3,3], in_channels=x_pp_ch, out_channels=None, out_range=channel_range, stride=2, nasmode=NASMODE.channelsearch, hw_model=self.hw_model, device=self.device)
        else:
            new_dict["xpp_pre"] = search.layers.Conv2d(layer_name=self.layer_name+"xpp_pre", kernel_size=[3,3], in_channels=x_pp_ch, out_channels=None, out_range=channel_range, stride=1, nasmode=NASMODE.channelsearch, hw_model=self.hw_model, device=self.device)

        new_dict["xp_t1"] = self.block_dict["xp_t1"].convert_to_channel_search(new_dict["xp_pre"].out_channels, channel_range)
        new_dict["xp_t2"] = self.block_dict["xp_t2"].convert_to_channel_search(new_dict["xp_pre"].out_channels, channel_range)
        new_dict["xp_t3"] = self.block_dict["xp_t3"].convert_to_channel_search(new_dict["xp_pre"].out_channels, channel_range)

        new_dict["xpp_t1"] = self.block_dict["xpp_t1"].convert_to_channel_search(new_dict["xpp_pre"].out_channels, channel_range)
        new_dict["xpp_t2"] = self.block_dict["xpp_t2"].convert_to_channel_search(new_dict["xpp_pre"].out_channels, channel_range)
        new_dict["xpp_t3"] = self.block_dict["xpp_t3"].convert_to_channel_search(new_dict["xpp_pre"].out_channels, channel_range)

        new_dict["t1_add"] = search.layers.Add(self.layer_name+"t1_add", [new_dict[b].out_channels for b in ["xp_t1", "xpp_t1"]], self.hw_model, self.device)

        new_dict["t1_t2"] = self.block_dict["t1_t2"].convert_to_channel_search(new_dict["t1_add"].out_channels, channel_range)
        new_dict["t2_add"] = search.layers.Add(self.layer_name+"t2_add", [new_dict[b].out_channels for b in ["t1_t2", "xp_t2", "xpp_t2"]], self.hw_model, self.device)

        new_dict["t2_t3"] = self.block_dict["t2_t3"].convert_to_channel_search(new_dict["t2_add"].out_channels, channel_range)
        new_dict["t1_t3"] = self.block_dict["t1_t3"].convert_to_channel_search(new_dict["t1_add"].out_channels, channel_range)
        
        new_dict["t3_add"] = search.layers.Add(self.layer_name+"t3_add", [new_dict[b].out_channels for b in ["t2_t3", "t1_t3", "xp_t3", "xpp_t3"]], self.hw_model, self.device)

        if out_maxpool:
            new_dict["out_maxpool"] = self.block_dict["out_maxpool"].convert_to_channel_search(new_dict["t3_add"].out_channels)
        
        self.arch_state = Arch_State(self.device)
        self.create_channel_alphas(new_dict, channel_range)

        return Cell(self.layer_name, new_dict, self.in_channels, None, channel_range, self.arch_state, NASMODE.channelsearch, in_maxpool, out_maxpool, self.hw_model, self.device)

    def convert_to_vanilla(self, x_p_ch, x_pp_ch, in_maxpool, out_maxpool):
        new_dict = nn.ModuleDict()

        new_dict["xp_pre"] = self.block_dict["xp_pre"].convert_to_vanilla(x_p_ch)
        new_dict["xpp_pre"] = self.block_dict["xpp_pre"].convert_to_vanilla(x_pp_ch)

        new_dict["xp_t1"]  = self.block_dict["xp_t1"].convert_to_vanilla(new_dict["xp_pre"].out_channels)
        new_dict["xp_t2"]  = self.block_dict["xp_t2"].convert_to_vanilla(new_dict["xp_pre"].out_channels)
        new_dict["xp_t3"]  = self.block_dict["xp_t3"].convert_to_vanilla(new_dict["xp_pre"].out_channels)

        new_dict["xpp_t1"]  = self.block_dict["xpp_t1"].convert_to_vanilla(new_dict["xpp_pre"].out_channels)
        new_dict["xpp_t2"]  = self.block_dict["xpp_t2"].convert_to_vanilla(new_dict["xpp_pre"].out_channels)
        new_dict["xpp_t3"]  = self.block_dict["xpp_t3"].convert_to_vanilla(new_dict["xpp_pre"].out_channels)

        new_dict["t1_add"] = vanilla.layers.Add([
            new_dict["xp_t1"],
            new_dict["xpp_t1"]])

        if new_dict["t1_add"].is_empty():
            new_dict["t1_t2"] = vanilla.layers.Zero() 
            new_dict["t1_t3"] = vanilla.layers.Zero() 
        else:
            new_dict["t1_t2"]  = self.block_dict["t1_t2"].convert_to_vanilla(new_dict["t1_add"].out_channels)
            new_dict["t1_t3"]  = self.block_dict["t1_t3"].convert_to_vanilla(new_dict["t1_add"].out_channels)

        new_dict["t2_add"] = vanilla.layers.Add([
            new_dict["t1_t2"], 
            new_dict["xp_t2"], 
            new_dict["xpp_t2"]])

        if new_dict["t2_add"].is_empty():
            new_dict["t2_t3"] = vanilla.layers.Zero()
        else:
            new_dict["t2_t3"] = self.block_dict["t2_t3"].convert_to_vanilla(new_dict["t2_add"].out_channels)
        
        # Prune graph backward
        if isinstance(new_dict["t2_t3"], vanilla.layers.Zero):
            new_dict["t1_t2"] = vanilla.layers.Zero()
            new_dict["xp_t2"] = vanilla.layers.Zero()
            new_dict["xpp_t2"] = vanilla.layers.Zero()
            new_dict["t2_add"] = vanilla.layers.Add([])

        if (isinstance(new_dict["t1_t3"], vanilla.layers.Zero) and 
            isinstance(new_dict["t1_t2"], vanilla.layers.Zero)):
            new_dict["xp_t1"] = vanilla.layers.Zero()
            new_dict["xpp_t1"] = vanilla.layers.Zero()
            new_dict["t1_add"] = vanilla.layers.Add([])

        if (isinstance(new_dict["xp_t1"], vanilla.layers.Zero) and
            isinstance(new_dict["xp_t2"], vanilla.layers.Zero) and
            isinstance(new_dict["xp_t3"], vanilla.layers.Zero)):
            new_dict["xp_pre"] = vanilla.layers.Zero()

        if (isinstance(new_dict["xpp_t1"], vanilla.layers.Zero) and
            isinstance(new_dict["xpp_t2"], vanilla.layers.Zero) and
            isinstance(new_dict["xpp_t3"], vanilla.layers.Zero)):
            new_dict["xpp_pre"] = vanilla.layers.Zero()

        new_dict["t3_add"] = vanilla.layers.Add([
            new_dict["t2_t3"], 
            new_dict["t1_t3"], 
            new_dict["xp_t3"], 
            new_dict["xpp_t3"]]) 

        assert new_dict["t3_add"].is_empty() == False, "Microarchitecture is empty"

        if out_maxpool:
            new_dict["out_maxpool"] = self.block_dict["out_maxpool"].convert_to_vanilla(new_dict["t3_add"].out_channels)

        return vanilla.cell.Cell(new_dict, in_channels=self.block_dict["xp_pre"].in_channels, out_channels=new_dict["t3_add"].out_channels)

    def get_hard_eff_channels(self):
        out = {}
        for name, block in self.block_dict.items():
            eff_ch = block.get_hard_eff_channels()
            if eff_ch is not None:
                out[name] = eff_ch
        return out
        
    def get_soft_eff_channels(self):
        out = {}
        for name, block in self.block_dict.items():
            eff_ch = block.get_soft_eff_channels()
            if eff_ch is not None:
                out[name] = eff_ch
        return out

    def convert_keras(self, xp, xpp, layer_name=None):
        xp = self.block_dict["xp_pre"].convert_keras(xp, layer_name=layer_name+"_xp_pre")
        xpp = self.block_dict["xpp_pre"].convert_keras(xpp, layer_name=layer_name+"_xpp_pre")

        xp_t1 = self.block_dict["xp_t1"].convert_keras(xp, layer_name=layer_name+"_xp_t1")
        xp_t2 = self.block_dict["xp_t2"].convert_keras(xp, layer_name=layer_name+"_xp_t2")
        xp_t3 = self.block_dict["xp_t3"].convert_keras(xp, layer_name=layer_name+"_xp_t3")

        xpp_t1 = self.block_dict["xpp_t1"].convert_keras(xpp, layer_name=layer_name+"_xpp_t1")
        xpp_t2 = self.block_dict["xpp_t2"].convert_keras(xpp, layer_name=layer_name+"_xpp_t2")
        xpp_t3 = self.block_dict["xpp_t3"].convert_keras(xpp, layer_name=layer_name+"_xpp_t3")

        t1_add = search.layers.Add(self.layer_name+"t1_add", [xp_t1.shape[3], xpp_t1.shape[3]], self.hw_model, self.device).convert_keras([xp_t1, xpp_t1])
        t1_t2 = self.block_dict["t1_t2"].convert_keras(t1_add, layer_name=layer_name+"_t1_t2")
        t1_t3 = self.block_dict["t1_t3"].convert_keras(t1_add, layer_name=layer_name+"_t1_t3")

        t2_add = search.layers.Add(self.layer_name+"t2_add", [xp_t2.shape[3], xpp_t2.shape[3], t1_t2.shape[3]], self.hw_model, self.device).convert_keras([xp_t2, xpp_t2, t1_t2])
        t2_t3 = self.block_dict["t2_t3"].convert_keras(t2_add, layer_name=layer_name+"_t2_t3")

        out = search.layers.Add(self.layer_name+"t3_add", [xp_t3.shape[3], xpp_t3.shape[3], t1_t3.shape[3], t2_t3.shape[3]], self.hw_model, self.device).convert_keras([xp_t3, xpp_t3, t1_t3, t2_t3])

        if self.out_maxpool:
            out = self.block_dict["out_maxpool"].convert_keras(out, layer_name=layer_name+"_out_maxpool")

        return out

    def print_layer(self):
        s = ""
        for i in ["xp_t1", "xp_t2", "xp_t3", "xpp_t1", "xpp_t2", "xpp_t3", "t1_t2", "t1_t3", "t2_t3"]:
            s += "\n" + i + ":\t" +  self.block_dict[i].print_layer()
        return s

    def new(self):
        return self.__class__(self.layer_name, self.block_dict, self.in_channels, self.nasmode, self.hw_model, self.device)
