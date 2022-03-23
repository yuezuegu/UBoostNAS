

import torch
import torch.nn as nn
import numpy as np
import json

import src.search as search
import src.vanilla as vanilla

def closest(x, rng):
    closest_point = rng[0]
    min_dist = abs(x-rng[0])
    for r in rng:
        if abs(r-x) < min_dist:
            min_dist = abs(r-x)
            closest_point = r
    return closest_point

def closest_two(x, rng):
    if len(rng) == 1:
        return [rng[0], rng[0]]
    
    min1 = rng[0]
    min2 = rng[1]
    min_dist1 = abs(x-rng[0])
    min_dist2 = abs(x-rng[0])
    for r in rng:
        delta = abs(r-x)
        if delta < min_dist1:
            min_dist2 = min_dist1
            min_dist1 = delta
            min2 = min1
            min1 = r 
        elif delta < min_dist2:
            min_dist2 = delta
            min2 = r

    return sorted([min1, min2])

def _interpolate(x, y1, y2, x1, x2):
    if x1 == x2:
        return y1
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

def interpolate1d(ch, ref_dict):
    avail_out_ch = [int(k) for k in ref_dict.keys()]

    x1 = closest(ch, avail_out_ch)
    y1 = ref_dict[str(x1)]

    runtime_csim = y1["runtime_csim"]
    util_csim = y1["util_csim"]

    return {"runtime_csim": runtime_csim, "util_csim": util_csim}

def interpolate2d(in_ch, out_ch, ref_dict):
    avail_in_ch = [int(k) for k in ref_dict.keys()]

    x1 = closest(in_ch, avail_in_ch)
    if str(out_ch) in ref_dict[str(x1)]:
        val1 = ref_dict[str(x1)][str(out_ch)]
    else:
        val1 = interpolate1d(out_ch, ref_dict[str(x1)])
    runtime_csim = val1["runtime_csim"]
    util_csim = val1["util_csim"]
    
    return {"runtime_csim": runtime_csim, "util_csim": util_csim}

class BlackBox():
    def __init__(self, fname):
        with open(fname, "r") as infile:  
            self.table = json.load(infile)

    def get_measurements(self, layer, input_size, in_ch, out_ch, kernel_size):
        if layer not in ["Conv2d", "DepthwiseConv2d"]:
            raise NotImplementedError

        input_size_str = str(input_size)
        kernel_size_str = f"{kernel_size[0]}x{kernel_size[1]}"
        in_ch_str = str(in_ch)
        out_ch_str = str(out_ch)

        if layer in self.table:
            if input_size_str in self.table[layer]:
                if kernel_size_str in self.table[layer][input_size_str]:
                    if in_ch_str in self.table[layer][input_size_str][kernel_size_str]:
                        if out_ch_str in self.table[layer][input_size_str][kernel_size_str][in_ch_str]:
                            out = self.table[layer][input_size_str][kernel_size_str][in_ch_str][out_ch_str]
                            return out["runtime_csim"], out["util_csim"]
                        else:
                            out = interpolate1d(out_ch, self.table[layer][input_size_str][kernel_size_str][in_ch_str])
                            return out["runtime_csim"], out["util_csim"]
                    else:
                        out = interpolate2d(in_ch, out_ch, self.table[layer][input_size_str][kernel_size_str])
                        return out["runtime_csim"], out["util_csim"]

                else:
                    raise ValueError(f"kernel_size {kernel_size} not found in lookup table!")
            else:
                raise ValueError(f"input_size {input_size} not found in lookup table!")
        else:
            raise ValueError(f"Layer {layer} not found in lookup table!")




class smooth_ceil(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()

        self.B = 20
        self.C = 0.2
        self.v = 0.5
        self.K = self.C ** (1/self.v)
        self.no_steps = 300
        self.device = device
        self.offset = torch.arange(0, self.no_steps, step=1).to(device)

    def forward(self, x):
        if x > self.no_steps:
            print("smooth_ceil overflows!")

        x = x - self.offset
        x = torch.clamp(x, min=-2, max=2)
        return torch.sum(self.K/torch.pow(self.C + torch.exp(-self.B*x),1/self.v))        

class HardwareModel(nn.Module):
    def __init__(self, hw_type, batch_size, memory_bw, array_size, device, args):
        super().__init__()

        if hw_type not in ["flops", "systolic", "roofline", "blackbox"]:
            raise NotImplementedError

        self.args = args
        self.hw_type = hw_type
        self.clk_freq = args.clk_freq
        self.array_size = array_size
        self.device = device
        self.batch_size = batch_size
        self.memory_bw = memory_bw #GB/s

        self.smooth_ceil = smooth_ceil(device=device)

        if hw_type == "blackbox":
            self.blackbox = BlackBox("src/hardware/lookup.json")

    def runtime(self, cell, in_ch, out_ch, out_shape):
        if isinstance(cell, nn.Conv2d):
            if cell.groups == 1:
                return self.conv_runtime(in_ch, out_ch, out_shape, cell.kernel_size)
            else:
                return self.dw_conv_runtime(in_ch, out_ch, out_shape, cell.kernel_size)
        elif isinstance(cell, nn.Linear):
            return self.fc_runtime(in_ch, out_ch, out_shape)
        elif isinstance(cell, nn.Identity):
            return (0., 1.) #(runtime, util)
        elif isinstance(cell, search.layers.ZeroLayer):
            return (0., 1.) #(runtime, util)
        elif isinstance(cell, vanilla.layers.Zero):
            return (0., 1.) #(runtime, util)
        elif isinstance(cell, nn.Flatten):
            return (0., 1.) #(runtime, util)
        elif isinstance(cell, nn.MaxPool2d):
            return (0., 1.) #(runtime, util)
        else:
            raise NotImplementedError


    def dw_conv_runtime(self, in_ch, out_ch, y_shape, kernel_shape):
        # assert in_ch == out_ch, 'In depthwise conv2d, we assume in_channels is equal to out_channels'
        out_ch = in_ch
        
        filter_height = kernel_shape[0]
        filter_width = kernel_shape[1]
        out_window_height = y_shape[2]
        out_window_width = y_shape[3]
        no_macs = filter_height * filter_width * out_window_height * out_window_width * in_ch * self.batch_size

        if self.hw_type == "systolic":
            d2 = filter_height * filter_width

            no_tiles_d2 = self.smooth_ceil(d2 / self.array_size[0])
            no_tiles_d3 = out_ch

            runtime = no_tiles_d2 * no_tiles_d3 * self.batch_size * out_window_height * out_window_width / self.clk_freq * 1e3
            util = no_macs / runtime / self.peak_throughput() * 1e3

        elif self.hw_type == "flops":
            # assumes full utilization
            macs_per_second = self.array_size[0] * self.array_size[1] * self.clk_freq 
            runtime = no_macs / macs_per_second * 1e3
            util = no_macs / runtime / self.peak_throughput() * 1e3

        elif self.hw_type == "roofline":
            memory_size = 2*(filter_height * filter_width * in_ch + self.batch_size * out_window_height * out_window_width * in_ch + self.batch_size * filter_height * filter_width * in_ch) #Bytes
            
            memory_time = memory_size / (self.memory_bw * 1e9) * 1e3 #milliseconds

            macs_per_second = self.array_size[0] * self.array_size[1] * self.clk_freq 
            pe_runtime = no_macs / macs_per_second * 1e3

            binary_coef = torch.sigmoid(1e9*torch.tensor((memory_time - pe_runtime)).to(self.device))

            runtime = memory_time * binary_coef + pe_runtime * (1-binary_coef)
            util = no_macs / runtime / self.peak_throughput() * 1e3

        elif self.hw_type == "blackbox":
            assert out_window_height == out_window_width, "Non-square image size is not supported"
            runtime, util = self.blackbox.get_measurements(layer="DepthwiseConv2d", input_size=out_window_height, in_ch=in_ch, out_ch=out_ch, kernel_size=kernel_shape)

        else:
            raise NotImplementedError        

        return runtime, util


    def conv_runtime(self, in_ch, out_ch, y_shape, kernel_shape):
        filter_height = kernel_shape[0]
        filter_width = kernel_shape[1]
        out_window_height = y_shape[2]
        out_window_width = y_shape[3]
        no_macs = filter_height * filter_width * in_ch * self.batch_size * out_window_height * out_window_width * out_ch
        
        if self.hw_type == "systolic":
            d2 = filter_height * filter_width * in_ch

            no_tiles_d2 = self.smooth_ceil(d2 / self.array_size[0])
            no_tiles_d3 = self.smooth_ceil(out_ch / self.array_size[1])

            runtime = no_tiles_d2 * no_tiles_d3 * self.batch_size * out_window_height * out_window_width / self.clk_freq * 1e3
            util = no_macs / runtime / self.peak_throughput() * 1e3

        elif self.hw_type == "flops":
            # assumes full utilization
            macs_per_second = self.array_size[0] * self.array_size[1] * self.clk_freq 
            runtime = no_macs / macs_per_second * 1e3
            util = no_macs / runtime / self.peak_throughput() * 1e3

        elif self.hw_type == "roofline":
            memory_size = 2*(filter_height * filter_width * in_ch * out_ch + self.batch_size * out_window_height * out_window_width * in_ch + self.batch_size * filter_height * filter_width * out_ch) #Bytes
            
            memory_time = memory_size / (self.memory_bw * 1e9) * 1e3 #milliseconds

            macs_per_second = self.array_size[0] * self.array_size[1] * self.clk_freq 
            pe_runtime = no_macs / macs_per_second * 1e3

            binary_coef = torch.sigmoid(1e9*torch.tensor((memory_time - pe_runtime)).to(self.device))

            runtime = memory_time * binary_coef + pe_runtime * (1-binary_coef)
            util = no_macs / runtime / self.peak_throughput() * 1e3

        elif self.hw_type == "blackbox":
            assert out_window_height == out_window_width, "Non-square image size is not supported"
            runtime, util = self.blackbox.get_measurements(layer="Conv2d", input_size=out_window_height, in_ch=in_ch, out_ch=out_ch, kernel_size=kernel_shape)

        else:
            raise NotImplementedError

        return runtime, util

    def fc_runtime(self, in_ch, out_ch, y_shape):
        y_shape=[y_shape[0], y_shape[1], 1, 1]
        return self.conv_runtime(in_ch=in_ch, out_ch=out_ch, y_shape=y_shape, kernel_shape=(1,1))

    def get_utilization(self, in_ch, out_ch, kernel_shape):
        filter_height = kernel_shape[0]
        filter_width = kernel_shape[1]

        if self.hw_type == "systolic":
            d2 = filter_height * filter_width * in_ch
            no_tiles_d2 = self.smooth_ceil(d2 / self.array_size[0])
            no_tiles_d3 = self.smooth_ceil(out_ch / self.array_size[1])

            util = (d2 * out_ch) / ( no_tiles_d2 * no_tiles_d3 * self.array_size[0] * self.array_size[1] )
            return util
        elif self.hw_type == "flops":
            return 1.
        else:
            raise NotImplementedError

    def peak_throughput(self):
        return self.array_size[0] * self.array_size[1] * self.clk_freq





if __name__=="__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm

    sc_obj = smooth_ceil()
    ceil_fn = np.ceil

    d2 = 128
    rng = list(range(4,512,4))

    util_appr = []
    util_exact = []
    for out_ch in rng:
        no_tiles_d2 = sc_obj(d2 / 128)
        no_tiles_d3 = sc_obj(out_ch / 128)

        _util_appr = 100*(d2 * out_ch) / ( no_tiles_d2 * no_tiles_d3 * 128 * 128 )

        util_appr.append(_util_appr)

        no_tiles_d2 = np.ceil(d2 / 128)
        no_tiles_d3 = np.ceil(out_ch / 128)

        _util_exact = 100*(d2 * out_ch) / ( no_tiles_d2 * no_tiles_d3 * 128 * 128 )

        util_exact.append(_util_exact)

    from matplotlib import rc

    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':14})
    fig, ax = plt.subplots(figsize=(7, 5.5))

    plt.plot(rng, util_exact, linestyle='-', color='black', label="Exact")
    plt.plot(rng, util_appr, linestyle='--', linewidth=3, color='tab:red', label="Smooth")

    plt.ylabel("Utilization (%)")
    plt.xlabel("Number of output channels")
    plt.legend(frameon=False, loc="lower right")

    plt.savefig("results/smooth_ceil.png")
    plt.savefig("results/smooth_ceil.pdf")