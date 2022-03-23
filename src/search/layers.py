
import torch 
import torch.nn as nn

import tensorflow as tf

import numpy as np

from src.search.dmask import Dmask, NASMODE
import src.vanilla as vanilla

class ConvBase(Dmask):
    def __init__(self, layer_name, kernel_size, in_channels, out_channels: int = None, out_range: dict = None, batchnorm=True, relu=True, groups=1, dilation=1, stride=1, nasmode=NASMODE.vanilla, hw_model=None, device='cpu'):
        super().__init__(layer_name, in_channels, out_channels, out_range, nasmode, hw_model, device)

        if dilation > 1:
            padding = "same"
        else:
            padding = int(np.floor((kernel_size[0]-1)/2))

        self.cell = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation).to(device)
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation 
        self.padding = padding 
        self.bn = nn.BatchNorm2d(self.out_channels).to(device) if batchnorm else None
        self.relu = nn.ReLU().to(device) if relu else None

class Conv2d(ConvBase):
    def __init__(self, layer_name, kernel_size, in_channels, out_channels: int = None, out_range: dict = None, batchnorm=True, relu=True, stride=1, nasmode=NASMODE.vanilla, hw_model=None, device='cpu'):
        self.layer_type = "Conv2d"
        super().__init__(layer_name, kernel_size, in_channels, out_channels=out_channels, out_range=out_range, batchnorm=batchnorm, relu=relu, groups=1, dilation=1, stride=stride, nasmode=nasmode, hw_model=hw_model, device=device)

    def convert_to_channel_search(self, in_channels, channel_range):
        return Conv2d(
            layer_name=self.layer_name,
            kernel_size=self.kernel_size,
            in_channels=in_channels,
            out_channels=None,
            out_range=channel_range,
            batchnorm=self.bn,
            relu=self.relu,
            stride=self.stride,
            nasmode=NASMODE.channelsearch,
            hw_model=self.hw_model,
            device=self.device
        )

    def convert_to_vanilla(self, in_channels):
        out_channels = self.get_hard_eff_channels()

        return vanilla.layers.Conv2d(self.kernel_size, in_channels, out_channels, self.bn != None, self.relu != None, self.stride)

    def convert_keras(self, x, layer_name=None):
        x = tf.keras.layers.Conv2D(
            filters=self.get_hard_eff_channels(),
            kernel_size=self.kernel_size,
            strides=(self.stride, self.stride),
            padding="same",
            activation=None,
            use_bias=True,
            name=layer_name
        )(x)

        if self.bn is not None:
            x = tf.keras.layers.BatchNormalization()(x)
        if self.relu is not None:
            x = tf.keras.layers.ReLU()(x)
        return x

    def new(self):
        return self.__class__(self.layer_name, self.kernel_size, self.in_channels, self.out_channels, self.out_range, self.bn, self.relu, self.stride, self.nasmode, self.hw_model, self.device)

class DepthwiseConv2d(ConvBase):
    def __init__(self, layer_name, kernel_size, in_channels, out_channels: int = None, out_range: dict = None, batchnorm=True, relu=True, nasmode=NASMODE.vanilla, hw_model=None, device='cpu'):
        self.layer_type = "DepthwiseConv2d"
        super().__init__(layer_name, kernel_size, in_channels, out_channels=out_channels, out_range=out_range, batchnorm=batchnorm, relu=relu, groups=in_channels, dilation=1, nasmode=nasmode, hw_model=hw_model, device=device)

    def convert_to_channel_search(self, in_channels, channel_range):
        return DepthwiseConv2d(
            layer_name=self.layer_name,
            kernel_size=self.kernel_size, 
            in_channels=in_channels, 
            out_channels=in_channels, #out_channels must be equal to in_channels
            out_range=None, 
            batchnorm=self.bn, 
            relu=self.relu, 
            nasmode=NASMODE.vanilla,
            hw_model=self.hw_model,
            device=self.device)

    def convert_to_vanilla(self, in_channels):
        out_channels = in_channels

        return vanilla.layers.DepthwiseConv2d(self.kernel_size, in_channels, out_channels, self.bn != None, self.relu != None, self.stride)


    def convert_keras(self, x, layer_name=None):
        x = tf.keras.layers.DepthwiseConv2D(
                kernel_size=self.kernel_size,
                strides=(1, 1),
                depth_multiplier=1,
                padding="same",
                activation=None,
                use_bias=True,
                name=layer_name
                )(x)

        if self.bn is not None:
            x = tf.keras.layers.BatchNormalization()(x)
        if self.relu is not None:
            x = tf.keras.layers.ReLU()(x)
        return x

    def new(self):
        return self.__class__(self.layer_name, self.kernel_size, self.in_channels, self.out_channels, self.out_range, self.bn, self.relu, self.stride, self.nasmode, self.hw_model, self.device)

class DilatedConv2d(ConvBase):
    def __init__(self, layer_name, kernel_size, in_channels, out_channels: int = None, out_range: dict = None, batchnorm=True, relu=True, nasmode=NASMODE.vanilla, hw_model=None, dilation=2, device='cpu'):
        self.layer_type = "DilatedConv2d"
        super().__init__(layer_name, kernel_size, in_channels, out_channels=out_channels, out_range=out_range, batchnorm=batchnorm, relu=relu, dilation=dilation, nasmode=nasmode, hw_model=hw_model, device=device)

    def convert_to_channel_search(self, in_channels, channel_range):
        return DilatedConv2d(
            layer_name=self.layer_name,
            kernel_size=self.kernel_size, 
            in_channels=in_channels, 
            out_channels=None, 
            out_range=channel_range, 
            batchnorm=self.bn, 
            relu=self.relu, 
            nasmode=NASMODE.channelsearch,
            dilation=self.dilation,
            hw_model=self.hw_model,
            device=self.device)

    def convert_to_vanilla(self, in_channels):
        out_channels = self.get_hard_eff_channels()

        return vanilla.layers.DilatedConv2d(self.kernel_size, in_channels, out_channels, self.bn != None, self.relu != None, self.stride, self.dilation)

    def convert_keras(self, x, layer_name=None):
        x = tf.keras.layers.Conv2D(
            filters=self.get_hard_eff_channels(),
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation,
            strides=(1, 1),
            padding="same",
            activation=None,
            use_bias=True,
            name=layer_name)(x)
        if self.bn is not None:
            x = tf.keras.layers.BatchNormalization()(x)
        if self.relu is not None:
            x = tf.keras.layers.ReLU()(x)
        return x

    def new(self):
        return self.__class__(self.layer_name, self.kernel_size, self.in_channels, self.out_channels, self.out_range, self.bn, self.relu, self.stride, self.nasmode, self.hw_model, self.dilation, self.device)

class SeparableConv(nn.Module):
    def __init__(self, layer_name, block_dict, kernel_size, in_channels, out_channels: int = None, out_range: dict = None, batchnorm=True, relu=True, nasmode=NASMODE.vanilla, hw_model=None, device='cpu') -> None:
        super().__init__()
        self.layer_type = "SeparableConv"

        self.layer_name = layer_name
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_range = out_range

        if out_channels is None: 
            assert out_range is not None, 'both out_channels and out_range cannot be None'
            self.out_channels = self.out_range["stop"]
        else:
            assert out_range is None, 'either out_channels or out_range can be None'
            self.out_channels = out_channels

        self.batchnorm = batchnorm
        self.relu = relu 
        self.nasmode = nasmode
        self.hw_model = hw_model
        self.device = device 

        self.block_dict = nn.ModuleDict()
        if block_dict is None:
            self.block_dict["depthwise"] = DepthwiseConv2d(layer_name+"dw", kernel_size, in_channels, in_channels, out_range=None, batchnorm=True, relu=True, nasmode=NASMODE.vanilla, hw_model=hw_model, device=device)
            self.block_dict["pointwise"] = Conv2d(layer_name+"pw", [1,1], in_channels, out_channels, out_range, batchnorm=True, relu=True, nasmode=nasmode, hw_model=hw_model, device=device)
        else:
            self.block_dict = block_dict

    def forward(self, x, in_channels, alpha_gumbel=None, soft_eff_channels=None, masks=None):
        x, _, runtime0, util0 = self.block_dict["depthwise"](x, in_channels, alpha_gumbel, soft_eff_channels, masks)
        x, out_channels, runtime1, util1 = self.block_dict["pointwise"](x, in_channels, alpha_gumbel, soft_eff_channels, masks)

        return x, out_channels, runtime0+runtime1, (util0+util1)/2

    def get_arch_parameters(self):
        return self.block_dict["pointwise"].get_arch_parameters()

    def set_arch_parameters(self, alpha):
        return self.block_dict["pointwise"].set_arch_parameters(alpha)

    def get_hard_eff_channels(self):
        return self.block_dict["pointwise"].get_hard_eff_channels()

    def get_soft_eff_channels(self):
        return self.block_dict["pointwise"].get_soft_eff_channels()

    def convert_keras(self, x, layer_name=None):
        if layer_name is None:
            x = self.block_dict["depthwise"].convert_keras(x, layer_name)
            return self.block_dict["pointwise"].convert_keras(x, layer_name)
        else:
            x = self.block_dict["depthwise"].convert_keras(x, layer_name+"_dwc")
            return self.block_dict["pointwise"].convert_keras(x, layer_name+"_pwc")            

    def convert_to_channel_search(self, in_channels, channel_range):
        new_block = nn.ModuleDict()
        new_block["depthwise"] = self.block_dict["depthwise"].convert_to_channel_search(in_channels, channel_range)
        new_block["pointwise"] = self.block_dict["pointwise"].convert_to_channel_search(in_channels, channel_range)
        return SeparableConv(self.layer_name, new_block, self.kernel_size, in_channels, out_channels=None, out_range=channel_range, batchnorm=self.batchnorm, relu=self.relu, nasmode=NASMODE.channelsearch, hw_model=self.hw_model, device=self.device)

    def convert_to_vanilla(self, in_channels):
        return vanilla.layers.SeparableConv(self.kernel_size, in_channels, self.get_hard_eff_channels())

    def print_layer(self):
        return self.block_dict["depthwise"].print_layer() + "\t" + self.block_dict["pointwise"].print_layer()

class Linear(Dmask):
    def __init__(self, layer_name, in_channels, out_channels: int = None, out_range: dict = None, batchnorm=True, relu=True, nasmode=NASMODE.vanilla, hw_model=None, device='cpu'):
        self.layer_type = "Linear"
        super().__init__(layer_name, in_channels, out_channels, out_range, nasmode, hw_model, device)

        self.cell = nn.Linear(in_features=self.in_channels, out_features=self.out_channels).to(device)

        self.bn = nn.BatchNorm2d(self.out_channels) if batchnorm else None
        self.relu = nn.ReLU() if relu else None

    def convert_to_channel_search(self, in_channels, channel_range):
        return Linear(
            layer_name=self.layer_name,
            in_channels=in_channels, 
            out_channels=None, 
            out_range=channel_range, 
            batchnorm=self.bn, 
            relu=self.relu, 
            nasmode=NASMODE.channelsearch,
            hw_model=self.hw_model,
            device=self.device)

    def convert_to_vanilla(self, in_channels):
        out_channels = self.get_hard_eff_channels()
        return vanilla.layers.Linear(in_channels, out_channels, self.bn != None, self.relu != None)

    def convert_keras(self, x, layer_name=None):
        x = tf.keras.layers.Dense(units=self.get_hard_eff_channels(), activation=None, use_bias=True, name=layer_name)(x)
        if self.bn is not None:
            x = tf.keras.layers.BatchNormalization()(x)
        if self.relu is not None:
            x = tf.keras.layers.ReLU()(x)
        return x

    def new(self):
        return self.__class__(self.layer_name, self.in_channels, self.out_channels, self.out_range, self.bn, self.relu, self.nasmode, self.hw_model, self.device)

class Identity(Dmask):
    def __init__(self, layer_name, in_channels, hw_model, device):
        self.layer_type = "Identity"
        self.in_channels = in_channels
        self.out_channels = in_channels

        super().__init__(layer_name, self.in_channels, self.out_channels, out_range=None, nasmode=NASMODE.vanilla, hw_model=hw_model, device=device)

        self.cell = nn.Identity().to(device)

    def convert_to_channel_search(self, in_channels=None, channel_range=None):
        self.in_channels = in_channels
        return self.__class__(self.layer_name, in_channels, self.hw_model, self.device)

    def convert_to_vanilla(self, in_channels):
        return vanilla.layers.Identity(in_channels)

    def convert_keras(self, x, layer_name=None):
        return tf.keras.layers.Lambda(lambda x: x)(x)

    def new(self):
        return self.__class__(self.layer_name, self.in_channels, self.hw_model, self.device).to(self.device)

class ZeroLayer(nn.Module):
    def __init__(self, layer_name, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.layer_name = layer_name
        
    def forward(self, x):
        return 0*x

class Zero(Dmask):
    def __init__(self, layer_name, in_channels, hw_model, device):
        self.layer_type = "Zero"
        self.in_channels = in_channels
        self.out_channels = in_channels

        super().__init__(layer_name, self.in_channels, self.out_channels, out_range=None, nasmode=NASMODE.vanilla, hw_model=hw_model, device=device)

        self.cell = ZeroLayer(layer_name=layer_name, in_channels=in_channels).to(device)

    def convert_to_channel_search(self, in_channels=None, channel_range=None):
        return self.__class__(self.layer_name, in_channels, self.hw_model, self.device)

    def convert_to_vanilla(self, in_channels):
        return vanilla.layers.Zero()

    def convert_keras(self, x, layer_name=None):
        return tf.keras.layers.Lambda(lambda x: 0*x)(x)

    def new(self):
        return self.__class__(self.layer_name, self.in_channels, self.hw_model, self.device)

class Flatten(Dmask):
    def __init__(self, layer_name, in_channels, multiplier, hw_model, device):
        self.layer_type = "Flatten"

        self.multiplier = int(multiplier)
        self.in_channels = in_channels
        self.out_channels = in_channels * self.multiplier ** 2
        super().__init__(layer_name, in_channels, self.out_channels, out_range=None, nasmode=NASMODE.vanilla, hw_model=hw_model, device=device)

        self.cell = nn.Flatten().to(device)

    def convert_to_channel_search(self, in_channels=None, channel_range=None):
        return self.__class__(self.layer_name, in_channels, self.hw_model, self.device)

    def convert_to_vanilla(self, in_channels):
        return vanilla.layers.Flatten(in_channels, self.multiplier)

    def convert_keras(self, x, layer_name=None):
        return tf.keras.layers.Flatten()(x)

    def new(self):
        return self.__class__(self.layer_name, self.in_channels, self.multiplier, self.hw_model, self.device)

class Add(nn.Module):
    def __init__(self, layer_name, in_channels, hw_model, device):
        self.layer_type = "Add"

        max_channel = max(in_channels)
        self.in_channels = max_channel
        self.out_channels = max_channel
        self.nasmode = NASMODE.vanilla
        self.layer_name = layer_name

        super().__init__()

        self.module_list = nn.ModuleList()
        for i, x in enumerate(in_channels):
            if x < max_channel:
                self.module_list.append(
                    Conv2d(
                        layer_name=layer_name+"_op"+str(i),
                        kernel_size=[1,1], 
                        in_channels=x,
                        out_channels=max_channel,
                        out_range=None,
                        batchnorm=True,
                        relu=True,
                        nasmode=NASMODE.vanilla,
                        hw_model=hw_model,
                        device=device
                        )
                    )
            else:
                self.module_list.append(Identity(layer_name=layer_name+"_op"+str(i), in_channels=x, hw_model=hw_model, device=device))

    def forward(self, x: list, in_channels: list):
        runtime = 0
        util = 0

        y = []
        for i in range(0, len(self.module_list)):
            _y, _, _t, _u = self.module_list[i](x[i], in_channels[i])
            y.append(_y)

            if _t > 0:
                runtime += _t
                util += _u

        y = torch.sum(torch.stack(y), dim=0)

        return y, self.out_channels, runtime, util

    def get_arch_parameters(self):
        return None

    def get_soft_eff_channels(self):
        return None 
    
    def get_hard_eff_channels(self):
        return None 

    def convert_keras(self, x, layer_name=None):
        y = []
        for i in range(0, len(self.module_list)):
            y.append(self.module_list[i].convert_keras(x[i]))

        return tf.keras.layers.Add()(y)

class MaxPool(Dmask):
    def __init__(self, layer_name, in_channels, kernel_size=2, stride=2, hw_model=None, device='cpu'):
        self.layer_type = "MaxPool"

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride

        super().__init__(layer_name, in_channels, self.out_channels, out_range=None, nasmode=NASMODE.vanilla, hw_model=hw_model, device=device)

        self.cell = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride)

    def convert_to_channel_search(self, in_channels=None, channel_range=None):
        return self.__class__(self.layer_name, in_channels, self.kernel_size, self.stride, self.hw_model, self.device)

    def convert_to_vanilla(self, in_channels=None):
        return vanilla.layers.MaxPool(in_channels, kernel_size=self.kernel_size, stride=self.stride)

    def convert_keras(self, x, layer_name=None):
        return tf.keras.layers.MaxPool2D(pool_size=self.kernel_size, strides=self.stride)(x)

    def new(self):
        return self.__class__(self.layer_name, self.in_channels, self.kernel_size, self.stride, self.hw_model, self.device)

class Lmask(nn.Module):
    def __init__(self, layer_name, in_channels, nasmode, hw_model, device):
        super().__init__()

        self.layer_name = layer_name
        conv2d_3x3 = Conv2d(layer_name=layer_name+"_conv2d_3x3", kernel_size=[3,3], in_channels=in_channels, out_channels=in_channels, nasmode=nasmode, hw_model=hw_model, device=device)
        conv2d_5x5 = Conv2d(layer_name=layer_name+"_conv2d_5x5", kernel_size=[5,5], in_channels=in_channels, out_channels=in_channels, nasmode=nasmode, hw_model=hw_model, device=device)
        dw_3x3 = SeparableConv(layer_name=layer_name+"_dw_3x3", block_dict=None, kernel_size=[3,3], in_channels=in_channels, out_channels=in_channels, nasmode=nasmode, hw_model=hw_model, device=device)
        dw_5x5 = SeparableConv(layer_name=layer_name+"_dw_5x5", block_dict=None, kernel_size=[5,5], in_channels=in_channels, out_channels=in_channels, nasmode=nasmode, hw_model=hw_model, device=device)
        dil_3x3 = DilatedConv2d(layer_name=layer_name+"_dil_3x3", kernel_size=[3,3], in_channels=in_channels, out_channels=in_channels, dilation=2, nasmode=nasmode, hw_model=hw_model, device=device)
        dil_5x5 = DilatedConv2d(layer_name=layer_name+"_dil_5x5", kernel_size=[5,5], in_channels=in_channels, out_channels=in_channels, dilation=2, nasmode=nasmode, hw_model=hw_model, device=device)
        identity = Identity(layer_name=layer_name+"_identity", in_channels=in_channels, hw_model=hw_model, device=device)
        zero = Zero(layer_name=layer_name+"_zero", in_channels=in_channels, hw_model=hw_model, device=device)

        self.cells = nn.ModuleDict({
            "conv2d_3x3": conv2d_3x3, 
            "conv2d_5x5": conv2d_5x5, 
            "dw_3x3": dw_3x3,
            "dw_5x5": dw_5x5, 
            "dil_3x3": dil_3x3, 
            "dil_5x5": dil_5x5,
            "identity": identity,
            "zero": zero 
        })

        self.device = device
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hw_model = hw_model
        self.nasmode = nasmode 

        self.alpha = None

    def forward(self, x, in_channels, alpha_gumbel, soft_eff_channels, masks):
        runtime = 0
        util = 0

        out = []
        denom = 0
        for i, cell in enumerate(self.cells.values()):
            _x, _, _runtime, _util = cell(x, in_channels, alpha_gumbel, soft_eff_channels, masks)
            out.append(alpha_gumbel[i] * _x)

            if _runtime > 0:
                runtime += alpha_gumbel[i] * _runtime
                util += alpha_gumbel[i] * _util
                denom += alpha_gumbel[i]

        x = torch.sum(torch.stack(out), dim=0)

        runtime = runtime / denom
        util = util / denom        

        return x, self.out_channels, runtime, util


    def get_arch_parameters(self):
        return self.alpha

    def set_arch_parameters(self, alpha):
        self.alpha = alpha

    def get_hard_eff_channels(self):
        return self.out_channels

    def get_soft_eff_channels(self):
        return self.out_channels

    def get_final_arch(self):
        selected_ind = torch.argmax(self.alpha)
        selected_key = list(self.cells.keys())[selected_ind]
        return self.cells[selected_key]

    def convert_to_channel_search(self, in_channels, channel_range):
        selected_cell = self.get_final_arch()
        return selected_cell.convert_to_channel_search(in_channels, channel_range)

    def convert_to_vanilla(self, in_channels):
        selected_cell = self.get_final_arch()
        return selected_cell.convert_to_vanilla(in_channels)

    def convert_keras(self, x, layer_name=None):
        return self.get_final_arch().convert_keras(x)

    def new(self):
        return self.__class__(self.layer_name, self.in_channels, self.nasmode, self.hw_model, self.device)

    def print_layer(self):
        return "Lmask"

