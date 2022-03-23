

import torch 
import torch.nn as nn
import tensorflow as tf

import numpy as np

class Conv2d(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, batchnorm=True, relu=True, stride=1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.padding = int(np.floor((kernel_size[0]-1)/2))

        self.stride = stride 

        self.conv = nn.Conv2d(in_channels, out_channels, self.kernel_size, stride=self.stride, padding=self.padding, groups=1, dilation=1)
        self.bn = nn.BatchNorm2d(self.out_channels) if batchnorm else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def convert_keras(self, x, layer_name=None):
        x = tf.keras.layers.Conv2D(
            filters=self.out_channels,
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

    def print_layer(self):
        s = self.__class__.__name__
        s += str(self.kernel_size)
        s += "\t" + str(self.in_channels) + "->" + str(self.out_channels)
        return s


class DepthwiseConv2d(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, batchnorm=True, relu=True, stride=1):
        super().__init__()
        
        assert in_channels == out_channels, "out_channels cannot be different than in_channels"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride 

        self.conv = nn.Conv2d(in_channels, out_channels, self.kernel_size, stride=self.stride, padding="same", groups=out_channels, dilation=1)
        self.bn = nn.BatchNorm2d(self.out_channels) if batchnorm else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def convert_keras(self, x, layer_name=None):
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=(self.stride, self.stride),
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

    def print_layer(self):
        s = self.__class__.__name__
        s += str(self.kernel_size)
        s += "\t" + str(self.in_channels) + "->" + str(self.out_channels)
        return s

class DilatedConv2d(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, batchnorm=True, relu=True, stride=1, dilation=2):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.padding = "same"

        self.conv = nn.Conv2d(in_channels, out_channels, self.kernel_size, stride=self.stride, padding=self.padding, groups=1, dilation=self.dilation)
        self.bn = nn.BatchNorm2d(self.out_channels) if batchnorm else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def convert_keras(self, x, layer_name=None):
        x = tf.keras.layers.Conv2D(
            filters=self.out_channels,
            kernel_size=self.kernel_size,
            strides=(self.stride, self.stride),
            dilation_rate=self.dilation,
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

    def print_layer(self):
        s = self.__class__.__name__
        s += str(self.kernel_size)
        s += "\t" + str(self.in_channels) + "->" + str(self.out_channels)
        return s

class SeparableConv(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.depthwise = DepthwiseConv2d(kernel_size, in_channels, in_channels, batchnorm=True, relu=True)
        self.pointwise = Conv2d(kernel_size=[1,1], in_channels=in_channels, out_channels=out_channels, batchnorm=True, relu=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

    def convert_keras(self, x, layer_name=None):
        x = self.depthwise.convert_keras(x, layer_name+"_dwc")
        x = self.pointwise.convert_keras(x, layer_name+"_pwc")
        return x

    def print_layer(self):
        return self.depthwise.print_layer() + "\t" + self.pointwise.print_layer()

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=True, relu=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(self.out_channels) if batchnorm else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def convert_keras(self, x, layer_name=None):
        x = tf.keras.layers.Dense(units=self.out_channels, activation=None, use_bias=True, name=layer_name)(x)
        if self.bn is not None:
            x = tf.keras.layers.BatchNormalization()(x)
        if self.relu is not None:
            x = tf.keras.layers.ReLU()(x)
        return x

    def print_layer(self):
        s = self.__class__.__name__
        s += "\t" + str(self.in_channels) + "->" + str(self.out_channels)
        return s

class Identity(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels

        self.identity = nn.Identity()
    
    def forward(self, x):
        return self.identity(x)

    def convert_keras(self, x, layer_name=None):
        return tf.keras.layers.Lambda(lambda x: x)(x)

    def print_layer(self):
        s = self.__class__.__name__
        s += "\t" + str(self.in_channels) + "->" + str(self.out_channels)
        return s

class Zero(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = None
        self.out_channels = None
        
    def forward(self, x):
        return None

    def convert_keras(self, x, layer_name=None):
        return None

    def print_layer(self):
        s = self.__class__.__name__
        # s += "\t" + str(self.in_channels) + "->" + str(self.out_channels)
        return s

class Flatten(nn.Module):
    def __init__(self, in_channels, multiplier):
        super().__init__()

        self.multiplier = int(multiplier)
        self.in_channels = in_channels
        self.out_channels = in_channels * self.multiplier ** 2
        
        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.flatten(x)

    def convert_keras(self, x, layer_name=None):
        return tf.keras.layers.Flatten()(x)

    def print_layer(self):
        s = self.__class__.__name__
        s += "\t" + str(self.in_channels) + "->" + str(self.out_channels)
        return s


class Add(nn.Module):
    def __init__(self, in_modules):
        super().__init__()

        self.module_list = nn.ModuleList()

        in_channels = [m.out_channels for m in in_modules if m.out_channels is not None]

        if len(in_channels) > 0:
            max_channel = max(in_channels)
            self.in_channels = max_channel
            self.out_channels = max_channel
            for m in in_modules:
                if m.out_channels is None or m.out_channels==max_channel:
                    self.module_list.append(Identity(in_channels=self.in_channels))
                else:
                    self.module_list.append(Conv2d(kernel_size=[1,1], in_channels=m.out_channels, out_channels=self.out_channels, batchnorm=True, relu=True))
        else:
            self.in_channels = None
            self.out_channels = None

    def forward(self, x):
        if len(self.module_list) > 0:
            return torch.sum(torch.stack([self.module_list[i](x[i]) for i in range(len(self.module_list)) if x[i] is not None]), dim=0)
        else:
            return None

    def convert_keras(self, x, layer_name=None):
        y = []
        for i in range(0, len(self.module_list)):
            if x[i] is not None:
                y.append(self.module_list[i].convert_keras(x[i]))

        if len(y) > 0:
            return tf.keras.layers.Add()(y)
        else:
            return None

    def is_empty(self):
        return len(self.module_list) == 0

class MaxPool(nn.Module):
    def __init__(self, in_channels, kernel_size=2, stride=2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.maxpool = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        return self.maxpool(x)

    def convert_keras(self, x, layer_name=None):
        return tf.keras.layers.MaxPool2D(pool_size=self.kernel_size, strides=self.stride)(x)

    def print_layer(self):
        s = self.__class__.__name__
        s += "\t" + str(self.in_channels) + "->" + str(self.out_channels)
        return s


class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = [3,3]
        self.conv = Conv2d(self.kernel_size, self.in_channels, self.out_channels, batchnorm=True, relu=True)
        self.maxpool = MaxPool(self.in_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        return x  

    def convert_keras(self, x, layer_name):
        x = self.conv.convert_keras(x, layer_name+"_conv")
        x = self.maxpool.convert_keras(x, layer_name+"_conv")
        return x

    def print_layer(self):
        return self.conv.print_layer() + "\n" + self.maxpool.print_layer()

class Classifier(nn.Module):
    def __init__(self, in_channels, out_channels, multiplier):
        super().__init__()

        self.multiplier = int(multiplier)
        self.in_channels = in_channels * self.multiplier ** 2
        self.out_channels = out_channels
        
        self.flatten = Flatten(self.in_channels, self.multiplier)
        self.linear = Linear(self.in_channels, self.out_channels, batchnorm=False, relu=False)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x

    def convert_keras(self, x, layer_name):
        x = self.flatten.convert_keras(x, layer_name+"_flatten")
        x = self.linear.convert_keras(x, layer_name+"_linear")
        return x

    def print_layer(self):
        return self.flatten.print_layer() + "\n" + self.linear.print_layer()