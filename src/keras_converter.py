
import torch 
import torch.nn as nn 

import tensorflow as tf

def convert_model(x, torch_module, layer_name):
    if isinstance(torch_module, nn.Conv2d):
        use_bias = torch_module.bias is not None
        groups = torch_module.groups 
        pad = ((0,0), torch_module.padding, torch_module.padding, (0,0))  

        if groups == 1:
            keras_module = tf.keras.layers.Conv2D(
                filters=torch_module.out_channels,
                kernel_size=torch_module.kernel_size,
                strides=torch_module.stride,
                padding=pad,
                activation=None,
                use_bias=use_bias,
                name=layer_name+"_conv"
            )
            keras_module.build(x.shape)

            weights = [torch_module.weight.permute((2,3,1,0)).detach().numpy()]
            if use_bias:
                weights.append(torch_module.bias.detach().numpy())
        else:
            assert groups == torch_module.out_channels, "Only depth_multiplier=1 supported"

            keras_module = tf.keras.layers.DepthwiseConv2D(
                kernel_size=torch_module.kernel_size,
                strides=torch_module.stride,
                depth_multiplier=1,
                padding="same",
                activation=None,
                use_bias=use_bias,
                name=layer_name+"_conv"
            )
            keras_module.build(x.shape)

            weights = [torch_module.weight.permute((2,3,0,1)).detach().numpy()]
            if use_bias:
                weights.append(torch_module.bias.detach().numpy())

        keras_module.set_weights(weights)
    elif isinstance(torch_module, nn.Linear):
        use_bias = torch_module.bias is not None

        keras_module = tf.keras.layers.Dense(
            filters=torch_module.out_features,
            activation=None,
            use_bias=use_bias,
            name=layer_name+"_fc"
        ).build(x.shape)

        weights = [torch_module.weight.detach().numpy()]
        if use_bias:
            weights.append(torch_module.bias.detach().numpy())

        keras_module.set_weights(weights)

    elif isinstance(torch_module, nn.AdaptiveAvgPool2d):
        keras_module = tf.keras.layers.AveragePooling2D(
            pool_size=(x.shape[1], x.shape[2])
        )
        keras_module.build(x.shape)
    else:
        raise NotImplementedError

    
    return keras_module(x)