import torch.nn as nn
import tensorflow as tf
import json
import time 

import sys
sys.path.append('.')
import src.vanilla as vanilla
from src.hardware.precompile import precompile_model 
from src.hardware.sim_binder import run_csim

def run_layer(array_size, layer_type, in_ch, out_ch, kernel_size, input_size, batch_size):
    input_image_dims = [input_size,input_size,in_ch]
    input_layer = tf.keras.Input(shape=input_image_dims, batch_size=batch_size, name="input_1")
    x = input_layer

    if layer_type == "Conv2d":
        x = tf.keras.layers.Conv2D(
            filters=out_ch,
            kernel_size=list(kernel_size),
            strides=(1, 1),
            padding="same",
            activation=None,
            use_bias=True,
            name="dummy_layer"
        )(x)

    elif layer_type == "DepthwiseConv2d":
        x = tf.keras.layers.DepthwiseConv2D(
                kernel_size=kernel_size,
                strides=(1, 1),
                depth_multiplier=1,
                padding="same",
                activation=None,
                use_bias=True,
                name="dummy_layer"
                )(x)        
    
    else:
        raise NotImplementedError

    keras_model = tf.keras.models.Model(inputs=input_layer, outputs=x)
    layers = precompile_model(keras_model, array_size=array_size, partition_size=None)

    no_ops = 0
    for layer_name in layers:
        gemm_op = layers[layer_name]['gemm_op']
        if gemm_op is not None:
            X = gemm_op['input_size']
            W = gemm_op['weight_size']
            no_ops += (2 * X[0] * X[1] * W[1])

    json_out = {"args":None, "model1":{"order":list(layers.keys()), "layers":layers, "no_repeat":1, "no_ops":no_ops}}
    sim_res = run_csim(json_out)

    csim_runtime = sim_res['no_cycles'] * 1e-9 * 1e3
    throughput = sim_res['no_ops'] / 2 / (csim_runtime/1e3)

    peak_throughput = array_size[0] * array_size[1] * 1e9
    csim_util = throughput / peak_throughput

    return csim_runtime, csim_util


if __name__ == "__main__":


    array_size = [128,128]

    batch_size = 1

    table = {}

    start = time.time()

    discrete_step = 16

    op = vanilla.Conv2d
    opname = op.__name__
    table[opname] = {}
    for input_size in [2,4,8,16,32]:
        table[opname][input_size] = {}
        for kernel_size in [(1,1), (3,3), (5,5)]:
            kernel_size_str = "{}x{}".format(kernel_size[0], kernel_size[1])
            table[opname][input_size][kernel_size_str] = {}
            for in_ch in range (discrete_step,256+1,discrete_step):
                table[opname][input_size][kernel_size_str][in_ch] = {}
                for out_ch in range (discrete_step,256+1,discrete_step):
                    runtime_csim, util_csim = run_layer(array_size, opname, in_ch, out_ch, kernel_size, input_size, batch_size)

                    print("op:{}\t input_size: {}\t kernel_size: {}\t in_ch:{}\t out_ch:{}\t runtime_csim: {} ms\t util_csim: {}".format(opname, input_size, kernel_size_str, in_ch, out_ch, runtime_csim, util_csim))

                    table[opname][input_size][kernel_size_str][in_ch][out_ch] = {"runtime_csim": runtime_csim, "util_csim": util_csim}

    for input_size in [1]:
        table[opname][input_size] = {}
        for kernel_size in [(1,1)]:
            kernel_size_str = "{}x{}".format(kernel_size[0], kernel_size[1])
            table[opname][input_size][kernel_size_str] = {}
            for in_ch in range (discrete_step, 512+1,discrete_step):
                table[opname][input_size][kernel_size_str][in_ch] = {}
                for out_ch in range (discrete_step, 512+1,discrete_step):
                    runtime_csim, util_csim = run_layer(array_size, opname, in_ch, out_ch, kernel_size, input_size, batch_size)

                    print("op:{}\t input_size: {}\t kernel_size: {}\t in_ch:{}\t out_ch:{}\t runtime_csim: {} ms\t util_csim: {}".format(opname, input_size, kernel_size_str, in_ch, out_ch, runtime_csim, util_csim))

                    table[opname][input_size][kernel_size_str][in_ch][out_ch] = {"runtime_csim": runtime_csim, "util_csim": util_csim}

    op = vanilla.DepthwiseConv2d
    opname = op.__name__
    table[opname] = {}
    for input_size in [2,4,8,16,32]:
        table[opname][input_size] = {}
        for kernel_size in [(3,3), (5,5)]:
            kernel_size_str = "{}x{}".format(kernel_size[0], kernel_size[1])
            table[opname][input_size][kernel_size_str] = {}
            for in_ch in range (discrete_step,256+1,discrete_step):
                table[opname][input_size][kernel_size_str][in_ch] = {}

                out_ch = in_ch
                runtime_csim, util_csim = run_layer(array_size, opname, in_ch, out_ch, kernel_size, input_size, batch_size)

                print("op:{}\t input_size: {}\t kernel_size: {}\t in_ch:{}\t out_ch:{}\t runtime_csim: {} ms\t util_csim: {}".format(opname, input_size, kernel_size_str, in_ch, out_ch, runtime_csim, util_csim))

                table[opname][input_size][kernel_size_str][in_ch][out_ch] = {"runtime_csim": runtime_csim, "util_csim": util_csim}

    with open("src/hardware/lookup.json", "w") as outfile:  
        json.dump(table, outfile)

    print("Completed in: {} s".format(time.time() - start))