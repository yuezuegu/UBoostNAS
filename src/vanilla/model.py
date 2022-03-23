
import torch
import torch.nn as nn

import tensorflow as tf
from tensorflow.keras.utils import plot_model

import src.vanilla as vanilla

import logging 

class Model(nn.Module):
    def __init__(self, module_dict, input_dims, no_classes, args):
        super().__init__()

        self.input_dims = input_dims
        self.no_classes = no_classes
        self.args = args
        self.module_dict = module_dict

    def forward(self, x):
        x_curr = x
        x_curr = self.module_dict["stem"](x_curr)

        x_prev = x_curr
        x_prevprev = x_curr

        for layer_name, layer in self.module_dict.items():
            if isinstance(layer, vanilla.cell.Cell):
                x_curr = layer(x_prev, x_prevprev)

                x_prevprev = x_prev
                x_prev = x_curr

        x_curr = self.module_dict["classifier"](x_curr)

        return x_curr

    def convert_to_keras(self):
        input_layer = tf.keras.Input(shape=self.input_dims, batch_size=1, name="input_1")
        x = input_layer

        x = self.module_dict["stem"].convert_keras(x, layer_name="stem")

        x_prev = x
        x_prevprev = x 
        for k, module in self.module_dict.items():
            if isinstance(module, vanilla.cell.Cell):
                x = module.convert_keras(x_prev, x_prevprev, layer_name=k)

                x_prevprev = x_prev
                x_prev = x

        x = self.module_dict["classifier"].convert_keras(x, layer_name="classifier")

        model = tf.keras.models.Model(inputs=input_layer, outputs=x)
        return model

    def plot_model(self, filename):
        s = ""
        for k, module in self.module_dict.items():
            _s = "{}".format(module.print_layer())
            s += "\n" + _s
            logging.info(_s)
        
        with open(filename+".txt", "w") as f:
            f.write(s)

        plot_model(self.convert_to_keras(), to_file=filename+".png", show_shapes=True, show_layer_names=True)
