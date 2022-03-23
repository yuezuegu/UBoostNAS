
import torch 
import torch.nn as nn

import src.vanilla as vanilla

class Cell(nn.Module):
    def __init__(self, module_dict, in_channels, out_channels):
        super().__init__()

        self.module_dict = module_dict
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, xp, xpp):
        xp = self.module_dict["xp_pre"](xp)
        xpp = self.module_dict["xpp_pre"](xpp)

        xp_t1 = self.module_dict["xp_t1"](xp)
        xp_t2 = self.module_dict["xp_t2"](xp)
        xp_t3 = self.module_dict["xp_t3"](xp)

        xpp_t1 = self.module_dict["xpp_t1"](xpp)
        xpp_t2 = self.module_dict["xpp_t2"](xpp)
        xpp_t3 = self.module_dict["xpp_t3"](xpp)

        t1 = self.module_dict["t1_add"]([xp_t1, xpp_t1])

        t1_t2 = self.module_dict["t1_t2"](t1)
        t1_t3 = self.module_dict["t1_t3"](t1)

        t2 = self.module_dict["t2_add"]([t1_t2, xp_t2, xpp_t2])
        t2_t3 = self.module_dict["t2_t3"](t2)

        y = self.module_dict["t3_add"]([t2_t3, t1_t3, xp_t3, xpp_t3])

        if "out_maxpool" in self.module_dict:
            y = self.module_dict["out_maxpool"](y)

        return y
        
    def convert_keras(self, xp, xpp, layer_name=None):
        xp = self.module_dict["xp_pre"].convert_keras(xp, layer_name=layer_name+"_xp_pre")
        xpp = self.module_dict["xpp_pre"].convert_keras(xpp, layer_name=layer_name+"_xpp_pre")

        xp_t1 = self.module_dict["xp_t1"].convert_keras(xp, layer_name=layer_name+"_xp_t1")
        xp_t2 = self.module_dict["xp_t2"].convert_keras(xp, layer_name=layer_name+"_xp_t2")
        xp_t3 = self.module_dict["xp_t3"].convert_keras(xp, layer_name=layer_name+"_xp_t3")

        xpp_t1 = self.module_dict["xpp_t1"].convert_keras(xpp, layer_name=layer_name+"_xpp_t1")
        xpp_t2 = self.module_dict["xpp_t2"].convert_keras(xpp, layer_name=layer_name+"_xpp_t2")
        xpp_t3 = self.module_dict["xpp_t3"].convert_keras(xpp, layer_name=layer_name+"_xpp_t3")

        t1_add = vanilla.layers.Add([self.module_dict["xp_t1"], 
                                        self.module_dict["xpp_t1"]]).convert_keras([xp_t1, xpp_t1])

        t1_t2 = self.module_dict["t1_t2"].convert_keras(t1_add, layer_name=layer_name+"_t1_t2")
        t1_t3 = self.module_dict["t1_t3"].convert_keras(t1_add, layer_name=layer_name+"_t1_t3")

        t2_add = vanilla.layers.Add([self.module_dict["xp_t2"], 
                                        self.module_dict["xpp_t2"], 
                                        self.module_dict["t1_t2"]]).convert_keras([xp_t2, xpp_t2, t1_t2])

        t2_t3 = self.module_dict["t2_t3"].convert_keras(t2_add, layer_name=layer_name+"_t2_t3")

        out = vanilla.layers.Add([self.module_dict["xp_t3"], 
                                    self.module_dict["xpp_t3"], 
                                    self.module_dict["t1_t3"], 
                                    self.module_dict["t2_t3"]]).convert_keras([xp_t3, xpp_t3, t1_t3, t2_t3])

        if "out_maxpool" in self.module_dict:
            out = self.module_dict["out_maxpool"].convert_keras(out, layer_name=layer_name+"_out_maxpool")

        return out

    def print_layer(self):
        s = ""
        for i in ["xp_t1", "xp_t2", "xp_t3", "xpp_t1", "xpp_t2", "xpp_t3", "t1_t2", "t1_t3", "t2_t3"]:
            s += "\n" + i + ":\t" +  self.module_dict[i].print_layer()
        return s