
import torch
from src.data.datasets import return_dataset
from src.utils import accuracy, Stats
from robustness.datasets import CustomImageNet
from torch.utils.data import DataLoader
from torch.autograd import Variable

import tensorflow as tf 

import time 
import os
import sys
import numpy as np
from tensorflow.keras.utils import plot_model

from pytorch2keras import pytorch_to_keras

sys.path.append('./src/baselines/mobile_vision')

from mobile_cv.model_zoo.models.fbnet_v2 import fbnet
from mobile_cv.model_zoo.models.preprocess import get_preprocess

device = torch.device('cpu')

model = fbnet("FBNetV2_F4", pretrained=True).to(device)
model.convert_keras()


exit()

# Dataloaders for FT stage
train_queue, valid_queue, test_queue, input_dims, no_classes  = return_dataset(
    dataset='imagenet', 
    batch_size=1, 
    num_workers=8)

stats = Stats()

model.eval()
no_steps = len(test_queue)
for step, (x, y) in enumerate(test_queue):
    start = time.time()
    x = x.requires_grad_(False).to(device)
    y = y.requires_grad_(False).to(device)

    logits = model(x)

    prec1, prec5 = accuracy(logits, y, topk=(1, 5))
    n = x.size(0)
    stats.update_avg("valid_top1", n, prec1)
    stats.update_avg("valid_top5", n, prec5)
    elapsed = time.time() - start
    print("step: {} / {} elapsed {}s".format(step, no_steps, elapsed))

print("top1: {} top5: {}".format(stats.get_metric("valid_top1"), stats.get_metric("valid_top5")))





