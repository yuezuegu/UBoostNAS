import logging
import math
from sklearn.utils import shuffle

import torch
import torchvision
import torchvision.datasets as dset
from torchvision import transforms

# cd
# mkdir datasets
# cd datasets/
# mkdir tiny
# wget -O ~/datasets/tiny/tiny-imagenet-200.zip http://cs231n.stanford.edu/tiny-imagenet-200.zip
# cd tiny/
# unzip tiny-imagenet-200.zip


def prepare_tiny_imagenet(batch_size, tr="on"):

    # ROOT = "/datasets2/tiny_imagenet200/tiny-imagenet-200/"
    ROOT = "/home/ndimitri/datasets/tiny/tiny-imagenet-200/"

    logging.warning("PREPROCESSING ONLY INCLUDES ToTensor() FOR THE MOMENT")
    # mean and std values taken from https://github.com/DennisHanyuanXu/Tiny-ImageNet/blob/master/src/data_prep.py
    if tr == "off":
        tr = None
    else:
        tr = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    # tr = None

    train = torchvision.datasets.ImageFolder(ROOT + "train/", transform=tr)
    valid = torchvision.datasets.ImageFolder(ROOT + "val/", transform=tr)

    train_dl = torch.utils.data.DataLoader(train, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)

    valid_dl = torch.utils.data.DataLoader(valid, batch_size=batch_size, num_workers=0, drop_last=True)

    return train_dl, valid_dl, valid_dl
