from .cifar10 import prepare_cifar10
from .cifar100 import prepare_cifar100
from .tiny_imagenet import prepare_tiny_imagenet
from .imagenet import prepare_imagenet, prepare_full_imagenet

import torch
import torchvision.datasets as dset
from torchvision import transforms

def return_dataset(dataset, batch_size, num_workers):
    if dataset == "cifar10":
        return prepare_cifar10(batch_size, num_workers)
    elif dataset == "imagenet100":
        return prepare_imagenet(batch_size, num_workers)
    elif dataset == "imagenet":
        return prepare_full_imagenet(batch_size, num_workers)
    else:
        raise NotImplementedError