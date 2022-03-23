import logging

import torch
import torchvision
import torchvision.datasets as dset
from torchvision import transforms


def cifar100_preprocess(args):
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    preprocess = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD),
        ]
    )
    return preprocess


def prepare_cifar100(batch_size):
    # initialize train and valid datasets
    image_size = 32
    preprocess = cifar100_preprocess(image_size)

    train_dataset = dset.CIFAR100(
        root="./cifar100", train=True, download=True, transform=preprocess
    )
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [40000, 10000])

    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=0, drop_last=True
    )

    valid_dl = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=0, drop_last=True
    )

    return train_dl, valid_dl
