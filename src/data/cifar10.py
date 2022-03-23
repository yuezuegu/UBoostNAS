import logging

import torch
import torchvision
import torchvision.datasets as dset
from torchvision import transforms


def prepare_cifar10(batch_size, num_workers):
    input_dims = [32, 32, 3]
    no_classes = 10

    # Taken from DART code
    def cifar10_preprocess(train=True):
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

        if train:
            transform_list = [
                transforms.RandomCrop(input_dims[0], padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
            ]
        else:
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
            ]

        return transforms.Compose(transform_list)

    # initialize train and valid datasets
    train_data = dset.CIFAR10(root="./cifar10", train=True, download=True, transform=cifar10_preprocess(train=True))
    train_data, valid_data = torch.utils.data.random_split(train_data, [45000, 5000])

    test_data = dset.CIFAR10(root="./cifar10", train=False, download=True, transform=cifar10_preprocess(train=False))

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=True
    )

    valid_queue = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=True
    )

    test_queue = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True
    )

    return train_queue, valid_queue, test_queue, input_dims, no_classes
