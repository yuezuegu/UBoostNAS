import os
import tempfile
from typing import Optional

import torch
import pytorch_lightning as pl
import torchvision.transforms as T
from robustness.datasets import CustomImageNet
from torch.utils.data import ConcatDataset, DataLoader

import logging

def prepare_imagenet(batch_size, num_workers):
    input_dims = [128,128,3]
    no_classes = 100

    # initialize train and valid datasets
    train_dataset = ImageNet100DataModule(
        data_dir="/home/yuezuegu/Imagenet/",
        train_batch_size=batch_size,
        test_batch_size=batch_size,
        cropsize=input_dims[0],
        resize=input_dims[0],
        num_workers=num_workers,
    )
    train_dataset.setup()

    train_dl = train_dataset.train_dataloader()
    valid_dl = train_dataset.val_dataloader()
    test_dl = train_dataset.test_dataloader()

    return train_dl, valid_dl, test_dl, input_dims, no_classes


def prepare_full_imagenet(batch_size, num_workers):
    input_dims = [224,224,3]
    no_classes = 1000

    # initialize train and valid datasets
    dm = ImageNetDataModule(
        data_dir="/home/yuezuegu/Imagenet/",
        train_batch_size=batch_size,
        test_batch_size=batch_size,
        cropsize=input_dims[0],
        resize=input_dims[0],
        num_workers=num_workers,
    )
    dm.setup()

    train_dl = dm.train_dataloader()
    valid_dl = dm.val_dataloader()
    test_dl = dm.test_dataloader()

    logging.info(f"The input has the following dimensions: {train_dl.dataset[0][0].shape}")

    x, y = next(iter(train_dl))
    logging.info(f"The input fed to the model has the following dimensions: {x.shape}")

    return train_dl, valid_dl, test_dl, input_dims, no_classes


class ImageNet(CustomImageNet):
    def __init__(self, data_path, **kwargs):
        super().__init__(
            data_path=data_path,
            custom_grouping=[[label] for label in range(0, 1000)],
            **kwargs,
        )


class ImageNetC(CustomImageNet):
    def __init__(self, data_path, corruption_type, severity, **kwargs):
        tmp_data_path = tempfile.mkdtemp()
        os.symlink(
            os.path.join(data_path, corruption_type, str(severity)),
            os.path.join(tmp_data_path, "test"),
        )
        super().__init__(
            data_path=tmp_data_path,
            custom_grouping=[[label] for label in range(0, 1000)],
            **kwargs,
        )

class ImageNet100(CustomImageNet):
    def __init__(self, data_path, **kwargs):
        super().__init__(
            data_path=data_path,
            custom_grouping=[[label] for label in range(0, 1000, 10)],
            **kwargs,
        )


class ImageNet100C(CustomImageNet):
    def __init__(self, data_path, corruption_type, severity, **kwargs):
        tmp_data_path = tempfile.mkdtemp()
        os.symlink(
            os.path.join(data_path, corruption_type, str(severity)),
            os.path.join(tmp_data_path, "test"),
        )
        super().__init__(
            data_path=tmp_data_path,
            custom_grouping=[[label] for label in range(0, 1000, 10)],
            **kwargs,
        )

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        train_batch_size=128,
        test_batch_size=128,
        num_workers=16,
        cropsize = 224,
        resize = 256,
        pin_memory=True,
        shuffle_train=True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.cropsize = cropsize
        self.train_transform = T.Compose(
            [
                T.ToTensor(),
                T.RandomResizedCrop(self.cropsize),
                T.RandomHorizontalFlip(),
                T.Normalize(self.mean, self.std),
            ]
        )
        self.test_transform = T.Compose([
                T.ToTensor(), 
                T.Resize(resize), 
                T.CenterCrop(self.cropsize),
                T.Normalize(self.mean, self.std),
            ]
        )
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.dims = (3, self.cropsize, self.cropsize)
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.shuffle_train = shuffle_train

    def setup(self, stage: Optional[str] = None):
        self.imagenet = ImageNet(self.data_dir)
        train_loader, test_loader = self.imagenet.make_loaders(
            workers=self.num_workers,
            batch_size=self.train_batch_size,
            val_batch_size=self.test_batch_size,
            shuffle_val=False,
        )
        self.imagenet_train = train_loader.dataset
        self.imagenet_train.transform = self.train_transform

        self.imagenet_test = test_loader.dataset
        self.imagenet_test.transform = self.test_transform

    def train_dataloader(self):
        return DataLoader(
            #torch.utils.data.Subset(self.imagenet_train, list(range(10240))),
            self.imagenet_train,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle_train,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.imagenet_test,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )

    def test_dataloader(self):
        return self.val_dataloader()


class ImageNet100DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        train_batch_size=128,
        test_batch_size=128,
        cropsize=224,
        resize=256,
        num_workers=1,
        pin_memory=True,
        shuffle_train=True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_transform = T.Compose(
            [
                T.ToTensor(),
                T.RandomResizedCrop(cropsize),
                T.RandomHorizontalFlip(),
            ]
        )
        self.test_transform = T.Compose([T.ToTensor(), T.Resize(resize), T.CenterCrop(cropsize)])
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.dims = (3, cropsize, cropsize)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.shuffle_train = shuffle_train

    def setup(self, stage: Optional[str] = None):
        self.imagenet100 = ImageNet100(self.data_dir)
        train_loader, test_loader = self.imagenet100.make_loaders(
            workers=self.num_workers,
            batch_size=self.train_batch_size,
            val_batch_size=self.test_batch_size,
            shuffle_val=False,
        )
        self.imagenet100_train = train_loader.dataset
        self.imagenet100_train.transform = self.train_transform

        self.imagenet100_test = test_loader.dataset
        self.imagenet100_test.transform = self.test_transform

    def train_dataloader(self):
        return DataLoader(
            self.imagenet100_train,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle_train,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.imagenet100_test,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )

    def test_dataloader(self):
        return self.val_dataloader()


class LP_ImageNet100CDataModule(ImageNet100DataModule):
    """Class to encapsulate low pass ImageNet100c"""

    def __init__(self, data_dir: str, bandwidth, train_batch_size, test_batch_size, num_workers, pin_memory):
        self.bandwidth = bandwidth
        super().__init__(
            data_dir=f"{data_dir}/{bandwidth}",
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


class ImageNet100CDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size=128, num_workers=1, pin_memory=True, normalized=True):
        super().__init__()
        self.data_dir = data_dir
        self.transform = None
        self.batch_size = batch_size
        self.dims = (3, 224, 224)
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.normalized = normalized
        self.transform = T.Compose([T.ToTensor(), T.Resize(256), T.CenterCrop(224)])
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.corruptions = [
            "brightness",
            "contrast",
            "defocus_blur",
            "elastic_transform",
            "fog",
            "frost",
            # "gaussian_blur",
            "gaussian_noise",
            "glass_blur",
            "impulse_noise",
            "jpeg_compression",
            "motion_blur",
            "pixelate",
            # "saturate",
            "shot_noise",
            "snow",
            # "spatter",
            # "speckle_noise",
            "zoom_blur",
        ]

    def setup(self, stage: Optional[str] = None, severity_min: int = 1, severity_max: int = 6):
        self.imagenet100c = {}

        for corruption in self.corruptions:
            imagenet100c_corruption = []
            for severity in range(severity_min, severity_max):
                base_dataset = ImageNet100C(self.data_dir, corruption, severity)
                _, test_loader = base_dataset.make_loaders(self.num_workers, self.batch_size, only_val=True)
                imagenet100c_severity = test_loader.dataset
                imagenet100c_severity.transform = self.transform
                imagenet100c_corruption.append(imagenet100c_severity)

            self.imagenet100c[corruption] = ConcatDataset(imagenet100c_corruption)

    def test_dataloader(self):
        return {
            corruption: DataLoader(
                self.imagenet100c[corruption],
                batch_size=self.batch_size // len(self.corruptions),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            for corruption in self.corruptions
        }


class ImageNetCDataModule(ImageNet100CDataModule):
    def setup(self, stage: Optional[str] = None, severity_min: int = 1, severity_max: int = 6):
        self.imagenetC = {}

        for corruption in self.corruptions:
            imagenetC_corruption = []
            for severity in range(severity_min, severity_max):
                base_dataset = ImageNetC(self.data_dir, corruption, severity)
                _, test_loader = base_dataset.make_loaders(self.num_workers, self.batch_size, only_val=True)
                imagenetC_severity = test_loader.dataset
                imagenetC_severity.transform = self.transform
                imagenetC_corruption.append(imagenetC_severity)

            self.imagenetC[corruption] = ConcatDataset(imagenetC_corruption)

    def test_dataloader(self):
        return {
            corruption: DataLoader(
                self.imagenetC[corruption],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            for corruption in self.corruptions
        }


# class AugMixImageNet100DataModule(ImageNet100DataModule):
#     def __init__(
#             self,
#             data_dir: str = "./",
#             train_batch_size=128,
#             test_batch_size=128,
#             num_workers=1,
#             pin_memory=True,
#             augmix_cfg=None,
#             aug_list=None
#     ):
#         super().__init__(
#             data_dir=data_dir,
#             train_batch_size=train_batch_size,
#             test_batch_size=test_batch_size,
#             num_workers=num_workers,
#             pin_memory=pin_memory, )

#         self.data_dir = data_dir
#         self.train_transform = T.Compose(
#             [
#                 T.RandomResizedCrop(224),
#                 T.RandomHorizontalFlip(),
#             ]
#         )
#         self.preprocess = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
#         self.test_transform = T.Compose([T.ToTensor(), T.Resize(256), T.CenterCrop(224), T.Normalize(self.mean, self.std)])
#         self.train_batch_size = train_batch_size
#         self.test_batch_size = test_batch_size
#         self.dims = (3, 224, 224)
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#         self.mean = [0.485, 0.456, 0.406]
#         self.std = [0.229, 0.224, 0.225]

#         self.aug_list = aug_list
#         if augmix_cfg is None:
#             self.augmix_cfg = {
#                 "all_ops": False,
#                 "mixture_width": 3,
#                 "mixture_depth": -1,
#                 "aug_severity": 3,
#                 "no_jsd": True,
#             }
#         else:
#             self.augmix_cfg = augmix_cfg

#     def setup(self, stage: Optional[str] = None):

#         self.imagenet100 = ImageNet100(self.data_dir)
#         train_loader, test_loader = self.imagenet100.make_loaders(workers=self.num_workers,
#                                                                   batch_size=self.train_batch_size,
#                                                                   val_batch_size=self.test_batch_size,
#                                                                   shuffle_val=False)
#         self.imagenet100_train = train_loader.dataset
#         self.imagenet100_train.transform = self.train_transform

#         self.imagenet100_test = test_loader.dataset
#         self.imagenet100_test.transform = self.test_transform

#         self.augmix_imagenet100_train = AugMixDataset(dataset=self.imagenet100_train,
#                                                       preprocess=self.preprocess,
#                                                       all_ops=self.augmix_cfg["all_ops"],
#                                                       mixture_width=self.augmix_cfg["mixture_width"],
#                                                       mixture_depth=self.augmix_cfg["mixture_depth"],
#                                                       aug_severity=self.augmix_cfg["aug_severity"],
#                                                       no_jsd=self.augmix_cfg["no_jsd"],
#                                                       aug_list=self.aug_list,
#                                                       img_sz=224
#                                                       )

#     def train_dataloader(self):
#         return DataLoader(
#             self.augmix_imagenet100_train,
#             batch_size=self.train_batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             shuffle=True,
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.imagenet100_test,
#             batch_size=self.test_batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#         )

#     def test_dataloader(self):
#         return self.val_dataloader()


# class GeneralizedAugMixImageNet100DataModule(ImageNet100DataModule):
#     def __init__(
#             self,
#             data_dir: str = "./",
#             train_batch_size=128,
#             test_batch_size=128,
#             num_workers=1,
#             pin_memory=True,
#             augmix_cfg=None,
#             aug_list=None
#     ):
#         super().__init__(
#             data_dir=data_dir,
#             train_batch_size=train_batch_size,
#             test_batch_size=test_batch_size,
#             num_workers=num_workers,
#             pin_memory=pin_memory, )

#         self.data_dir = data_dir
#         self.train_transform = T.Compose(
#             [
#                 T.ToTensor(),
#                 T.RandomResizedCrop(224),
#                 T.RandomHorizontalFlip(),
#             ]
#         )
#         self.preprocess = T.Compose([T.Normalize(self.mean, self.std)])
#         self.test_transform = T.Compose([T.ToTensor(), T.Resize(256), T.CenterCrop(224), T.Normalize(self.mean, self.std)])
#         self.train_batch_size = train_batch_size
#         self.test_batch_size = test_batch_size
#         self.dims = (3, 224, 224)
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#         self.mean = [0.485, 0.456, 0.406]
#         self.std = [0.229, 0.224, 0.225]

#         self.aug_list = aug_list
#         if augmix_cfg is None:
#             self.augmix_cfg = {
#                 "all_ops": False,
#                 "mixture_width": 3,
#                 "mixture_depth": -1,
#                 "aug_severity": 3,
#                 "no_jsd": True,
#             }
#         else:
#             self.augmix_cfg = augmix_cfg

#     def setup(self, stage: Optional[str] = None):

#         self.imagenet100 = ImageNet100(self.data_dir)
#         train_loader, test_loader = self.imagenet100.make_loaders(workers=self.num_workers,
#                                                                   batch_size=self.train_batch_size,
#                                                                   val_batch_size=self.test_batch_size,
#                                                                   shuffle_val=False)
#         self.imagenet100_train = train_loader.dataset
#         self.imagenet100_train.transform = self.train_transform

#         self.imagenet100_test = test_loader.dataset
#         self.imagenet100_test.transform = self.test_transform

#         self.augmix_imagenet100_train = GeneralizedAugMixDataset(dataset=self.imagenet100_train,
#                                                                  preprocess=self.preprocess,
#                                                                  all_ops=self.augmix_cfg["all_ops"],
#                                                                  mixture_width=self.augmix_cfg["mixture_width"],
#                                                                  mixture_depth=self.augmix_cfg["mixture_depth"],
#                                                                  no_jsd=self.augmix_cfg["no_jsd"],
#                                                                  aug_list=self.aug_list,
#                                                                  img_sz=224
#                                                                  )

# def train_dataloader(self):
#     return DataLoader(
#         self.augmix_imagenet100_train,
#         batch_size=self.train_batch_size,
#         num_workers=self.num_workers,
#         pin_memory=self.pin_memory,
#         shuffle=True,
#         # persistent_workers=True
#     )
#     # return AsynchronousLoader(
#     #     DataLoader(
#     #         self.augmix_imagenet100_train,
#     #         batch_size=self.train_batch_size,
#     #         num_workers=self.num_workers,
#     #         pin_memory=self.pin_memory,
#     #         shuffle=True,
#     #         # persistent_workers=True
#     #     )
#     # )

# def val_dataloader(self):
#     return DataLoader(
#         self.imagenet100_test,
#         batch_size=self.test_batch_size,
#         num_workers=self.num_workers,
#         pin_memory=self.pin_memory,
#     )

# def test_dataloader(self):
#     return self.val_dataloader()
