# seeai_datamodule.py

import os
from typing import Any
import torch
from torchvision import datasets
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from lightly.transforms.dino_transform import DINOTransform
import torchvision.transforms as transforms


class SEEAIDataModule(LightningDataModule):

    name = "seeai"

    def __init__(self, config, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.config = config
        self.data_dir = config['path']
        self.train_val_split = config.get('train_val_split', 0.8)
        self.train_test_split = config.get('train_test_split', 0.8)
        self.num_workers = config['training']['num_workers']
        self.batch_size = config['training']['batch_size']
        self.seed = config.get('seed', 42)
        self.shuffle = True
        self.pin_memory = True
        self.drop_last = False
        self.img_size = config['img_size']

        # Initialize transforms
        self.transforms = DINOTransform(
            global_crop_size=config['transforms'].get('global_crop_size', 224),
            global_crop_scale=tuple(config['transforms'].get('global_crop_scale', [0.4, 1.0])),
            local_crop_size=config['transforms'].get('local_crop_size', 96),
            local_crop_scale=tuple(config['transforms'].get('local_crop_scale', [0.05, 0.4])),
            n_local_views=config['transforms'].get('n_local_views', 6),
            hf_prob=config['transforms'].get('hf_prob', 0.5),
            vf_prob=config['transforms'].get('vf_prob', 0.0),
            rr_prob=config['transforms'].get('rr_prob', 0.0),
            rr_degrees=config['transforms'].get('rr_degrees', None),
            cj_prob=config['transforms'].get('cj_prob', 0.8),
            cj_strength=config['transforms'].get('cj_strength', 0.5),
            cj_bright=config['transforms'].get('cj_bright', 0.8),
            cj_contrast=config['transforms'].get('cj_contrast', 0.8),
            cj_sat=config['transforms'].get('cj_sat', 0.4),
            cj_hue=config['transforms'].get('cj_hue', 0.2),
            random_gray_scale=config['transforms'].get('random_gray_scale', 0.2),
            gaussian_blur=tuple(config['transforms'].get('gaussian_blur', [1.0, 0.1, 0.5])),
            sigmas=tuple(config['transforms'].get('sigmas', [0.1, 2])),
            solarization_prob=config['transforms'].get('solarization_prob', 0.2),
            normalize=transforms.Normalize(
                mean=config['transforms']['normalize']['mean'],
                std=config['transforms']['normalize']['std']
            )
        )

    @property
    def num_classes(self) -> int:
        # Update this if you have labels; for self-supervised learning, this may not be needed
        return 0

    def prepare_data(self) -> None:
        # If you need to download data, do it here
        pass

    def setup(self, stage: str = None) -> None:
        seeAI_full = datasets.ImageFolder(self.data_dir, transform=self.transforms)
        total_len = len(seeAI_full)
        train_len = int(self.train_test_split * total_len)
        test_len = total_len - train_len
        val_len = int(self.train_val_split * train_len)
        train_len = train_len - val_len

        self.seeAI_train, self.seeAI_val, self.seeAI_test = random_split(
            seeAI_full,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.seeAI_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.seeAI_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.seeAI_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )