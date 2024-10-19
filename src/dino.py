# dino.py

import argparse
import copy
from typing import Any

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
from lightly.models.utils import deactivate_requires_grad, update_momentum
from vision_transformer import VisionTransformer
from seeai_datamodule import SEEAIDataModule
import yaml

parser = argparse.ArgumentParser("DINO Training")
parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")


class DINO(pl.LightningModule):
    def __init__(self, config, pretrained=False):
        super(DINO, self).__init__()
        self.save_hyperparameters()
        self.config = config
        img_size = config['img_size']
        vitb16_s = VisionTransformer(img_size=[img_size])
        vitb16_t = copy.deepcopy(vitb16_s)
        input_dim = vitb16_s.embed_dim
        if pretrained:
            state_dict_s = torch.load(
                "./EndoFM/endo_fm.pth", map_location=torch.device('cpu'))["student"]
            new_state_dict_s = {}
            for k, v in state_dict_s.items():
                if k.startswith('module.backbone.'):
                    new_key = k[len('module.backbone.'):]
                    new_state_dict_s[new_key] = v
                else:
                    new_state_dict_s[k] = v
            state_dict_t = torch.load(
                "./EndoFM/endo_fm.pth", map_location=torch.device('cpu'))["teacher"]
            new_state_dict_t = {}
            for k, v in state_dict_t.items():
                if k.startswith('module.backbone.'):
                    new_key = k[len('module.backbone.'):]
                    new_state_dict_t[new_key] = v
                else:
                    new_state_dict_t[k] = v

            vitb16_s.load_state_dict(new_state_dict_s, strict=False)
            vitb16_t.load_state_dict(new_state_dict_t, strict=False)

        self.student_backbone = vitb16_s
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, self.config['dino_loss']['output_dim'], freeze_last_layer=1)

        self.teacher_backbone = vitb16_t
        self.teacher_head = DINOProjectionHead(
            input_dim, 512, 64, self.config['dino_loss']['output_dim'])
        deactivate_requires_grad(self.teacher_head)
        deactivate_requires_grad(self.teacher_backbone)

        self.criterion = DINOLoss(
            output_dim=self.config['dino_loss']['output_dim'],
            warmup_teacher_temp=self.config['dino_loss']['warmup_teacher_temp'],
            teacher_temp=self.config['dino_loss']['teacher_temp'],
            warmup_teacher_temp_epochs=self.config['dino_loss']['warmup_teacher_temp_epochs'],
            student_temp=self.config['dino_loss']['student_temp'],
            center_momentum=self.config['dino_loss']['center_momentum']
        )

        # Initialize parameters for momentum schedule
        self.base_momentum = self.config['momentum_schedule']['base_momentum']
        self.final_momentum = self.config['momentum_schedule']['final_momentum']
        self.max_epochs = self.config['training']['epochs']

    def forward(self, x) -> Any:
        x = self.student_backbone(x).flatten(start_dim=1)
        x = self.student_head(x)
        return x

    def forward_teacher(self, x):
        x = self.teacher_backbone(x).flatten(start_dim=1)
        x = self.teacher_head(x)
        return x

    def training_step(self, batch, batch_idx):
        current_epoch = self.current_epoch
        momentum = cosine_schedule(
            current_epoch,
            self.max_epochs,
            self.base_momentum,
            self.final_momentum
        )
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log('train_loss', loss)
        return loss

    def on_after_backward(self) -> None:
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=0
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


if __name__ == "__main__":
    args = parser.parse_args()

    # Load the YAML configuration file
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)

    # Initialize the model
    model = DINO(config=config, pretrained=True)

    # Initialize the data module
    dataset = SEEAIDataModule(config=config)

    # Define callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='train_loss',
        save_top_k=1,
        mode='min',
        dirpath='checkpoints/',
        filename='dino-{epoch:02d}-{train_loss:.2f}'
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    # Initialize the Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=16 if config['training'].get('precision', 32) == 16 else 32,
        amp_backend='native',
        gradient_clip_val=1.0,
        deterministic=True,
        check_val_every_n_epoch=5  # Adjust as needed
    )

    # Start training
    trainer.fit(model, datamodule=dataset)