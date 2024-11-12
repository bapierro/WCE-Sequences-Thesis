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
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy  # Import DDPStrategy
from generateEval import run_evaluation
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
            # Load pre-trained student weights
            state_dict_s = torch.load(
                "./EndoFM/endo_fm.pth", map_location=torch.device('cpu'))["student"]
            new_state_dict_s = {}
            for k, v in state_dict_s.items():
                if k.startswith('module.backbone.'):
                    new_key = k[len('module.backbone.'):]
                    new_state_dict_s[new_key] = v
                else:
                    new_state_dict_s[k] = v
            vitb16_s.load_state_dict(new_state_dict_s, strict=False)

            # Load pre-trained teacher weights
            state_dict_t = torch.load(
                "./EndoFM/endo_fm.pth", map_location=torch.device('cpu'))["teacher"]
            new_state_dict_t = {}
            for k, v in state_dict_t.items():
                if k.startswith('module.backbone.'):
                    new_key = k[len('module.backbone.'):]
                    new_state_dict_t[new_key] = v
                else:
                    new_state_dict_t[k] = v
            vitb16_t.load_state_dict(new_state_dict_t, strict=False)

        self.student_backbone = vitb16_s
        self.student_head = DINOProjectionHead(
            self.config['dino_loss']['input_dim'],
            self.config['dino_loss']['hidden_dim'],
            self.config['dino_loss']['bottleneck_dim'],
            self.config['dino_loss']['output_dim'],
            freeze_last_layer=3
        )

        self.teacher_backbone = vitb16_t
        self.teacher_head = DINOProjectionHead(
            self.config['dino_loss']['input_dim'],
            self.config['dino_loss']['hidden_dim'],
            self.config['dino_loss']['bottleneck_dim'],
            self.config['dino_loss']['output_dim']
        )
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

        for param in self.student_backbone.patch_embed.parameters():
            param.requires_grad = False
            
        for block in self.student_backbone.blocks:
            # Freeze FFN layers
            for param in block.mlp.parameters():
                param.requires_grad = False

        for idx, block in enumerate(self.student_backbone.blocks):
            for param in block.attn.parameters():
                assert param.requires_grad == True, f"MHSA parameters in block {idx} should be trainable."


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
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            views = batch[0]
            views = [view.to(self.device) for view in views]
            global_views = views[:2]
            teacher_out = [self.forward_teacher(view) for view in global_views]
            student_out = [self.forward(view) for view in views]
            loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
            self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
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
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        dirpath=f'./checkpoints_{config["name"]}/',
        filename='dino-{epoch:02d}-{val_loss:.2f}',
        save_on_train_epoch_end=False,
        verbose=True
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    # Initialize logger
    logger = TensorBoardLogger("tb_logs", name=f"dino_model_{config['name']}")

    early_stopping = pl.callbacks.EarlyStopping(
        monitor=config['training']['early_stopping']['monitor'],
        patience=config['training']['early_stopping']['patience'],
        mode=config['training']['early_stopping']['mode'],
        verbose=True
    )

    # Determine the number of GPUs
    if isinstance(config["devices"], list):
        num_gpus = len(config["devices"])
    else:
        num_gpus = int(config["devices"])

    # Adjust accumulate_grad_batches based on the number of GPUs
    desired_effective_batch_size = config['training']["desired_effective_batch_size"]
    actual_batch_size = config['training']['batch_size']
    accumulate_grad_batches = desired_effective_batch_size // (actual_batch_size * num_gpus)
    accumulate_grad_batches = max(accumulate_grad_batches, 1)  # Ensure at least 1

    # Initialize the DDPStrategy with find_unused_parameters=False
    strategy = DDPStrategy(find_unused_parameters=False)

    # Initialize the Trainer with DDPStrategy
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        accelerator='gpu',
        devices=config["devices"],
        strategy=strategy,  # Use the DDPStrategy with find_unused_parameters=False
        precision=16 if config['training'].get('precision', 32) == 16 else 32,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=5,
        logger=logger,
        enable_progress_bar=True,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps= 25 if len(config["devices"]) == 8 else 50,
    )

    # Start training
    trainer.fit(model, datamodule=dataset,ckpt_path="./checkpoints_AdjustedHyperparams_Freeze_HeaviestAug_HeavyColorJitter/dino-epoch=134-val_loss=6.29.ckpt")
    
    run_evaluation(cb=checkpoint_callback.best_model_path,name=config["name"],recompute=True,draw_plot=True,save_reps=True)
    