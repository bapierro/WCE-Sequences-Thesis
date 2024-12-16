import argparse
import copy
from typing import Any

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
from lightly.models.utils import deactivate_requires_grad, update_momentum
from vision_transformer import VisionTransformer
from seeai_datamodule import SEEAIDataModule
import yaml

parser = argparse.ArgumentParser("Transform configs")
parser.add_argument("--config", type=str, required=True, help="path to the YAML config file")


class DINO(pl.LightningModule):
    def __init__(self, pretrained=False, img_size=400, ):
        super(DINO, self).__init__()
        self.save_hyperparameters()
        self.config = config
        vitb16_s = VisionTransformer(img_size=[img_size])
        vitb16_t = copy.deepcopy(vitb16_s)
        input_dim = vitb16_s.embed_dim
        if pretrained:
            state_dict_s = torch.load("./EndoFM/endo_fm.pth", map_location=torch.device('cpu'))["student"]
            new_state_dict_s = {}
            for k, v in state_dict_s.items():
                if k.startswith('module.backbone.'):
                    new_key = k[len('module.backbone.'):]
                    new_state_dict_s[new_key] = v
                else:
                    new_state_dict_s[k] = v
            state_dict_t = torch.load("./EndoFM/endo_fm.pth", map_location=torch.device('cpu'))["teacher"]
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
        self.student_head = DINOProjectionHead(input_dim, 512, 64, 2048, freeze_last_layer=1)

        self.teacher_backbone = vitb16_t
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_head)
        deactivate_requires_grad(self.teacher_backbone)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

    def forward(self, x) -> Any:
        x = self.student_backbone(x).flatten(start_dim=1)
        x = self.teacher_head(x)
        return x

    def forward_teacher(self, x):
        x = self.teacher_backbone(x).flatten(start_dim=1)
        x = self.teacher_head(x)
        return x

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 10, 0.0996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        return loss

    def on_after_backward(self) -> None:
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self) -> Any:
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim


if __name__ == "__main__":
    args = parser.parse_args()

    # Load the YAML configuration file
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)

    # Extract transformation parameters from the YAML file
    transform_params = config["transforms"]
    dataset_path = config["path"]

    # Pass the loaded parameters to the DINOTransform class
    transform = DINOTransform(
        global_crop_size=transform_params.get("global_crop_size", 224),
        global_crop_scale=tuple(transform_params.get("global_crop_scale", (0.4, 1.0))),
        local_crop_size=transform_params.get("local_crop_size", 96),
        local_crop_scale=tuple(transform_params.get("local_crop_scale", (0.05, 0.4))),
        n_local_views=transform_params.get("n_local_views", 6),
        hf_prob=transform_params.get("hf_prob", 0.5),
        vf_prob=transform_params.get("vf_prob", 0.0),
        rr_prob=transform_params.get("rr_prob", 0.0),
        rr_degrees=transform_params.get("rr_degrees", None),
        cj_prob=transform_params.get("cj_prob", 0.8),
        cj_strength=transform_params.get("cj_strength", 0.5),
        cj_bright=transform_params.get("cj_bright", 0.8),
        cj_contrast=transform_params.get("cj_contrast", 0.8),
        cj_sat=transform_params.get("cj_sat", 0.4),
        cj_hue=transform_params.get("cj_hue", 0.2),
        random_gray_scale=transform_params.get("random_gray_scale", 0.2),
        gaussian_blur=tuple(transform_params.get("gaussian_blur", (1.0, 0.1, 0.5))),
        sigmas=tuple(transform_params.get("sigmas", (0.1, 2))),
        solarization_prob=transform_params.get("solarization_prob", 0.2),
        normalize=transform_params.get("normalize", None)
    )

    # Initialize the model
    model = DINO(img_size=config["img_size"])
    dataset = SEEAIDataModule(dataset_path, footage_size=config["img_size"])
