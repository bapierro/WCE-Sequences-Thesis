import os
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152, swin_v2_b
)
from model_name import Model
from EndoFM import vision_transformer
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

import pytorch_lightning as pl  # Import PyTorch Lightning if needed

DEVICE = 'cuda:3' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

depth_anything_model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

depth_anything_encoder = 'vitb'  # Default encoder

class FeatureGenerator:
    def __init__(
        self,
        model_name: Model,
        pretrained=True,
        img_size=224,
        student=False,
        cb = None
    ):
        self.model = self._get_model(
            model_name, pretrained, img_size, student=student,cb=cb
        ).to(DEVICE)
        
    def _get_model(
        self, model_name: Model, pretrained, img_size, student,cb
    ):
        match model_name:
            case Model.Swin_v2_B:
                model = swin_v2_b(weights='DEFAULT' if pretrained else None)
            
            case Model.RES_NET_18:
                model = resnet18(weights='DEFAULT' if pretrained else None)
            case Model.RES_NET_34:
                model = resnet34(weights='DEFAULT' if pretrained else None)
            case Model.RES_NET_50:
                model = resnet50(weights='DEFAULT' if pretrained else None)
            case Model.RES_NET_101:
                model = resnet101(weights='DEFAULT' if pretrained else None)
            case Model.RES_NET_152:
                model = resnet152(weights='DEFAULT' if pretrained else None)
            case Model.ENDO_FM:
                vit = vision_transformer.VisionTransformer()
                weights_path = "./EndoFM/endo_fm.pth"
                state_dict = torch.load(weights_path, map_location='cpu')["student" if student else "teacher"]

                # Remove potential 'module.backbone.' prefix
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.backbone.'):
                        new_key = k[len('module.backbone.'):]
                        new_state_dict[new_key] = v
                    else:
                        new_state_dict[k] = v

                vit.load_state_dict(new_state_dict,strict=False)
                
                model = vit

            case Model.DEPTH_ANY_SMALL | Model.DEPTH_ANY_BASE | Model.DEPTH_ANY_LARGE:
                if model_name == Model.DEPTH_ANY_SMALL:
                    depth_anything_encoder = 'vits'
                elif model_name == Model.DEPTH_ANY_BASE:
                    depth_anything_encoder = 'vitb'
                else:
                    depth_anything_encoder = 'vitl'
                    
                model = DepthAnythingV2(**depth_anything_model_configs[depth_anything_encoder])
                d_a_v2_weights = f"./Depth_Anything_V2/checkpoints/depth_anything_v2_{depth_anything_encoder}.pth"
                model.load_state_dict(
                    torch.load(d_a_v2_weights, map_location=DEVICE)
                )
            
            case Model.CENDO_FM:
                vit = vision_transformer.VisionTransformer()
                
                # checkpoint_path = "checkpoints_OrientationAugments/dino-epoch=54-val_loss=5.60.ckpt"
                #checkpoint_path = "checkpoints_HeavyAugment/Hide/dino-epoch=129-val_loss=5.45.ckpt" #HeavyAugment Light
                #checkpoint_path = "checkpoints_HeavyAugment/dino-epoch=109-val_loss=5.76.ckpt" # Mehr local views FPS->13
                #checkpoint_path = "checkpoints_KeinCJAberGaussHeavy8GPU/dino-epoch=179-val_loss=5.83.ckpt" # KeinCJ aber straker gauß FPS-->9
                #checkpoint_path = "checkpoints_HeavyAugmentLowLR/dino-epoch=74-val_loss=5.20.ckpt" 
                # checkpoint_path = "checkpoints_HeavyAugmentLowLR_OVERLAPPING_CROPS_4Views/dino-epoch=59-val_loss=4.85.ckpt" <----
                #checkpoint_path = "checkpoints_HeavyAugmentLowLR_OVERLAPPING_CROPS_6Views/dino-epoch=69-val_loss=4.92.ckpt"
                #checkpoint_path = "checkpoints_HeavyAugment_NoCJ/dino-epoch=79-val_loss=4.08.ckpt"# KeinCJ
                #checkpoint_path = "checkpoints_HeavyAugmentBigLocalVies/dino-epoch=89-val_loss=4.71.ckpt"# biggerLocal Views
                # checkpoint_path = "checkpoints_LightAugmentsDinoEsque/dino-epoch=44-val_loss=7.98.ckpt"# Basic dino transforms
                #checkpoint_path = "checkpoints_HeavyAugment_OVERLAPPING_CROPS_4_BigDinoHEAD/dino-epoch=79-val_loss=7.52.ckpt"
                # checkpoint_path = "checkpoints_Larger OC/dino-epoch=69-val_loss=4.75.ckpt" # 
                # checkpoint_path = "checkpoints_GlobalEqLocalCrop6LV/dino-epoch=59-val_loss=4.69.ckpt" #<--- Bestes atm
                # checkpoint_path = "./checkpoints_GlobalEqLocalCrop/dino-epoch=64-val_loss=4.70.ckpt"
                # checkpoint_path = "/checkpoints_GlobalEqLocalCropBigLovalView/dino-epoch=74-val_loss=3.53.ckpt"
                # checkpoint_path = "checkpoints_AdjustedHyperparams/dino-epoch=64-val_loss=4.50.ckpt"
                # checkpoint_path = "./checkpoints_AdjustedHyperparams_Freeze/dino-epoch=44-val_loss=4.00.ckpt"
                # checkpoint_path = "checkpoints_AdjustedHyperparams_Freeze_HeavierAugs/dino-epoch=74-val_loss=6.71.ckpt" 
                # checkpoint_path = "checkpoints_AdjustedHyperparams_Freeze_HeaviestAug/dino-epoch=114-val_loss=6.03.ckpt"#<--- BEstes atm
                checkpoint_path = "checkpoints_AdjustedHyperparams_Freeze_HeaviestAug_HeavyColorJitter/dino-epoch=229-val_loss=6.02.ckpt"
                if cb:
                    checkpoint_path = cb
                    
                if not os.path.isfile(checkpoint_path):
                    raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
                
                state_dict = torch.load(checkpoint_path, map_location=DEVICE)
                    
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        new_key = k[len('model.'):]
                        new_state_dict[new_key] = v
                    else:
                        new_state_dict[k] = v
                vit.load_state_dict(new_state_dict, strict=False)
                model = vit

            case _:
                raise ValueError(f"Unsupported model type: {model_name}")

        if not pretrained and model_name != Model.CENDO_FM:
            model.apply(self._init_weights)
        
        # Remove the final classification layer for feature extraction
        if model_name not in (
            Model.ENDO_FM,
            Model.DEPTH_ANY_SMALL,
            Model.DEPTH_ANY_BASE,
            Model.DEPTH_ANY_LARGE,
            Model.CENDO_FM,
            Model.Swin_v2_B
        ):
            model = nn.Sequential(*list(model.children())[:-1])
        
        # model = nn.DataParallel(model,device_ids=[3,4,5])
        return model
            
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def generate(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(DEVICE) 
        self.model.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            features = self.model(image)
        return features