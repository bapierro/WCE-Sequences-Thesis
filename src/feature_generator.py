import torch
import torch.nn as nn
from torchvision.models import resnet18,resnet34,resnet50,resnet101,resnet152,ResNet18_Weights,ResNet34_Weights,ResNet50_Weights, ResNet
from model_name import Model
from EndoFM import vision_transformer
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2



DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

depth_anything_model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

depth_anything_encoder = 'vitb' # or 'vits', 'vitb', 'vitg'

"""_summary_
    Returns:
        _type_: _description_
    """
class FeatureGenerator:
    def __init__(self,model_name : Model, pretrained = True):
        self.model = self._get_model(model_name, pretrained)
        
        
    def _get_model(self, model_name: Model, pretrained):
        match model_name:
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
                state_dict = torch.load(weights_path, map_location=torch.device('cpu'))["student"]

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
                if Model.DEPTH_ANY_SMALL:
                    depth_anything_encoder = 'vits'
                if Model.DEPTH_ANY_BASE:
                    depth_anything_encoder = 'vitb'
                else:
                    depth_anything_encoder = 'vitl'
                    
                    
                model = DepthAnythingV2(**depth_anything_model_configs[depth_anything_encoder])
                d_a_v2_weights = f"./Depth_Anything_V2/checkpoints/depth_anything_v2_{depth_anything_encoder}.pth"
                model.load_state_dict(
                    torch.load(d_a_v2_weights, map_location="cpu")
                    )
                
        if not pretrained:
            model.apply(self._init_weights)
        if model_name not in (Model.ENDO_FM,Model.DEPTH_ANY_SMALL, Model.DEPTH_ANY_BASE, Model.DEPTH_ANY_LARGE):
            model = nn.Sequential(*list(model.children())[:-1])
        return model
        
        
    
    def _init_weights(self,m):
        if isinstance(m,(nn.Conv2d,nn.Linear)):
            nn.init.kaiming_normal(m.weight)
            if m.bias is not None:
                nn.init.constant(m.bias,0)
        
        
    def generate(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.model(image)
        return features
            
    
    