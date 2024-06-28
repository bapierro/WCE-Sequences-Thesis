import torch
import torch.nn as nn
from torchvision.models import resnet18,resnet34,resnet50,ResNet18_Weights,ResNet34_Weights,ResNet50_Weights, ResNet
from model_name import Model


"""_summary_
    Returns:
        _type_: _description_
    """
class FeatureGenerator:
    def __init__(self,model_name : Model, pretrained = True):
        self.model = self._get_model(model_name, pretrained)
        
        
    def _get_model(self,model_name : Model, pretrained):
        match model_name:
            case Model.RES_NET_18:
                model = resnet18(weights='DEFAULT' if pretrained else None)    
            case Model.RES_NET_34:
                model = resnet34(weights='DEFAULT' if pretrained else None)
            case Model.RES_NET_50:
                model = resnet50(weights='DEFAULT' if pretrained else None)
            
            
        if not pretrained:
            model.apply(self._init_weights)
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
            
    
    