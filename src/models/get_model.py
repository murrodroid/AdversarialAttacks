import torch
from torchvision.models import mobilenet_v3_large, resnet50, swin_t
from torchvision.models import (MobileNet_V3_Large_Weights, ResNet50_Weights,
                                 Swin_T_Weights)

def get_model(name):
    builders = dict(
        mobilenet = (mobilenet_v3_large, MobileNet_V3_Large_Weights.DEFAULT),
        resnet    = (resnet50,          ResNet50_Weights.DEFAULT),
        swin      = (swin_t,            Swin_T_Weights.DEFAULT),
    )
    f, w = builders[name]
    return f(weights=w).eval().cuda() if torch.cuda.is_available() else f(weights=w).eval().cpu()