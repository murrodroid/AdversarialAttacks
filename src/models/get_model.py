import torch
from torchvision.models import mobilenet_v3_large, resnet50, swin_t
from torchvision.models import (MobileNet_V3_Large_Weights, ResNet50_Weights,
                                 Swin_T_Weights)
from pathlib import Path
from src.finetuning.base import _replace_head
from src.utils.torch_util import getDevice

def get_model(name):
    builders = dict(
        mobilenet = (mobilenet_v3_large, MobileNet_V3_Large_Weights.DEFAULT),
        resnet    = (resnet50,          ResNet50_Weights.DEFAULT),
        swin      = (swin_t,            Swin_T_Weights.DEFAULT),
    )
    f, w = builders[name]
    return f(weights=w).eval().cuda() if torch.cuda.is_available() else f(weights=w).eval().cpu()

def get_finetuned_model(name, device = getDevice()):
    builders = dict(
        mobilenet = mobilenet_v3_large,
        resnet    = resnet50,
        swin      = swin_t,
    )
    ckpt   = Path('src/models/weights') / f'{name}.pt'
    model  = _replace_head(builders[name](weights=None), name)
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    return model.eval().to(device)