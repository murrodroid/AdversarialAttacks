from torchvision.models import resnet50
import torch
from src.models.get_model import _replace_head
from src.models.get_model import get_finetuned_model


# changing architecture of resnet50 to be more robust
# by deepening the network and incorporting squeeze-and-excitation blocks

def robust_resnet(width_mult=1):
    # increase the number of channels throughout the model by 1.25
    model = resnet50(weights=None)
    print("model imported")
    model = _replace_head(model, "resnet", {"output_dim": 20})
    print("head replaced with output_dim=20")
    # deepening the network by adding more layers
    model.add_module("layer4",)
    # this is a simplified example, actual implementation may vary
    

    # adding squeeze-and-excitation blocks

    return model