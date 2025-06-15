import torch
from torchvision.models import mobilenet_v3_large, resnet50, swin_t
from torchvision.models import (MobileNet_V3_Large_Weights, ResNet50_Weights,
                                 Swin_T_Weights)
from pathlib import Path
import lzma

from src.finetuning.base import _replace_head
from src.utils.torch_util import getDevice
from src.finetuning.custom_swin_backbone import build_custom_swin_tiny_with_temp


def get_model(name):
    builders = dict(
        mobilenet = (mobilenet_v3_large, MobileNet_V3_Large_Weights.DEFAULT),
        resnet    = (resnet50,          ResNet50_Weights.DEFAULT),
        swin      = (swin_t,            Swin_T_Weights.DEFAULT),
    )
    f, w = builders[name]
    return f(weights=w).eval().cuda() if torch.cuda.is_available() else f(weights=w).eval().cpu()

def get_finetuned_model(name, device=getDevice(), cfg={"output_dim": 20}):
    """Get a finetuned model with the specified number of output classes.

    For models that don't have matching checkpoint files (like ImageNet20),
    this will load the backbone from ImageNet pretrained weights and
    initialize a new head with the specified output dimension.
    """
    builders = dict(
        mobilenet = mobilenet_v3_large,
        resnet    = resnet50,
        swin      = swin_t,
    )

    ckpt = Path("src/models/weights") / f"{name}{cfg.get('output_dim')}.pt.xz"
    model = _replace_head(builders[name](weights=None), name, cfg)

    if ckpt.exists():
        # Try to load the checkpoint
        try:
            with lzma.open(ckpt, "rb") as f:
                state_dict = torch.load(f, map_location=device, weights_only=True)
                model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                # If there's a size mismatch (e.g., different number of classes),
                # load only the backbone weights
                print(f"Size mismatch detected for {name}, loading backbone only...")
                with lzma.open(ckpt, "rb") as f:
                    state_dict = torch.load(f, map_location=device, weights_only=True)
                    model_state = model.state_dict()

                    # Copy weights that match in size
                    for key, value in state_dict.items():
                        if key in model_state and model_state[key].shape == value.shape:
                            model_state[key] = value

                    model.load_state_dict(model_state)
            else:
                raise e
    else:
        # No checkpoint exists, use pretrained ImageNet weights for backbone
        print(f"No checkpoint found for {name}, using pretrained ImageNet weights...")
        pretrained_model = builders[name](weights="DEFAULT")
        model_state = model.state_dict()
        pretrained_state = pretrained_model.state_dict()

        # Copy backbone weights
        for key, value in pretrained_state.items():
            if key in model_state and model_state[key].shape == value.shape:
                model_state[key] = value

        model.load_state_dict(model_state)

    return model.eval().to(device)

def get_robust_model(name, device=getDevice(), cfg={"output_dim": 20}):
    if name == "swin":
        model = build_custom_swin_tiny_with_temp(
            temperature=1.5,
            num_classes=cfg["output_dim"]
        )
    else:
        builders = dict(
            mobilenet = mobilenet_v3_large,
            resnet    = resnet50,
        )
        model = _replace_head(builders[name](weights=None), name, cfg)

    ckpt = Path('src/models/weights') / f'{name}20.pt.xz'
    with lzma.open(ckpt, "rb") as f:
        model.load_state_dict(torch.load(f, map_location=device), strict=True)

    return model.eval().to(device)
