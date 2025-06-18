import torch
from torchvision.models import mobilenet_v3_large, resnet50, swin_t
from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights, Swin_T_Weights
from pathlib import Path
import lzma

from src.finetuning.base import _replace_head
from src.utils.torch_util import getDevice
from src.finetuning.robust_swin import robust_swin
from src.finetuning.robust_mobilenet import robust_mobilenet
from src.finetuning.robust_resnet import se_resnet50

def get_model(name):
    builders = dict(
        mobilenet = (mobilenet_v3_large, MobileNet_V3_Large_Weights.DEFAULT),
        resnet    = (resnet50,          ResNet50_Weights.DEFAULT),
        swin      = (swin_t,            Swin_T_Weights.DEFAULT),
    )
    f, w = builders[name]
    return f(weights=w).eval().cuda() if torch.cuda.is_available() else f(weights=w).eval().cpu()


def get_finetuned_model(device=getDevice(), cfg={"output_dim": 20, "model_name": "mobilenet", "adv": False}):
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

    if cfg["adv"]:
        ckpt = Path("src/models/weights") / f"{cfg['model_name']}{cfg.get('output_dim')}_adv.pt.xz"
    else:
        ckpt = Path("src/models/weights") / f"{cfg['model_name']}{cfg.get('output_dim')}.pt.xz"
    model = _replace_head(builders[cfg["model_name"]](weights=None), cfg["model_name"], cfg)

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
                print(f"Size mismatch detected for {cfg['model_name']}, loading backbone only...")
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
        print(f"No checkpoint found for {cfg['model_name']}, using pretrained ImageNet weights...")
        pretrained_model = builders[cfg['model_name']](weights="DEFAULT")
        model_state = model.state_dict()
        pretrained_state = pretrained_model.state_dict()

        # Copy backbone weights
        for key, value in pretrained_state.items():
            if key in model_state and model_state[key].shape == value.shape:
                model_state[key] = value

        model.load_state_dict(model_state)

    return model.eval().to(device)


def get_robust_model(
    device=getDevice(),
    cfg={"output_dim": 20, "width_mult": 1.5},
):
    if cfg["model_name"] == "swin":
        model = robust_swin(temperature=1.5, num_classes=cfg["output_dim"])
    elif cfg["model_name"] == "mobilenet":
        model = robust_mobilenet(width_mult=cfg["width_mult"], stem_kernel_size=7)
    elif cfg["model_name"] == "resnet":
        model = se_resnet50()

    model = _replace_head(model,cfg['model_name'],cfg)

    if cfg.get('robust_weights',None):
        path = Path(cfg['robust_weights'])
        loader_kwargs = {"map_location": device}
        try:
            loader_kwargs["weights_only"] = True  # PyTorch ≥ 2.1
        except TypeError:
            pass

        if path.suffix == ".xz":
            with lzma.open(path, "rb") as f:
                state_dict = torch.load(f, **loader_kwargs)
        else:
            state_dict = torch.load(path, **loader_kwargs)

        state_dict = state_dict.get("state_dict", state_dict)
        model.load_state_dict(state_dict, strict=False)

    return model.eval().to(device)
