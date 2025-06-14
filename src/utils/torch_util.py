import torch
import torchvision.transforms.functional as TF
from PIL import Image
import os
import lzma
import pathlib
import shutil

from src.finetuning.configs.base_finetune import train_cfg
from src.datasets.imagenet import ImageNet100, ImageNet20
from src.datasets.cifar10 import Cifar10


def getDevice() -> torch.device:
    if torch.cuda.is_available():
        rank = int(os.getenv("LOCAL_RANK", 0))
        return torch.device(f"cuda:{rank}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def getVRAM() -> int: 
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1024**3
        return round(total_gb)
    else: 
        return 4

def tensor_to_pil(tensor, dataset_name):
    """
    Converts a tensor (potentially normalized) back to a PIL Image.
    Assumes tensor is [B, C, H, W] or [C, H, W].
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.detach().cpu()

    try:
        if dataset_name == "cifar10":
            tensor = Cifar10.inverse_transforms(tensor)
        elif dataset_name == "imagenet20":
            tensor = ImageNet20.inverse_transforms(tensor)
        elif dataset_name == "imagenet100":
            tensor = ImageNet100.inverse_transforms(tensor)
    except Exception as e:
        print(
            f"Warning: Error during inverse_transform: {e}. Clamping tensor.")

    tensor = torch.clamp(tensor, 0, 1)

    try:
        pil_image = TF.to_pil_image(tensor)
    except Exception as e:
        print(f"Error converting tensor to PIL image: {e}")
        pil_image = Image.new(
            "RGB", (tensor.shape[2], tensor.shape[1]), color="black")

    return pil_image


def unnormalize_tensor(tensor, dataset_name):
    """
    Convert normalized tensor back to [0,1] range for the specified dataset.

    Args:
        tensor: Normalized tensor in dataset-specific range
        dataset_name: Name of the dataset ("cifar10", "imagenet100", etc.)

    Returns:
        Tensor in [0,1] range suitable for attacks that need unnormalized input
    """
    tensor = tensor.detach().clone()

    try:
        if dataset_name == "cifar10":
            tensor = Cifar10.inverse_transforms(tensor)
        elif dataset_name == "imagenet100":
            tensor = ImageNet100.inverse_transforms(tensor)
        elif dataset_name == "imagenet20":
            tensor = ImageNet20.inverse_transforms(tensor)
        else:
            print(
                f"Warning: Unknown dataset '{dataset_name}'. Assuming tensor is already unnormalized."
            )
    except Exception as e:
        print(
            f"Warning: Error during inverse_transform for {dataset_name}: {e}. Tensor may not be properly unnormalized."
        )

    tensor = torch.clamp(tensor, 0, 1)
    return tensor


def normalize_tensor(tensor, dataset_name):
    """
    Convert [0,1] range tensor to normalized range for the specified dataset.

    Args:
        tensor: Tensor in [0,1] range
        dataset_name: Name of the dataset ("cifar10", "imagenet100", etc.)

    Returns:
        Normalized tensor in dataset-specific range
    """
    tensor = tensor.detach().clone()

    try:
        if dataset_name == "cifar10":
            transform = TF.normalize(
                tensor, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            )
            return transform
        elif dataset_name == "imagenet20":
            transform = TF.normalize(
                tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )
            return transform
        elif dataset_name == "imagenet100":
            transform = TF.normalize(
                tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )
            return transform
        else:
            # For unknown datasets, return as-is
            print(
                f"Warning: Unknown dataset '{dataset_name}'. Returning tensor without normalization."
            )
            return tensor
    except Exception as e:
        print(
            f"Warning: Error during normalization for {dataset_name}: {e}. Returning original tensor."
        )
        return tensor

def save_compressed(file_path: str | pathlib.Path, level: int = 9):
    src = pathlib.Path(file_path).expanduser().resolve()
    dst = src.with_suffix(src.suffix + ".xz")          # model.pt → model.pt.xz
    if dst.exists():
        return                                          # already compressed
    with open(src, "rb") as fp_in, lzma.open(dst, "wb", preset=level) as fp_out:
        shutil.copyfileobj(fp_in, fp_out) 