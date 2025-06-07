import torch
import torchvision.transforms.functional as TF
from src.datasets.cifar10 import Cifar10
from PIL import Image


def getDevice():
    device = 'cuda' if torch.cuda.is_available(
    ) else 'mps' if torch.backends.mps.is_available() else 'cpu'
    return device


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
