import torch
from PIL import Image
from torchvision import transforms

_to_tensor = transforms.ToTensor()

def image_to_tensor(image_path: str) -> torch.Tensor:
    """
    Convert a .png image to a PyTorch tensor using torchvision.
    """
    image = Image.open(image_path).convert('RGB')
    return _to_tensor(image)