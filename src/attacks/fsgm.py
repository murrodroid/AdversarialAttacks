import torch
import torch.nn as nn
from torch import Tensor


def fgsm_attack(model:object, image:Tensor, target_class:int, epsilon:int=0.3) -> Tensor:
  image.requires_grad = True
  output = model(image)
  loss = nn.CrossEntropyLoss()(output, torch.tensor([target_class], device=image.device))
  loss.backward()
  adv = image + epsilon * image.grad.sign()
  return torch.clamp(adv, image.min(), image.max())