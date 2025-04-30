import torch
import torch.nn as nn

def pgd_attack(model, image, target_class, epsilon=0.3, alpha=0.01, max_iter=100, break_early: bool = False):
  original = image.clone()
  adv = image.clone().detach() + torch.empty_like(image).uniform_(-epsilon/2, epsilon/2)
  adv = torch.clamp(adv, image.min(), image.max())
  target = torch.tensor([target_class], device=image.device)
  criterion = nn.CrossEntropyLoss()
  success = False
  for i in range(max_iter):
    adv.requires_grad = True
    output = model(adv)
    pred_class = output.argmax(dim=1).item()
    if pred_class == target_class:
      success = True
      if break_early: break
    loss = criterion(output, target)
    model.zero_grad()
    loss.backward()
    adv = adv.detach() - alpha * torch.sign(adv.grad.data)
    eta = torch.clamp(adv - original, -epsilon, epsilon)
    adv = torch.clamp(original + eta, image.min(), image.max())
  return adv, success