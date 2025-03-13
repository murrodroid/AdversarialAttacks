import torch
import torch.nn as nn

def pgd_attack(model, image, target_class, device, epsilon=0.3, alpha=0.01, max_iter=100):
  original = image.clone()
  adv = image.clone().detach() + torch.empty_like(image).uniform_(-epsilon/2, epsilon/2)
  adv = torch.clamp(adv, image.min(), image.max())
  target = torch.tensor([target_class], device=device)
  criterion = nn.CrossEntropyLoss()
  for i in range(max_iter):
    adv.requires_grad = True
    output = model(adv)
    loss = criterion(output, target)
    model.zero_grad()
    loss.backward()
    adv = adv.detach() - alpha * torch.sign(adv.grad.data)
    eta = torch.clamp(adv - original, -epsilon, epsilon)
    adv = torch.clamp(original + eta, image.min(), image.max())
    if model(adv).argmax(1).item() == target_class:
      print(f"Attack successful after {i+1} iterations!")
      break
  return adv