import torch
import torch.nn as nn
from torch import Tensor
import numpy as np



def fgsm_attack(model: object, image: Tensor, target_class: int, epsilon: float = 0.1, max_iters: int = 100, break_early: bool = False) -> Tensor:
    perturbed_image = image.clone().detach().requires_grad_(True)
    target = torch.tensor([target_class], device=perturbed_image.device)
    citerion = nn.CrossEntropyLoss()
    success = False

    first_success_iter = None
    first_success_output = None
    final_output = None

    for i in range(max_iters):
        output = model(perturbed_image)
        pred_class = output.argmax(dim=1).item()
        if pred_class == target_class:
            if not success:
                first_success_iter = i
                first_success_output = output.detach().clone()
            success = True
            
            if break_early:
                final_output = output.detach().clone()
                break
        
        loss = -citerion(output, target) 
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            perturbed_image += epsilon * perturbed_image.grad.sign()
            delta = perturbed_image - image
            delta = torch.clamp(delta, -epsilon * 10, epsilon * 10)  
            perturbed_image = image + delta
        
        perturbed_image.requires_grad_(True)

        if i == max_iters - 1:
            final_output = output.detach().clone()
    
    return perturbed_image, success, first_success_iter, first_success_output, final_output

