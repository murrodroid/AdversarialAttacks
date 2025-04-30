import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

def cw_attack(model:object, image: Tensor, target_class: int, lr = int, steps = int, c = int, kappa = 0, break_early: bool = False) -> Tensor:
    """
    c = constant controlling the importance of misclassification vs. invisibility.
    kappa is a confidence margin. Higher means more confident misclassification.
    """
    
    #Bound adversarial image within valid range [0,1]
    """
    Allows us to optimize on an unconstrained variable w
    (image * 2 - 1) rescale image pixels from [0,1] to [-1,1]
    * 0.999999 multiply to avoid undefined parenthesis
    torch.atanh - apply to move input into tanh-space
    .requires_grad_() track gradients for w
    """
    w = torch.atanh((image * 2 - 1) * 0.999999).requires_grad_()

    #Create adam optimizer
    optimizer = torch.optim.Adam([w], lr=lr)
    
    #Loop for optimizing the adversarial example
    for _ in range(steps):
        #tahn reparameterization trick - ensures adversarial image stays in range [0,1]
        x_adv = 0.5 * (torch.tanh(w) + 1)

        #Feed adversarial image into model
        logits = model(x_adv)
        
        #logit score for the target class
        real = logits[0, target_class]

        #Gets highest logit scores among other classes
        other = torch.max(logits[0, torch.arange(logits.shape[1]) != y])
        
        if model.training:
            model.eval()  # Ensure deterministic behavior

        #misclassification loss
        loss1 = torch.clamp(other - real + kappa, min=0)

        #squared loss between adversarial and original image 
        loss2 = torch.sum((x_adv - image) ** 2)

        #objective function 
        loss = c * loss1 + loss2


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #returns final adversarial image
    adv = 0.5 * (torch.tanh(w) + 1).detach()
    
    return adv
    


