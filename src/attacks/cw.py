import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

def cw_attack(
    model: object,
    source_image: Tensor,
    target_class: list,
    epsilon: float = 0.03,
    lr: float = 0.01,
    steps: int = 10,
    c: float = 1.0,
    kappa: int = 0,
    break_early: bool = False,
) -> tuple[
    Tensor, list[bool], list[int |
                             None], list[list[float] | None], list[list[float]]
]:
    """
    c = constant controlling the importance of misclassification vs. invisibility.
    kappa is a confidence margin. Higher means more confident misclassification.
    """

    # Bound adversarial image within valid range [0,1]
    """
    Allows us to optimize on an unconstrained variable w
    (source_image * 2 - 1) rescale image pixels from [0,1] to [-1,1]
    * 0.999999 multiply to avoid undefined parenthesis
    torch.atanh - apply to move input into tanh-space
    .requires_grad_() track gradients for w
    """
    w = torch.atanh((source_image * 2 - 1) * 0.999999).requires_grad_()

    # Create adam optimizer
    optimizer = torch.optim.Adam([w], lr=lr)

    B = source_image.shape[0]
    target = torch.tensor(target_class, device=source_image.device)

    # Get original predictions to avoid false positives
    with torch.no_grad():
        original_output = model(source_image)
        original_pred = original_output.argmax(dim=1)

    success = torch.zeros(B, dtype=torch.bool, device=source_image.device)
    first_iter = torch.full((B,), -1, dtype=torch.int,
                            device=source_image.device)
    first_out = [None] * B
    final_out = None
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    # Loop for optimizing the adversarial example
    for i in range(steps):
        # tahn reparameterization trick - ensures adversarial image stays in range [0,1]
        x_adv_unclipped = 0.5 * (torch.tanh(w) + 1)
        delta = torch.clamp(x_adv_unclipped - source_image, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(source_image + delta, 0, 1) 

        # Feed adversarial image into model
        output = model(x_adv)
        probs = torch.softmax(output, dim=1)

        # Get predicted classes for each sample in batch
        pred = probs.argmax(dim=1)
        # Track first successful hit for each sample (but don't update final success yet)
        currently_successful = (pred == target) & (pred != original_pred)
        new_successes = currently_successful & (first_iter == -1)

        # Record first success iteration and probabilities
        if new_successes.any():
            idx = new_successes.nonzero(as_tuple=True)[0]
            first_iter[idx] = i
            for j in idx.tolist():
                first_out[j] = probs[j].detach().cpu().tolist()

        if break_early and currently_successful.all():
            final_out = [probs[j].detach().cpu().tolist() for j in range(B)]
            break

        # logit scores for the target classes (batched)
        real = output[torch.arange(B), target]
        tmp = output.clone()
        tmp[torch.arange(B), target] = -1e10
        other = tmp.max(dim=1).values

        loss1 = torch.clamp(other - real + kappa, min=0)  # [B]
        loss2 = (x_adv - source_image).pow(2).view(B, -1).sum(dim=1)  # [B]
        loss = (c * loss1 + loss2).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        final_out = [probs[j].detach().cpu().tolist() for j in range(B)]

    final_pred = probs.argmax(dim=1)
    success = (final_pred == target) & (final_pred != original_pred)

    perturbed = x_adv.detach()
    return (
        perturbed,
        success.cpu().tolist(),
        [int(n) if n >= 0 else None for n in first_iter.cpu().tolist()],
        first_out,
        final_out,
    )
