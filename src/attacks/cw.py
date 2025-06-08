import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


def cw_attack(
    model: object,
    source_image: Tensor,
    target_class: list,
    lr: float = 0.01,
    steps: int = 10,
    c: float = 1.0,
    kappa: int = 0,
    break_early: bool = False,
) -> tuple[
    Tensor, list[bool], list[int | None], list[list[float] | None], list[list[float]]
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

    success = [False] * B
    first_success_iter = [None] * B
    first_success_output = [None] * B
    final_output = [None] * B

    # Loop for optimizing the adversarial example
    for i in range(steps):
        # tahn reparameterization trick - ensures adversarial image stays in range [0,1]
        x_adv = 0.5 * (torch.tanh(w) + 1)

        # Feed adversarial image into model
        output = model(x_adv)
        probs = torch.softmax(output, dim=1).detach()

        # Get predicted classes for each sample in batch
        pred_classes = probs.argmax(dim=1)

        # Check if the predicted class matches the target class for each sample
        for j in range(B):
            if pred_classes[j] == target[j]:
                if not success[j]:
                    success[j] = True
                    first_success_iter[j] = i
                    first_success_output[j] = (
                        probs[j].detach().clone().cpu().numpy().tolist()
                    )

        # Check if we should break early
        if break_early and all(success):
            final_output = [
                probs[j].detach().clone().cpu().numpy().tolist() for j in range(B)
            ]
            break

        # logit scores for the target classes (batched)
        real = output[torch.arange(B), target]

        # Gets highest logit scores among other classes for each sample
        other_logits = []
        for j in range(B):
            mask = torch.arange(output.shape[1], device=output.device) != target[j]
            other_max = torch.max(output[j, mask])
            other_logits.append(other_max)
        other = torch.stack(other_logits)

        if model.training:
            model.eval()  # Ensure deterministic behavior

        # misclassification loss (batched)
        loss1 = torch.clamp(other - real + kappa, min=0).sum()

        # squared loss between adversarial and original image (batched)
        loss2 = torch.sum((x_adv - source_image) ** 2)

        # objective function
        loss = c * loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store final output on last iteration
        if i == steps - 1:
            final_output = [
                probs[j].detach().clone().cpu().numpy().tolist() for j in range(B)
            ]

    # returns final adversarial image
    perturbed_image = 0.5 * (torch.tanh(w) + 1).detach()

    return (
        perturbed_image,
        success,
        first_success_iter,
        first_success_output,
        final_output,
    )
