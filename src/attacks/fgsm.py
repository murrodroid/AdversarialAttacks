import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


def fgsm_attack(model: object, source_image: Tensor, target_class: list, epsilon: float = 0.1, max_iters: int = 100, break_early: bool = False) -> tuple[Tensor, list[bool], list[int | None], list[list[float] | None], list[list[float]]]:
    perturbed_image = source_image.clone().detach().requires_grad_(True)
    target = torch.tensor(target_class, device=perturbed_image.device)
    criterion = nn.CrossEntropyLoss()

    B = source_image.shape[0]
    success = [False] * B
    first_success_iter = [None] * B
    first_success_output = [None] * B
    final_output = [None] * B

    for i in range(max_iters):
        output = model(perturbed_image)
        probs = torch.softmax(output, dim=1).detach()
        pred_classes = probs.argmax(dim=1)

        for j in range(B):
            if pred_classes[j].item() == target[j].item():
                if not success[j]:
                    first_success_iter[j] = i
                    first_success_output[j] = (
                        probs[j].detach().clone().cpu().numpy().tolist()
                    )
                success[j] = True

        if break_early and all(success):
                final_output = [probs[j].detach().clone().cpu().numpy().tolist() for j in range(B)]
                break

        loss = -criterion(output, target)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            perturbed_image += epsilon * perturbed_image.grad.sign()
            delta = perturbed_image - source_image
            delta = torch.clamp(delta, -epsilon * 10, epsilon * 10)
            perturbed_image = source_image + delta

        perturbed_image.requires_grad_(True)

        if i == max_iters - 1:
            final_output = [probs[j].detach().clone().cpu().numpy().tolist() for j in range(B)]

    return perturbed_image, success, first_success_iter, first_success_output, final_output
