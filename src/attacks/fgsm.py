import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

@torch.compile(backend="eager")
def fgsm_attack(model: object, source_image: Tensor, target_class: list, epsilon: float = 0.1, max_iters: int = 100, break_early: bool = False) -> tuple[Tensor, list[bool], list[int | None], list[list[float] | None], list[list[float]]]:
    pert = source_image.clone().detach().requires_grad_(True)
    target = torch.tensor(target_class, device=pert.device)
    loss_fn = nn.CrossEntropyLoss()
    B = pert.shape[0]

    success = torch.zeros(B, dtype=torch.bool, device=pert.device)
    first_iter = torch.full((B,), -1, dtype=torch.int, device=pert.device)
    first_out = [None] * B
    final_out = None

    for i in range(max_iters):
        logits = model(pert)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)

        new_success = ~success & (pred == target)
        if new_success.any():
            idx = new_success.nonzero(as_tuple=True)[0]
            first_iter[idx] = i
            for j in idx.tolist():
                first_out[j] = probs[j].detach().cpu().numpy().tolist()
        success |= (pred == target)

        if break_early and success.all():
            final_out = [probs[j].detach().cpu().numpy().tolist()
                         for j in range(B)]
            break

        loss = -loss_fn(logits, target)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            pert += epsilon * pert.grad.sign()
            delta = pert - source_image
            pert = source_image + delta.clamp(-epsilon, epsilon)
        pert.requires_grad_(True)
    else:
        final_out = [probs[j].detach().cpu().numpy().tolist()
                     for j in range(B)]

    pert = pert.detach()
    success_list = success.cpu().tolist()
    first_iter_list = [
        int(n) if n >= 0 else None
        for n in first_iter.cpu().tolist()
    ]
    return pert, success_list, first_iter_list, first_out, final_out
