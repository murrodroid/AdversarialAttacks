import torch
import torch.nn as nn
from torch import Tensor


def pgd_attack(model: object, source_image: Tensor, target_class: list, epsilon=0.3, alpha=0.01, max_iter=100, break_early: bool = False):
    perturbed_image = source_image.clone().detach(
    ) + torch.empty_like(source_image).uniform_(-epsilon/2, epsilon/2)
    perturbed_image = torch.clamp(
        perturbed_image, source_image.min(), source_image.max())
    target = torch.tensor(target_class, device=source_image.device)
    criterion = nn.CrossEntropyLoss()

    B = source_image.shape[0]
    success = [False] * B
    first_success_iter = [None] * B
    first_success_output = [None] * B
    final_output = [None] * B

    for i in range(max_iter):
        perturbed_image.requires_grad_(True)
        output = model(perturbed_image)
        probs = torch.softmax(output, dim=1).detach()
        pred_classes = probs.argmax(dim=1)

        for j in range(B):
            if pred_classes[j] == target[j]:
                if not success[j]:
                    first_success_iter[j] = i
                    first_success_output[j] = probs[j].detach(
                    ).clone().cpu().numpy().tolist()
                success[j] = True

        if break_early and all(success):
            final_output = [probs[j].detach().clone().cpu().numpy().tolist()
                            for j in range(B)]
            break

        loss = criterion(output, target)
        model.zero_grad()
        loss.backward()

        perturbed_image = perturbed_image.detach() - alpha * \
            torch.sign(perturbed_image.grad)
        eta = torch.clamp(perturbed_image - source_image, -epsilon, epsilon)
        perturbed_image = torch.clamp(
            source_image + eta, source_image.min(), source_image.max())

        if i == max_iter - 1:
            final_output = [probs[j].detach().clone().cpu().numpy().tolist()
                            for j in range(B)]

    return perturbed_image, success, first_success_iter, first_success_output, final_output
