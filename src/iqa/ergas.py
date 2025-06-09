import torch
from .rmse import RMSE


class ERGAS:
    @staticmethod
    def evaluate(img1: torch.Tensor, img2: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        '''
        Computes the ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse) metric between two image tensors.

        :param img1: original image tensor [C, H, W]
        :param img2: manipulated image tensor [C, H, W]
        :param scale: the scale factor corresponding to the ratio (d/r) between pixel sizes
        :return: ERGAS metric (torch.Tensor), lower is better
        '''
        if img1.dim() == 3 and img2.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        
        img1 = img1.float()
        img2 = img2.float()

        rmse = RMSE.evaluate(img1, img2)
        mu = img1.mean(dim=(2,3)).clamp(min=1e-6)

        ergas = 100 / scale * torch.sqrt(((rmse / mu) ** 2).mean(dim=1))

        return ergas


__all__ = ['ERGAS']
