import torch
from .rmse import RMSE

class ERGAS:
    def evaluate(self, img1: torch.Tensor, img2: torch.Tensor, scale: float = 1.0) -> float:
        '''
        Computes the ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse) metric between two image tensors.
        
        :param img1: original image tensor [C, H, W]
        :param img2: manipulated image tensor [C, H, W]
        :param scale: the scale factor corresponding to the ratio (d/r) between pixel sizes
        :return: ERGAS metric (float), lower is better
        '''
        mu = img1.mean(dim=(1, 2))
        rmse = RMSE().evaluate(img1, img2)
        rel_sq = (rmse / mu) ** 2
        ergas = 100 * scale * torch.sqrt(rel_sq.mean()).item()
        return ergas

__all__ = ['ERGAS']