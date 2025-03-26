import torch
from .rmse import getRMSE

def getERGAS(img1: torch.Tensor, img2: torch.Tensor, scale: float = 1.0) -> float:
    '''
    Computes the ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse) metric between two image tensors.
    
    :param img1: represents the original image (as a torch tensor) with shape [C, H, W]
    :param img2: represents the manipulated image (as a torch tensor) with shape [C, H, W]
    :param scale: the scale factor corresponding to the ratio (d/r) between pixel sizes
    :return: a float representing the ERGAS metric
    
    output of ERGAS: Lower values indicate better quality; a value of 0 means perfect reconstruction.
    '''
    B = img1.size(0)
    mu = img1.mean(dim=(1, 2))
    rmse = getRMSE(img1, img2)
    rel_sq = (rmse / mu) ** 2
    ergas = 100 * scale * torch.sqrt(rel_sq.mean()).item()
    return ergas
