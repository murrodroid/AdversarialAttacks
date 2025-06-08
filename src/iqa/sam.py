import torch
import math


class SAM:
    def getSAM(self, img1: torch.Tensor, img2: torch.Tensor, eps: float = 1e-8, return_degrees: bool = True) -> torch.Tensor:
        '''
        Computes the Spectral Angle Mapper (SAM) between two image tensors.

        :param img1: original image tensor with shape [C, H, W]
        :param img2: manipulated image tensor with shape [C, H, W]
        :param eps: small constant to avoid division by zero
        :param return_degrees: whether to return the angle in degrees (default) or radians
        :return: average spectral angle across all pixels
        '''
        dot = (img1 * img2).sum(dim=0)
        norm1 = torch.sqrt((img1 ** 2).sum(dim=0) + eps)
        norm2 = torch.sqrt((img2 ** 2).sum(dim=0) + eps)
        cos_theta = dot / (norm1 * norm2)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        angle = torch.acos(cos_theta)
        angle_mean = angle.mean()
        if return_degrees:
            angle_mean = angle_mean * 180.0 / math.pi
        return angle_mean


__all__ = ['SAM']
