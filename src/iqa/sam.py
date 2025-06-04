import torch
import math

class SAM:
    def __init__(self, source_image: torch.Tensor, manipulated_image: torch.Tensor):
        self.source = source_image
        self.manipulated = manipulated_image

    def getSAM(self, eps: float = 1e-8, return_degrees: bool = True) -> float:
        '''
        Computes the Spectral Angle Mapper (SAM) between two image tensors.
        
        :param img1: represents the original image (as a torch tensor) with shape [C, H, W]
        :param img2: represents the manipulated image (as a torch tensor) with shape [C, H, W]
        :param eps: a small constant to avoid division by zero
        :param return_degrees: if True, returns the angle in degrees; otherwise, in radians
        :return: the average spectral angle across all pixels

        output of SAM: Lower values indicate better spectral similarity; a value of 0 means no spectral distortion.
        '''
        dot = (self.source * self.manipulated).sum(dim=0)
        norm1 = torch.sqrt((self.source ** 2).sum(dim=0) + eps)
        norm2 = torch.sqrt((self.manipulated ** 2).sum(dim=0) + eps)
        cos_theta = dot / (norm1 * norm2)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        angle = torch.acos(cos_theta)
        angle_mean = angle.mean()
        if return_degrees:
            angle_mean = angle_mean * 180.0 / math.pi
        return angle_mean.item()

__all__ = ['SAM']