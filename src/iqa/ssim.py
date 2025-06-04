import torch
import torch.nn.functional as F

class SSIM:
    def __init__(self, source_image: torch.Tensor, manipulated_image: torch.Tensor):
        self.source = source_image
        self.manipulated = manipulated_image

    def create_gaussian_kernel(window_size: int, sigma: float, channels: int) -> torch.Tensor:
        coords = torch.arange(window_size).float() - (window_size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        g2d = g.unsqueeze(1) * g.unsqueeze(0)
        g2d = g2d.unsqueeze(0).unsqueeze(0)
        return g2d.expand(channels, 1, window_size, window_size)

    def evaluate(self, window_size: int = 11, sigma: float = 1.5) -> float:
        '''
        Computes the Structural Similarity (SSIM) index between two image tensors.
        
        :param img1: represents original image (as torch tensor) with shape [C, H, W]
        :param img2: represents manipulated image (as torch tensor) with shape [C, H, W]
        :return: ssim score 

        output of SSIM: Higher values indicate better similarity; a value of 1 means perfect similarity.
        '''
        data_range = 255.0
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        if self.source.dim() == 3:
            self.source = self.source.unsqueeze(0)
        if self.manipulated.dim() == 3:
            self.manipulated = self.manipulated.unsqueeze(0)
        channels = self.source.size(1)
        window = self.create_gaussian_kernel(window_size, sigma, channels).to(self.source.device)
        mu1 = F.conv2d(self.source, window, padding=window_size // 2, groups=channels)
        mu2 = F.conv2d(self.manipulated, window, padding=window_size // 2, groups=channels)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(self.source * self.source, window, padding=window_size // 2, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(self.manipulated * self.manipulated, window, padding=window_size // 2, groups=channels) - mu2_sq
        sigma12 = F.conv2d(self.source * self.manipulated, window, padding=window_size // 2, groups=channels) - mu1_mu2
        ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean().item()
