import torch
import torch.nn.functional as F


class SSIM:
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        self.window_size = window_size
        self.sigma = sigma
        self.padding = window_size // 2
        self.C1 = (0.01 * 255.0) ** 2
        self.C2 = (0.03 * 255.0) ** 2
        self._kernel_cache = {}

    def _get_kernel(self, channels: int, device: torch.device) -> torch.Tensor:
        key = (channels, str(device))
        if key not in self._kernel_cache:
            x = torch.arange(self.window_size,
                             dtype=torch.float32, device=device)
            x = x - (self.window_size - 1) / 2
            g = torch.exp(-x.pow(2) / (2 * self.sigma ** 2))
            g = g / g.sum()
            kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)
            kernel = kernel_2d.unsqueeze(0).unsqueeze(
                0).expand(channels, 1, -1, -1).contiguous()
            self._kernel_cache[key] = kernel
        return self._kernel_cache[key]

    def evaluate(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        '''
        Computes the Structural Similarity (SSIM) index between two image tensors.

        :param img1: original image tensor with shape [C, H, W]
        :param img2: manipulated image tensor with shape [C, H, W]
        :return: ssim score (torch.Tensor), higher is better
        '''
        if img1.dim() == 3 and img2.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        channels = img1.size(1)
        kernel = self._get_kernel(channels, img1.device)
        mu1 = F.conv2d(img1, kernel, padding=self.padding, groups=channels)
        mu2 = F.conv2d(img2, kernel, padding=self.padding, groups=channels)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, kernel,
                             padding=self.padding, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, kernel,
                             padding=self.padding, groups=channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, kernel,
                           padding=self.padding, groups=channels) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
            ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        ssim = ssim_map.view(ssim_map.size(0), -1).mean(1)
        return ssim


__all__ = ['SSIM']
