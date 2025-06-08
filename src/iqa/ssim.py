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

    def evaluate(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        '''
        Computes the Structural Similarity (SSIM) index between two image tensors.

        :param img1: original image tensor with shape [C, H, W]
        :param img2: manipulated image tensor with shape [C, H, W]
        :return: ssim score (float), higher is better
        '''
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

        channels = img1.size(1)
        kernel = self._get_kernel(channels, img1.device)

        # Fuse related computations
        mu1 = F.conv2d(img1, kernel, padding=self.padding, groups=channels)
        mu2 = F.conv2d(img2, kernel, padding=self.padding, groups=channels)

        # Batch the squared convolutions
        img_concat = torch.cat([img1 * img1, img2 * img2, img1 * img2], dim=1)
        kernel_repeated = kernel.repeat(3, 1, 1, 1)
        conv_results = F.conv2d(
            img_concat, kernel_repeated, padding=self.padding, groups=channels * 3)

        # Split results
        conv1_sq = conv_results[:, :channels]
        conv2_sq = conv_results[:, channels:2*channels]
        conv12 = conv_results[:, 2*channels:3*channels]

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = conv1_sq - mu1_sq
        sigma2_sq = conv2_sq - mu2_sq
        sigma12 = conv12 - mu1_mu2

        # SSIM map
        ssim_map = ((2.0 * mu1_mu2 + self.C1) * (2.0 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        return ssim_map.mean().item()

__all__ = ['SSIM']