import torch

class PSNR:
    def evaluate(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        '''
        Computes Peak Signal to Noise Ratio between two 4D tensors.

        :param img1: original image (as torch tensor)
        :param img2: manipulated image (as torch tensor)
        :return: psnr score (float), higher is better
        '''
        diff = torch.abs(img1 - img2).float()
        sse = (diff ** 2).sum()

        if sse.item() <= 1e-12:
            psnr = 0
        else:
            mse = sse / img1.numel()
            psnr = 10.0 * torch.log10((255 * 255) / mse)

        return float(psnr)

__all__ = ['PSNR']
