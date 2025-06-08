import torch


class PSNR:
    def evaluate(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        '''
        Computes Peak Signal to Noise Ratio between two 4D tensors.

        :param img1: original image (as torch tensor)
        :param img2: manipulated image (as torch tensor)
        :return: psnr score (torch.Tensor), higher is better
        '''
        if img1.dim() == 3 and img2.dim() == 3:
            img1, img2 = img1.unsqueeze(0), img2.unsqueeze(0)
        mse = (img1.float() - img2.float()).pow(2).mean(dim=[1, 2, 3])
        mask = mse > 1e-12
        psnr = torch.zeros_like(mse)
        psnr[mask] = 10 * torch.log10(65025.0 / mse[mask])
        return psnr


__all__ = ['PSNR']
