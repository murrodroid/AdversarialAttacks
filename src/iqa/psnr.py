import torch


class PSNR:
    def evaluate(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        '''
        Computes Peak Signal to Noise Ratio between two 4D tensors.

        :param img1: original image (as torch tensor)
        :param img2: manipulated image (as torch tensor)
        :return: psnr score (float), higher is better
        '''
        mse = torch.mean((img1.float() - img2.float()) ** 2)

        if mse.item() <= 1e-12:
            return 0.0

        psnr = 10.0 * torch.log10(65025.0 / mse)  # 65025 = 255^2
        return psnr.item()


__all__ = ['PSNR']
