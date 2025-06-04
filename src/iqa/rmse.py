import torch

class RMSE:
    def evaluate(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        '''
        Computes the per-channel Root Mean Square Error (RMSE) between two image tensors.

        :param img1: original image tensor with shape [C, H, W]
        :param img2: manipulated image tensor with shape [C, H, W]
        :return: 1D torch tensor of RMSE values per channel
        '''
        diff_squared = (img1 - img2) ** 2
        mse_per_channel = diff_squared.mean(dim=(1, 2))
        rmse_per_channel = torch.sqrt(mse_per_channel)
        return rmse_per_channel

__all__ = ['RMSE']