import torch


class RMSE:
    @staticmethod
    def evaluate(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        '''
        Computes the per-channel Root Mean Square Error (RMSE) between two image tensors.

        :param img1: original image tensor with shape [C, H, W]
        :param img2: manipulated image tensor with shape [C, H, W]
        :return: 1D torch tensor of RMSE values per channel
        '''
        if img1.dim() == 3 and img2.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        diff_sq = (img1.float() - img2.float()).pow(2)
        mse_per_channel = diff_sq.mean(dim=(2, 3))
        rmse_per_channel = torch.sqrt(mse_per_channel)
        return rmse_per_channel


__all__ = ['RMSE']
