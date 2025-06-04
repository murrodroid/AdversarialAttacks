import torch

class RMSE:
    def __init__(self, source_image: torch.Tensor, manipulated_image: torch.Tensor):
        self.source = source_image
        self.manipulated = manipulated_image

    def evaluate(self) -> torch.Tensor:
        '''
        Computes the per-channel Root Mean Square Error (RMSE) between two image tensors.
        
        :param img1: represents the original image (as a torch tensor) with shape [C, H, W]
        :param img2: represents the manipulated image (as a torch tensor) with shape [C, H, W]
        :return: a 1D torch tensor containing RMSE values for each channel

        output of RMSE: Lower values indicate better quality; a value of 0 means perfect similarity.
        '''
        diff_squared = (self.source - self.manipulated) ** 2
        mse_per_channel = diff_squared.mean(dim=(1, 2))
        rmse_per_channel = torch.sqrt(mse_per_channel)
        return rmse_per_channel