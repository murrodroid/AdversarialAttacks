import torch

class PSNR:
    def __init__(self, source_image: torch.Tensor, manipulated_image: torch.Tensor):
        self.source = source_image
        self.manipulated = manipulated_image
        
    def evaluate(self) -> float:
        '''
        Computes Peak Signal to Noise Ratio between two 4D tensors.

        :param img1: represents original image (as torch tensor)
        :param img2: represents manipulated image (as torch tensor)
        :return: psnr score

        output of PSNR: Higher values indicate better quality; typically, values above 30 dB are acceptable. 
        '''
        diff = torch.abs(self.source,self.manipulated)
        diff = diff.float()

        sse = (diff**2).sum()

        if sse.item() <= 1e-12:
            psnr = 0
        else:
            mse = sse / self.source.numel()
            psnr = 10.0 * torch.log10((255 * 255) / mse)
        
        return psnr
    
__all__ = ['PSNR']