from src.iqa import *

def select_iqa_method(iqa_method):
    """
    Selects the appropriate image quality assessment method based on the input string.
    
    Parameters:
        iqa_method (str): The name of the image quality assessment method to use.
                          Options are 'ergas', 'psnr', 'rmse', 'sam', 'ssim'.
    
    Returns:
        function: The selected image quality assessment function.
    
    Raises:
        ValueError: If the input string does not match any of the available methods.
    """
    
    iqa_methods = {
        'ergas': ergas.getERGAS,
        'psnr': psnr.getPSNR,
        'rmse': rmse.getRMSE,
        'sam': sam.getSAM,
        'ssim': ssim.getSSIM
    }
    
    return iqa_methods[iqa_method]