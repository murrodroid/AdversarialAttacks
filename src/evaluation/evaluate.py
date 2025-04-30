from src.iqa.select import select_iqa_method
from src.utils.image import image_to_tensor
import pandas as pd

def evaluate_image(source_image: str, attack_image: str, source_class: int, attack_class: int, metadata: pd.DataFrame = None, iqa_method: str = 'psnr') -> dict:
    '''
    This function evaluates the performance of an adversarial attack on a given image.
    inputs:
        source_image: The path to the original image.
        attack_image: The path to the adversarial image.
        source_class: The true class of the original image.
        atack_class: The class given by the adversarial attack.
        iqa_method: The image quality assessment method to use. Options are 'ergas', 'psnr', 'rmse', 'sam', 'ssim'. Default is 'psnr'.
    outputs:
        dict: A dictionary containing the evaluation results.
            ('attack_success': bool,            : True if the attack was successful, False otherwise.
            'source_class': int,                : The true class of the original image.
            'attack_class': int,                : The class given by the adversarial attack.
            'iqa_method': str,                  : The name of the image quality assessment method used.
            'iqa': float,                       : The value of the image quality assessment method.
            'source_source_confidence': float,  : The confidence of the source class in the source image.
            'source_attack_confidence': float,  : The confidence of the source class in the attack image.
            'attack_source_confidence': float,  : The confidence of the attack class in the source image.
            'attack_attack_confidence': float,  : The confidence of the attack class in the attack image.
            )
    '''

    source_image = image_to_tensor(source_image)
    attack_image = image_to_tensor(attack_image)

    attack_success = False if source_class == attack_class else True
    iqa = select_iqa_method(iqa_method)(source_image, attack_image)

    source_source_confidence = metadata['source_source_confidence'].values[0] if metadata is not None else 0.0
    source_attack_confidence = metadata['source_attack_confidence'].values[0] if metadata is not None else 0.0
    attack_source_confidence = metadata['attack_source_confidence'].values[0] if metadata is not None else 0.0
    attack_attack_confidence = metadata['attack_attack_confidence'].values[0] if metadata is not None else 0.0

    return {
        'attack_success': attack_success,
        'source_class': source_class,
        'attack_class': attack_class,
        'iqa_method': iqa_method,
        'iqa': iqa,
        'source_source_confidence': source_source_confidence,
        'source_attack_confidence': source_attack_confidence,
        'attack_source_confidence': attack_source_confidence,
        'attack_attack_confidence': attack_attack_confidence
    }

if __name__ == '__main__':
    source_image = 'original.png'
    attack_image = 'adversarial.png'

    print(evaluate_image(source_image, attack_image, 1, 1))