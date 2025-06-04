import unittest
from PIL import Image
from src.iqa import *
from src.datasets.cifar10 import Cifar10


class TestImageQualityAssessment(unittest.TestCase):
    bird_original = Image.open("tests/assets/bird_original.png")
    bird_perturbed = Image.open("tests/assets/bird_perturbed.png")

    dataset = Cifar10()

    original_tensor = dataset.transform(bird_original).unsqueeze(0)
    perturbed_tensor = dataset.transform(bird_perturbed).unsqueeze(0)

    def test_psnr(self):
        psnr = PSNR(self.original_tensor, self.perturbed_tensor)
        result = psnr.evaluate()
        self.assertAlmostEqual(result, 69.47459411621094, places=6)

    def test_psnr_full_similarity(self):
        psnr = PSNR(self.original_tensor, self.original_tensor)
        result = psnr.evaluate()
        self.assertEqual(result, 0)

    def test_ssim(self):
        ssim = SSIM(self.original_tensor, self.perturbed_tensor)
        result = ssim.evaluate(window_size=11, sigma=1.5)
        self.assertAlmostEqual(result, 0.999845027923584, places=6)

    def test_ssim_full_similarity(self):
        ssim = SSIM(self.original_tensor, self.original_tensor)
        result = ssim.evaluate()
        self.assertEqual(result, 1)

if __name__ == '__main__':
    unittest.main()