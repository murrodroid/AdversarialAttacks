import unittest
from PIL import Image
from src.iqa import *
from src.datasets.cifar10 import Cifar10


class TestImageQualityAssessment(unittest.TestCase):
    bird_original = Image.open("tests/assets/bird_original.png")
    bird_perturbed = Image.open("tests/assets/bird_perturbed.png")
    original_tensor = Cifar10.transforms(bird_original).unsqueeze(0)
    perturbed_tensor = Cifar10.transforms(bird_perturbed).unsqueeze(0)

    def test_psnr(self):
        psnr = PSNR()
        result = psnr.evaluate(self.original_tensor,
                               self.perturbed_tensor).item()
        self.assertAlmostEqual(result, 69.47459411621094, places=6)

    def test_psnr_full_similarity(self):
        psnr = PSNR()
        result = psnr.evaluate(self.original_tensor,
                               self.original_tensor).item()
        self.assertEqual(result, 0)

    def test_ssim(self):
        ssim = SSIM(window_size=11, sigma=1.5)
        result = ssim.evaluate(
            self.original_tensor, self.perturbed_tensor).item()
        self.assertAlmostEqual(result, 0.999845027923584, places=6)

    def test_ssim_full_similarity(self):
        ssim = SSIM(window_size=11, sigma=1.5)
        result = ssim.evaluate(self.original_tensor,
                               self.original_tensor).item()
        self.assertEqual(result, 1)


if __name__ == '__main__':
    unittest.main()
