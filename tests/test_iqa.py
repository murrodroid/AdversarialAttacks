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
        result = PSNR.evaluate(self.original_tensor,
                               self.perturbed_tensor).item()
        self.assertAlmostEqual(result, 69.47459411621094, places=6)

    def test_psnr_full_similarity(self):
        result = PSNR.evaluate(self.original_tensor,
                               self.original_tensor).item()
        self.assertEqual(result, 0)

    def test_ssim(self):
        ssim = SSIM(window_size=11, sigma=1.5)
        result = ssim.evaluate(self.original_tensor,
                               self.perturbed_tensor).item()
        self.assertAlmostEqual(result, 0.999845027923584, places=6)

    def test_ssim_full_similarity(self):
        ssim = SSIM(window_size=11, sigma=1.5)
        result = ssim.evaluate(self.original_tensor,
                               self.original_tensor).item()
        self.assertEqual(result, 1)

    def test_ergas(self):
        result = ERGAS.evaluate(self.original_tensor,
                                self.perturbed_tensor).item()
        self.assertAlmostEqual(result, 6957726.0, places=1)

    def test_ergas_full_similarity(self):
        result = ERGAS.evaluate(self.original_tensor,
                                self.original_tensor).item()
        self.assertEqual(result, 0)

    def test_rmse(self):
        result = RMSE.evaluate(self.original_tensor,
                               self.perturbed_tensor).mean().item()
        self.assertAlmostEqual(result, 0.08565672487020493, places=4)

    def test_rmse_full_similarity(self):
        result = RMSE.evaluate(self.original_tensor,
                               self.original_tensor).mean().item()
        self.assertEqual(result, 0)

    def test_sam(self):
        result = SAM.evaluate(self.original_tensor, self.perturbed_tensor).item()
        self.assertAlmostEqual(result, 7.375673770904541, places=6)

    def test_sam_full_similarity(self):
        result = SAM.evaluate(self.original_tensor, self.original_tensor).item()
        self.assertAlmostEqual(result, 0.00700760493054986, places=6)


if __name__ == "__main__":
    unittest.main()
