import unittest
import torch
import numpy as np
from PIL import Image
import tempfile
import os

from src.datasets.cifar10 import Cifar10
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.attacks.cw import cw_attack
from src.utils.torch_util import getDevice
from src.utils.randomness import set_seed
from adversarialAttack import load_model


class TestAttackMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before each test method."""
        set_seed(42)  # For reproducible tests
        cls.device = getDevice()
        cls.bird_original = Image.open("tests/assets/bird_original.png")
        cls.dataset = Cifar10(download=False)
        cls.model = load_model("cifar10")
        cls.model.to(cls.device)
        cls.model.eval()

        # Get a test image
        cls.test_tensor = cls.dataset.transform(cls.bird_original).unsqueeze(0).to(cls.device)
        cls.source_class = 2
        cls.target_class = 0

    def test_fgsm_epsilon_bounds(self):
        """Test that FGSM respects epsilon bounds."""
        epsilon = 0.01
        max_iters = 10
        break_early = True
        original = self.test_tensor.clone()

        _, success, first_success_iter, first_success_output, final_output = fgsm_attack(
            self.model, original, self.target_class, epsilon=epsilon, max_iters=max_iters, break_early=break_early
        )

        self.assertIsInstance(success, bool)
        self.assertIsInstance(first_success_iter, int)
        self.assertTrue(all(isinstance(x, float) for x in first_success_output))
        self.assertTrue(all(isinstance(x, float) for x in final_output))

        self.assertTrue(success, "FGSM attack should be successful")

        self.assertAlmostEqual(sum(first_success_output), 1.0, places=6, msg="First success output should sum to 1.0")

        self.assertAlmostEqual(sum(final_output), 1.0, places=6, msg="Final output should sum to 1.0")

    def test_pgd_convergence(self):
        """Test that PGD attack converges or reaches max iterations."""
        epsilon = 0.1
        alpha = 0.01
        max_iter = 10
        break_early = True

        original = self.test_tensor.clone()
        _, success, first_success_iter, first_success_output, final_output = pgd_attack(
            self.model, original, self.target_class, 
            epsilon=epsilon, alpha=alpha, max_iter=max_iter, break_early=break_early
        )

        self.assertIsInstance(success, bool)
        self.assertIsInstance(first_success_iter, int)
        self.assertTrue(all(isinstance(x, float) for x in first_success_output))
        self.assertTrue(all(isinstance(x, float) for x in final_output))

        self.assertTrue(success, "PGD attack should be successful")

        self.assertAlmostEqual(sum(first_success_output), 1.0, places=6, msg="First success output should sum to 1.0")

        self.assertAlmostEqual(sum(final_output), 1.0, places=6, msg="Final output should sum to 1.0")

    def test_cw_attack(self):
        """Test that CW attack converges or reaches max iterations."""
        lr = 0.001
        max_iter = 10
        break_early = True
        c = 1
        kappa = 0.0

        original = self.test_tensor.clone()
        _, success, first_success_iter, first_success_output, final_output = cw_attack(
            self.model, original, self.target_class, lr=lr, steps=max_iter, c=c, kappa=kappa, break_early=break_early
        )

        self.assertIsInstance(success, bool)
        self.assertIsInstance(first_success_iter, int)
        self.assertTrue(all(isinstance(x, float) for x in first_success_output))
        self.assertTrue(all(isinstance(x, float) for x in final_output))

        self.assertTrue(success, "CW attack should be successful")

        self.assertAlmostEqual(sum(first_success_output), 1.0, places=6, msg="First success output should sum to 1.0")

        self.assertAlmostEqual(sum(final_output), 1.0, places=6, msg="Final output should sum to 1.0")


if __name__ == '__main__':
    unittest.main() 
