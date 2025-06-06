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
        cls.ship_original = Image.open("tests/assets/ship_original.png")
        cls.frog_original = Image.open("tests/assets/frog_original.png")

        cls.model = load_model("cifar10")
        cls.model.to(cls.device)
        cls.model.eval()

        bird_tensor = Cifar10.transforms(cls.bird_original).unsqueeze(0)
        ship_tensor = Cifar10.transforms(cls.ship_original).unsqueeze(0)
        frog_tensor = Cifar10.transforms(cls.frog_original).unsqueeze(0)

        cls.test_tensor = torch.cat([bird_tensor, ship_tensor, frog_tensor], dim=0).to(
            cls.device
        )

        cls.target_classes = [0, 1, 2] 

    def test_fgsm_batch(self):
        epsilon = 0.01
        max_iters = 10
        break_early = True
        original = self.test_tensor.clone()

        _, success, first_success_iter, first_success_output, final_output = (
            fgsm_attack(
                self.model,
                original,
                self.target_classes,
                epsilon=epsilon,
                max_iters=max_iters,
                break_early=break_early,
            )
        )

        self.assertIsInstance(success, list)
        self.assertIsInstance(first_success_iter, list)
        self.assertIsInstance(first_success_output, list)
        self.assertIsInstance(final_output, list)

        self.assertEqual(len(success), 3)
        self.assertEqual(len(first_success_iter), 3)
        self.assertEqual(len(first_success_output), 3)
        self.assertEqual(len(final_output), 3)

        for i in range(3):
            self.assertIsInstance(success[i], bool)
            if success[i]:
                self.assertIsInstance(first_success_iter[i], int)
                self.assertTrue(
                    all(isinstance(x, float) for x in first_success_output[i])
                )
                self.assertAlmostEqual(
                    sum(first_success_output[i]),
                    1.0,
                    places=6,
                    msg=f"First success output for image {i} should sum to 1.0",
                )

            self.assertTrue(all(isinstance(x, float) for x in final_output[i]))
            self.assertAlmostEqual(
                sum(final_output[i]),
                1.0,
                places=6,
                msg=f"Final output for image {i} should sum to 1.0",
            )

    def test_pgd_batch(self):
        """Test that PGD attack converges or reaches max iterations."""
        epsilon = 0.1
        alpha = 0.01
        max_iter = 10
        break_early = True

        original = self.test_tensor.clone()
        _, success, first_success_iter, first_success_output, final_output = pgd_attack(
            self.model,
            original,
            self.target_classes,
            epsilon=epsilon,
            alpha=alpha,
            max_iter=max_iter,
            break_early=break_early,
        )

        self.assertIsInstance(success, list)
        self.assertIsInstance(first_success_iter, list)
        self.assertIsInstance(first_success_output, list)
        self.assertIsInstance(final_output, list)

        self.assertEqual(len(success), 3)
        self.assertEqual(len(first_success_iter), 3)
        self.assertEqual(len(first_success_output), 3)
        self.assertEqual(len(final_output), 3)

        for i in range(3):
            self.assertIsInstance(success[i], bool)
            if success[i]:
                self.assertIsInstance(first_success_iter[i], int)
                self.assertTrue(
                    all(isinstance(x, float) for x in first_success_output[i])
                )
                self.assertAlmostEqual(
                    sum(first_success_output[i]),
                    1.0,
                    places=6,
                    msg=f"First success output for image {i} should sum to 1.0",
                )

            self.assertTrue(all(isinstance(x, float) for x in final_output[i]))
            self.assertAlmostEqual(
                sum(final_output[i]),
                1.0,
                places=6,
                msg=f"Final output for image {i} should sum to 1.0",
            )

    def test_cw_attack(self):
        """Test that CW attack converges or reaches max iterations."""
        lr = 0.001
        max_iter = 10
        break_early = True
        c = 1
        kappa = 0.0

        original = self.test_tensor[0:1]
        target_class = self.target_classes[0]

        _, success, first_success_iter, first_success_output, final_output = cw_attack(
            self.model,
            original,
            target_class,
            lr=lr,
            steps=max_iter,
            c=c,
            kappa=kappa,
            break_early=break_early,
        )

        self.assertIsInstance(success, bool)
        if success and first_success_iter is not None:
            self.assertIsInstance(first_success_iter, int)
        if success and first_success_output is not None:
            self.assertTrue(all(isinstance(x, float) for x in first_success_output))

        if final_output is not None:
            self.assertTrue(all(isinstance(x, float) for x in final_output))


if __name__ == '__main__':
    unittest.main() 
