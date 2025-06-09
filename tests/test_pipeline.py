import unittest
import torch
import os
import tempfile
from PIL import Image

from src.datasets.cifar10 import Cifar10
from src.utils.randomness import set_seed
from src.utils.torch_util import getDevice
from pipeline import run_single_generation
from config import ModelRegistry, AttackConfig


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_seed(42)
        cls.device = getDevice()
        cls.temp_dir = tempfile.mkdtemp()

        cls.bird_original = Image.open("tests/assets/bird_original.png")
        cls.ship_original = Image.open("tests/assets/ship_original.png")

        cls.bird_tensor = Cifar10.transforms(cls.bird_original).unsqueeze(0)
        cls.ship_tensor = Cifar10.transforms(cls.ship_original).unsqueeze(0)

        cls.test_batch = torch.cat([cls.bird_tensor, cls.ship_tensor], dim=0)
        # (source_class, target_class, image_index)
        cls.test_meta = [(0, 1, 0), (1, 0, 1)]

        cls.model = ModelRegistry.load_model("cifar10_resnet20", torch.device(cls.device)).eval()

    def test_single_generation(self):
        generation_config = {
            "model_name": "cifar10_resnet20",
            "dataset_name": "cifar10",
            "attack_name": "fgsm",
            "epsilon": 0.01,
            "alpha": 0.01,
            "iterations": 10,
            "seed": 42,
            "device": self.device,
            "image_output_dir": self.temp_dir,
            "batch_cpu": self.test_batch,
            "meta": self.test_meta,
            "_cached_model": self.model,
        }

        attack_config = AttackConfig(
            name="fgsm",
            epsilon=0.01,
            alpha=0.01,
            iterations=10,
        )

        results = run_single_generation(generation_config, attack_config)

        self.assertEqual(len(results), 2)

        for result in results:
            self.assertIn("model", result)
            self.assertIn("attack", result)
            self.assertIn("true_class", result)
            self.assertIn("target_class", result)
            self.assertIn("original_pred_class", result)
            self.assertIn("adversarial_pred_class", result)
            self.assertIn("first_success_iter", result)
            self.assertIn("attack_successful", result)
            self.assertIn("psnr_score", result)
            self.assertIn("ssim_score", result)
            self.assertIn("ergas_score", result)
            self.assertIn("adversarial_image_path", result)

            self.assertEqual(result["model"], "cifar10_resnet20")
            self.assertEqual(result["attack"], "fgsm")
            self.assertIsInstance(result["attack_successful"], bool)
            self.assertIsInstance(result["psnr_score"], float)
            self.assertIsInstance(result["ssim_score"], float)
            self.assertIsInstance(result["ergas_score"], float)

            if result["attack_successful"]:
                self.assertIsInstance(result["first_success_iter"], int)
            else:
                self.assertIsNone(result["first_success_iter"])

            self.assertTrue(os.path.exists(result["adversarial_image_path"]))

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.temp_dir)
        if cls.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
