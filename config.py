import argparse
from datetime import datetime
import torch
import torchvision.models as models
from dataclasses import dataclass
from typing import Dict, List, Callable, Optional, Any
from multiprocessing import cpu_count

from src.datasets.cifar10 import Cifar10
from src.datasets.imagenet import ImageNet100
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.attacks.cw import cw_attack
from src.models.get_model import get_model, get_finetuned_model
from src.utils.torch_util import getVRAM


@dataclass
class AttackConfig:
    """Configuration for a specific attack method."""

    name: str
    epsilon: float
    alpha: Optional[float] = None
    iterations: int = 40

    def get_attack_kwargs(self) -> Dict[str, Any]:
        """Get attack-specific keyword arguments."""
        kwargs = {"epsilon": self.epsilon}

        if self.name == "fgsm":
            kwargs["max_iters"] = self.iterations
        elif self.name == "pgd":
            kwargs["alpha"] = self.alpha
            kwargs["max_iter"] = self.iterations
        elif self.name == "cw":
            # CW doesn't use epsilon, alpha, or standard iterations
            kwargs = {
                "lr": 0.01,  # Learning rate for Adam optimizer
                "steps": self.iterations,  # Number of optimization steps
                "c": 10.0,  # Balance between adversarial loss and perturbation
                "kappa": 15,  # Confidence margin
            }

        return kwargs

    def needs_unnormalized_tensors(self) -> bool:
        """Return True if this attack requires unnormalized tensors in [0,1] range."""
        return self.name == "cw"


@dataclass
class GenerationConfig:
    """Configuration for the adversarial image generation pipeline."""

    models: List[str]
    dataset: str
    attacks: List[str]
    epsilon: float
    alpha: float
    iterations: int
    num_images_per_class: int
    parallel_processes: int
    device: Optional[str]
    image_output_dir: str
    metadata_output_path: str
    seed: int
    batch_size: int
    pairing_mode: str = "all_targets"  # "all_targets" or "random_target"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.parallel_processes <= 0:
            self.parallel_processes = 1

        # Set device if not specified
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_attack_configs(self) -> List[AttackConfig]:
        """Generate AttackConfig objects for all attack/epsilon combinations."""
        attack_configs = []
        for attack_name in self.attacks:
            config = AttackConfig(name=attack_name, epsilon=self.epsilon, alpha=self.alpha, iterations=self.iterations)
            attack_configs.append(config)
        return attack_configs


class ModelRegistry:
    """Registry for available models."""

    _MODELS: Dict[str, Callable] = {
        "mobilenet": lambda: get_model("mobilenet"),
        "resnet": lambda: get_finetuned_model("resnet"),
        "swin": lambda: get_model("swin"),
        "cifar10_resnet20": lambda: torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True
        ),
    }

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available model names."""
        return list(cls._MODELS.keys())

    @classmethod
    def load_model(cls, model_name: str, device: torch.device):
        """Load and return the specified model on the given device."""
        if model_name not in cls._MODELS:
            raise ValueError(f"Model '{model_name}' not recognized. Available: {cls.get_available_models()}")

        model = cls._MODELS[model_name]()
        model = torch.compile(model,backend = "eager")
        model = model.to(device)
        model.eval()
        return model

    @classmethod
    def register_model(cls, name: str, model_factory: Callable):
        """Register a new model factory function."""
        cls._MODELS[name] = model_factory


class DatasetRegistry:
    """Registry for available datasets."""

    _DATASETS: Dict[str, Callable] = {"cifar10": Cifar10, "imagenet100": ImageNet100}

    @classmethod
    def get_available_datasets(cls) -> List[str]:
        """Get list of available dataset names."""
        return list(cls._DATASETS.keys())

    @classmethod
    def get_dataset_instance(cls, dataset_name: str):
        """Get an instance of the specified dataset class."""
        if dataset_name not in cls._DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' not recognized. Available: {cls.get_available_datasets()}")
        return cls._DATASETS[dataset_name]()

    @classmethod
    def register_dataset(cls, name: str, dataset_class: Callable):
        """Register a new dataset class."""
        cls._DATASETS[name] = dataset_class


class AttackRegistry:
    """Registry for available attack methods."""

    _ATTACKS: Dict[str, Callable] = {"fgsm": fgsm_attack, "pgd": pgd_attack, "cw": cw_attack}

    @classmethod
    def get_available_attacks(cls) -> List[str]:
        """Get list of available attack names."""
        return list(cls._ATTACKS.keys())

    @classmethod
    def get_attack_function(cls, attack_name: str) -> Callable:
        """Get the attack function for the specified attack."""
        if attack_name not in cls._ATTACKS:
            raise ValueError(f"Attack '{attack_name}' not recognized. Available: {cls.get_available_attacks()}")
        return cls._ATTACKS[attack_name]

    @classmethod
    def register_attack(cls, name: str, attack_function: Callable):
        """Register a new attack function."""
        cls._ATTACKS[name] = attack_function


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Parallel Adversarial Image Generation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    default_models = ModelRegistry.get_available_models()[1]
    parser.add_argument(
        "--model",
        type=str,
        default=[default_models],
        nargs="+",
        choices=ModelRegistry.get_available_models(),
        help="Select one or more model architectures to use.",
    )

    # Dataset configuration
    default_dataset = DatasetRegistry.get_available_datasets()[1]
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        choices=DatasetRegistry.get_available_datasets(),
        help="Select dataset to use.",
    )

    # Attack configuration
    default_attacks = AttackRegistry.get_available_attacks()
    parser.add_argument(
        "--attack",
        type=str,
        default=default_attacks[0:1],
        nargs="+",
        choices=AttackRegistry.get_available_attacks(),
        help="Select one or more attack methods.",
    )

    # Generation parameters
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of original images per source class to generate attacks for.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.03,
        help="Epsilon value for perturbation magnitude (e.g., for FGSM, PGD). Not used for CW attack.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.007,
        help="Step size alpha for iterative attacks like PGD. Typically epsilon/iterations (single value applies to all PGD runs).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=40,
        help="Number of iterations for iterative attacks (applies to FGSM, PGD iterations, and CW steps).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of images to process in each batch.",
    )
    parser.add_argument(
        "--pairing_mode",
        type=str,
        default="all_targets",
        choices=["all_targets", "random_target"],
        help="Pairing mode for adversarial samples: 'all_targets' pairs each image with all possible target classes, 'random_target' assigns each image a single random target class.",
    )

    # Execution parameters
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,  # max(1, cpu_count() // 2),
        help="Number of parallel processes to launch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Force device usage (cpu or cuda). If None, defaults to cuda if available, else cpu.",
    )

    # Output configuration
    parser.add_argument(
        "--image_dir",
        type=str,
        default="results/adversarial_images",
        help="Directory where generated adversarial images will be saved.",
    )
    parser.add_argument(
        "--metadata_output",
        type=str,
        default="results/generation_metadata.csv",
        help="Path to save the CSV file containing metadata about generated images.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility.")

    return parser


def parse_args_to_config(args: argparse.Namespace) -> GenerationConfig:
    """Convert parsed arguments to a GenerationConfig object."""
    return GenerationConfig(
        models=args.model,
        dataset=args.dataset,
        attacks=args.attack,
        epsilon=args.epsilon,
        alpha=args.alpha,
        iterations=args.iterations,
        num_images_per_class=args.num_images,
        parallel_processes=args.parallel,
        device=args.device,
        image_output_dir=args.image_dir,
        metadata_output_path=args.metadata_output,
        seed=args.seed,
        batch_size=args.batch_size,
        pairing_mode=args.pairing_mode,
    )


def validate_configuration(config: GenerationConfig) -> None:
    """Validate the configuration and raise errors for invalid settings."""
    if not config.models:
        raise ValueError(
            "No models available or specified. Please add models to ModelRegistry or use the --model argument."
        )

    if not config.attacks:
        raise ValueError(
            "No attacks available or specified. Please add attacks to AttackRegistry or use the --attack argument."
        )

    # Validate models exist
    available_models = ModelRegistry.get_available_models()
    for model in config.models:
        if model not in available_models:
            raise ValueError(f"Model '{model}' not available. Available models: {available_models}")

    # Validate dataset exists
    available_datasets = DatasetRegistry.get_available_datasets()
    if config.dataset not in available_datasets:
        raise ValueError(f"Dataset '{config.dataset}' not available. Available datasets: {available_datasets}")

    # Validate attacks exist
    available_attacks = AttackRegistry.get_available_attacks()
    for attack in config.attacks:
        if attack not in available_attacks:
            raise ValueError(f"Attack '{attack}' not available. Available attacks: {available_attacks}")

    # GPU-specific validation
    if config.device == "cuda" and config.parallel_processes > 1:
        num_gpus = torch.cuda.device_count()
        if num_gpus < config.parallel_processes:
            print(
                f"Warning: Requesting {config.parallel_processes} parallel CUDA processes "
                f"but only {num_gpus} GPU(s) detected. Processes might share GPUs, "
                f"potentially causing slowdowns or memory issues."
            )
        elif num_gpus > config.parallel_processes:
            print(f"Info: {num_gpus} GPUs detected, but only using {config.parallel_processes} parallel processes.")

def create_wandb_run(model, attack, dataset, hyper_cfg, mode="online", project="adversarialAttacks", entity=None):
    run = wandb.init(
        project=project,
        entity=entity,
        mode=mode,
        name=f"{model}_{attack}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        group=model,
        job_type="attack",
        tags=[attack, dataset],
        config=hyper_cfg,
    )
    return run


def create_wandb_config(gen_cfg: GenerationConfig):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    wandb_cfg = dict(
        project  = "adversarialAttacks",
        entity   = None,   
        mode     = "online", 
        name = f"testAttack_{run_id}",
        tags = [*gen_cfg.attacks, *gen_cfg.models, gen_cfg.dataset],
        job_type = 'attack',
    )
    return wandb_cfg 

default_epsilon = {"imagenet100":4/255, "cifar10": 8/255}


def get_config(
    dataset,
    GB_vram=getVRAM(),
    num_images=100,
    pairing_mode="all_targets",
    attacks=["fgsm", "pgd", "cw"],
) -> GenerationConfig:
    """Get a default configuration for the generation pipeline."""
    if dataset not in DatasetRegistry.get_available_datasets():
        raise ValueError(f"Dataset '{dataset}' not recognized. Available datasets: {DatasetRegistry.get_available_datasets()}")

    models = ["mobilenet", "resnet", "swin"] if dataset == "imagenet100" else ["cifar10_resnet20"]
    max_iterations = 100
    epsilon = default_epsilon[dataset]
    alpha = epsilon/max_iterations

    if GB_vram <= 4: 
        batch_size = 16
    elif 4 < GB_vram <= 8:
        batch_size = 32
    else: 
        batch_size = 256

    if dataset == "cifar10": 
        batch_size *= 8

    return GenerationConfig(
        models=models,
        dataset=dataset,
        attacks=attacks,
        epsilon=epsilon,
        alpha=alpha,
        iterations=100,
        num_images_per_class=num_images,
        parallel_processes=max(1, cpu_count() // 2),
        device=None,  # Will default to cuda if available
        image_output_dir="results/adversarial_images",
        metadata_output_path="results/generation_metadata.csv",
        seed=42,
        batch_size=batch_size,
        pairing_mode=pairing_mode,
    )
