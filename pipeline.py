import numpy as np
import torch
import argparse
import os
import pandas as pd
import itertools
import random
import torchvision.models as models
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF

from src.datasets.cifar10 import Cifar10
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack

from src.utils.torch_util import getDevice
from src.utils.randomness import set_seed

# --- Configuration ---
AVAILABLE_DATASETS = {"cifar10": Cifar10}

AVAILABLE_MODELS = {
    "resnet18_imagenet": lambda: models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    ),
}

AVAILABLE_ATTACKS = {"fgsm": fgsm_attack, "pgd": pgd_attack}


def load_model(model_name, device):
    """Loads the specified model onto the given device."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model '{model_name}' not recognized. Available: {list(AVAILABLE_MODELS.keys())}"
        )
    model = AVAILABLE_MODELS[model_name]()
    model = model.to(device)
    model.eval()
    return model


def get_dataset_instance(dataset_name):
    """Gets an instance of the specified dataset class."""
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' not recognized. Available: {list(AVAILABLE_DATASETS.keys())}"
        )
    return AVAILABLE_DATASETS[dataset_name]()


def tensor_to_pil(tensor, dataset_instance):
    """
    Converts a tensor (potentially normalized) back to a PIL Image.
    Assumes tensor is [B, C, H, W] or [C, H, W].
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.detach().cpu()

    try:
        tensor = dataset_instance.inverse_transform(tensor)
    except AttributeError:
        pass
    except Exception as e:
        print(f"Warning: Error during inverse_transform: {e}. Clamping tensor.")

    tensor = torch.clamp(tensor, 0, 1)

    try:
        pil_image = TF.to_pil_image(tensor)
    except Exception as e:
        print(f"Error converting tensor to PIL image: {e}")
        pil_image = Image.new("RGB", (tensor.shape[2], tensor.shape[1]), color="black")

    return pil_image


def run_single_generation(config):
    """
    Generates adversarial images for a single configuration and saves them.
    Returns metadata about the generated images. Designed for multiprocessing.
    """
    model_name = config["model_name"]
    dataset_name = config["dataset_name"]
    attack_name = config["attack_name"]
    epsilon = config["epsilon"]
    alpha = config["alpha"]
    iterations = config["iterations"]
    num_images_per_class = config["num_images_per_class"]
    base_seed = config["seed"]
    device_str = config["device"]
    image_output_dir = config["image_output_dir"]
    process_id = config.get("process_id", 0)  # Get process ID for seeding

    current_seed = base_seed + process_id + hash(str(config)) % 10000
    set_seed(current_seed)
    device = torch.device(device_str)

    os.makedirs(image_output_dir, exist_ok=True)

    try:
        dataset_instance = get_dataset_instance(dataset_name)
        model = load_model(model_name, device)
    except Exception as e:
        print(
            f"[Proc {process_id} Error] Failed to load model/dataset for config {config}: {e}"
        )
        return []

    attack_func = AVAILABLE_ATTACKS.get(attack_name)
    if not attack_func:
        print(f"[Proc {process_id} Error] Attack '{attack_name}' not recognized.")
        return []

    try:
        num_classes = len(dataset_instance.labels)
    except AttributeError:
        print(
            f"Warning: Dataset {dataset_name} has no 'labels' attribute. Assuming 10 classes."
        )
        num_classes = 10

    source_classes = list(range(num_classes))
    target_classes = list(range(num_classes))
    metadata_results = []

    for source_class in source_classes:
        try:
            original_samples = dataset_instance.get_sample_from_class(
                source_class, train=False, num_images=num_images_per_class
            )
        except Exception as e:
            print(
                f"[Proc {process_id} Error] Failed getting samples for class {source_class}: {e}"
            )
            continue

        if not original_samples:
            continue

        for sample in original_samples:
            if (
                not isinstance(sample, dict)
                or "index" not in sample
                or "tensor" not in sample
            ):
                print(
                    f"[Proc {process_id} Warning] Skipping invalid sample format for source class {source_class}."
                )
                continue

            original_tensor_cpu = sample["tensor"]
            dataset_index = sample["index"]
            original_tensor = original_tensor_cpu.to(device)

            try:
                with torch.no_grad():
                    orig_pred_class = model(original_tensor).argmax(1).item()
            except Exception as e:
                print(
                    f"[Proc {process_id} Warning] Failed original prediction for idx {dataset_index}: {e}"
                )
                orig_pred_class = -1

            for target_class in target_classes:
                if target_class == source_class:
                    continue

                attack_kwargs = {"epsilon": epsilon}
                if attack_name == "fgsm":
                    attack_kwargs["max_iters"] = iterations
                elif attack_name == "pgd":
                    attack_kwargs["alpha"] = alpha
                    attack_kwargs["max_iter"] = iterations

                try:
                    adv_tensor = attack_func(
                        model,
                        original_tensor.clone().detach(),
                        target_class,
                        **attack_kwargs,
                    ).detach()

                    with torch.no_grad():
                        adv_pred_class = model(adv_tensor).argmax(1).item()

                    adv_pil = tensor_to_pil(adv_tensor, dataset_instance)

                    img_filename = (
                        f"adv_{dataset_name}_{attack_name}"
                        f"_model{model_name.replace('_','-')}"
                        f"_eps{epsilon:.4f}_iter{iterations}"
                        f"_src{source_class}_tgt{target_class}_idx{dataset_index}.png"
                    )
                    img_path = os.path.join(image_output_dir, img_filename)

                    adv_pil.save(img_path)

                    metadata_row = {
                        "model": model_name,
                        "dataset": dataset_name,
                        "attack": attack_name,
                        "epsilon": epsilon,
                        "alpha": alpha if attack_name == "pgd" else None,
                        "iterations": iterations,
                        "source_class": source_class,
                        "target_class": target_class,
                        "dataset_index": dataset_index,
                        "original_pred_class": orig_pred_class,
                        "adversarial_pred_class": adv_pred_class,
                        "adversarial_image_path": img_path,
                    }
                    metadata_results.append(metadata_row)

                except Exception as e:
                    print(
                        f"[Proc {process_id} Error] Failed during attack/saving for config {config}, idx {dataset_index}, target {target_class}: {e}"
                    )
                    metadata_results.append(
                        {
                            "model": model_name,
                            "dataset": dataset_name,
                            "attack": attack_name,
                            "epsilon": epsilon,
                            "alpha": alpha,
                            "iterations": iterations,
                            "source_class": source_class,
                            "target_class": target_class,
                            "dataset_index": dataset_index,
                            "adversarial_image_path": None,
                            "error": str(e),
                        }
                    )

    del model
    if device_str == "cuda":
        torch.cuda.empty_cache()

    return metadata_results


def run_pipeline(args):
    """Sets up and runs the parallel image generation process."""

    if args.device:
        device_str = args.device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Pipeline using device type: {device_str}")

    if device_str == "cuda" and args.parallel > 1:
        num_gpus = torch.cuda.device_count()
        if num_gpus < args.parallel:
            print(
                f"Warning: Requesting {args.parallel} parallel CUDA processes but only {num_gpus} GPU(s) detected. Processes might share GPUs, potentially causing slowdowns or memory issues."
            )
        elif num_gpus > args.parallel:
            print(
                f"Info: {num_gpus} GPUs detected, but only using {args.parallel} parallel processes."
            )

    param_combinations = list(
        itertools.product(
            args.model,
            args.dataset,
            args.attack,
            args.epsilon,
            # args.alpha,
            # args.iterations,
            # args.num_images,
            # args.seed,
            # args.device,
            # args.image_dir,
            # args.metadata_output,
        )
    )

    print(f"Generating {len(param_combinations)} images")

    experiment_configs = []
    for i, (model, dataset, attack, eps) in enumerate(param_combinations):
        config = {
            "model_name": model,
            "dataset_name": dataset,
            "attack_name": attack,
            "epsilon": eps,
            "alpha": args.alpha,
            "iterations": args.iterations,
            "num_images_per_class": args.num_images,
            "seed": args.seed,
            "device": device_str,
            "image_output_dir": args.image_dir,
            "process_id": i,
        }
        experiment_configs.append(config)

    num_configs = len(experiment_configs)
    print(f"Generated {num_configs} generation configurations.")
    if not experiment_configs:
        print("No configurations to run. Exiting.")
        return

    os.makedirs(args.image_dir, exist_ok=True)
    print(f"Ensured base image output directory exists: {args.image_dir}")

    num_workers = min(args.parallel, num_configs, cpu_count())
    print(f"Starting generation using {num_workers} parallel processes...")

    all_metadata = []
    with Pool(processes=num_workers) as pool:
        results_iterator = pool.imap(run_single_generation, experiment_configs)
        for metadata_list in tqdm(
            results_iterator, total=num_configs, desc="Generating Images"
        ):
            if metadata_list:
                all_metadata.extend(metadata_list)

    if not all_metadata:
        print("Warning: No metadata was generated across all processes.")
        return

    print(
        f"\nCollected metadata for {len(all_metadata)} generated or attempted images."
    )
    metadata_df = pd.DataFrame(all_metadata)

    if args.metadata_output:
        output_dir = os.path.dirname(args.metadata_output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        try:
            metadata_df.to_csv(args.metadata_output, index=False)
            print(f"Metadata saved successfully to: {args.metadata_output}")
        except Exception as e:
            print(f"Error saving metadata CSV to {args.metadata_output}: {e}")
            print("Displaying metadata instead:")
            print(metadata_df.head())
    else:
        print("\n--- Combined Metadata (Head) ---")
        print(metadata_df.head())

    print(f"\nAdversarial images saved in directory: {args.image_dir}")
    print("Pipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel Adversarial Image Generation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Configuration Arguments ---
    default_models = list(AVAILABLE_MODELS.keys())[:1]
    parser.add_argument(
        "--model",
        type=str,
        default=default_models,
        nargs="+",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Select one or more model architectures to use.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=["cifar10"],
        nargs="+",
        choices=list(AVAILABLE_DATASETS.keys()),
        help="Select one or more datasets to use.",
    )
    default_attacks = list(AVAILABLE_ATTACKS.keys())[:2]
    parser.add_argument(
        "--attack",
        type=str,
        default=default_attacks,
        nargs="+",
        choices=list(AVAILABLE_ATTACKS.keys()),
        help="Select one or more attack methods.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of original images per source class to generate attacks for.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=[0.03],
        nargs="+",
        help="Epsilon value(s) for perturbation magnitude (e.g., for FGSM, PGD).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Step size alpha for iterative attacks like PGD (single value applies to all PGD runs).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=40,
        help="Number of iterations for iterative attacks (single value applies to all relevant attacks).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=max(1, cpu_count() // 2),
        help="Number of parallel processes to launch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Force device usage (cpu or cuda). If None, defaults to cuda if available, else cpu.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="./generated_adversarial_images",
        help="Directory where generated adversarial images will be saved.",
    )
    parser.add_argument(
        "--metadata_output",
        type=str,
        default="results/generation_metadata.csv",
        help="Path to save the CSV file containing metadata about generated images.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Base random seed for reproducibility."
    )

    args = parser.parse_args()

    if args.parallel <= 0:
        print("Warning: --parallel must be positive. Setting to 1.")
        args.parallel = 1

    if not args.model:
        raise ValueError(
            "No models available or specified. Please add models to AVAILABLE_MODELS or use the --model argument."
        )
    if not args.attack:
        raise ValueError(
            "No attacks available or specified. Please add attacks to AVAILABLE_ATTACKS or use the --attack argument."
        )

    run_pipeline(args)
