import numpy as np
import torch
import os
import pandas as pd
import itertools
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF

from src.datasets.cifar10 import Cifar10
from src.utils.randomness import set_seed
from src.iqa import *

# Import configuration components
from config import (
    ModelRegistry,
    DatasetRegistry,
    AttackRegistry,
    GenerationConfig,
    AttackConfig,
    create_argument_parser,
    parse_args_to_config,
    validate_configuration,
)


def tensor_to_pil(tensor, dataset_name):
    """
    Converts a tensor (potentially normalized) back to a PIL Image.
    Assumes tensor is [B, C, H, W] or [C, H, W].
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.detach().cpu()

    try:
        if dataset_name == "cifar10":
            tensor = Cifar10.inverse_transforms(tensor)
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
    Generates adversarial images for a single model-attack configuration using batch processing.
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
    process_id = config.get("process_id", 0)

    current_seed = base_seed + process_id + hash(str(config)) % 10000
    set_seed(current_seed)
    device = torch.device(device_str)

    psnr_evaluator = PSNR()
    sim_evaluator = SSIM()
    ergas_evaluator = ERGAS()

    os.makedirs(image_output_dir, exist_ok=True)

    try:
        dataset_instance = DatasetRegistry.get_dataset_instance(dataset_name)
        model = ModelRegistry.load_model(model_name, device)
    except Exception as e:
        print(
            f"[Proc {process_id} Error] Failed to load model/dataset for config {config}: {e}"
        )
        return []

    attack_func = AttackRegistry.get_attack_function(attack_name)

    try:
        num_classes = len(dataset_instance.labels)
    except AttributeError:
        print(
            f"Warning: Dataset {dataset_name} has no 'labels' attribute. Assuming 10 classes."
        )
        num_classes = 10

    # Collect all samples and their metadata for batch processing
    batch_data = []

    for source_class in range(num_classes):
        try:
            sample_indices = config["shared_indices"][source_class]
            original_samples = [dataset_instance.get_by_index(idx, train=False) for idx in sample_indices]
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

            # Add all target classes for this sample
            for target_class in range(num_classes):
                if target_class == source_class:
                    continue

                batch_data.append(
                    {
                        "original_tensor": original_tensor.clone().detach().squeeze(0),
                        "source_class": source_class,
                        "target_class": target_class,
                        "dataset_index": dataset_index,
                        "orig_pred_class": orig_pred_class,
                    }
                )

    if not batch_data:
        print(f"[Proc {process_id} Warning] No valid samples found for processing.")
        return []

    # Create batch tensors and target lists
    batch_tensors = torch.stack([item["original_tensor"] for item in batch_data])
    batch_targets = [item["target_class"] for item in batch_data]

    # Set up attack parameters using AttackConfig
    attack_config = AttackConfig(
        name=attack_name, epsilon=epsilon, alpha=alpha, iterations=iterations
    )
    attack_kwargs = attack_config.get_attack_kwargs()
    attack_kwargs["break_early"] = True

    print(
        f"[Proc {process_id}] Processing batch of {len(batch_data)} examples for {model_name} + {attack_name}"
    )

    try:
        # Run batch attack
        (
            perturbed_batch,
            success_list,
            first_success_iter_list,
            first_success_output_list,
            final_output_list,
        ) = attack_func(
            model,
            batch_tensors,
            batch_targets,
            **attack_kwargs,
        )
        perturbed_batch = perturbed_batch.detach()

        # Get adversarial predictions for the entire batch
        with torch.no_grad():
            adv_pred_classes = model(perturbed_batch).argmax(1).cpu().numpy()

    except Exception as e:
        print(f"[Proc {process_id} Error] Failed during batch attack: {e}")
        return []

    metadata_results = []

    # Process results for each image in the batch
    for i, (
        batch_item,
        success,
        first_success_iter,
        first_success_output,
        final_output,
    ) in enumerate(
        zip(
            batch_data,
            success_list,
            first_success_iter_list,
            first_success_output_list,
            final_output_list,
        )
    ):
        try:
            source_class = batch_item["source_class"]
            target_class = batch_item["target_class"]
            dataset_index = batch_item["dataset_index"]
            orig_pred_class = batch_item["orig_pred_class"]
            original_tensor = batch_item["original_tensor"]
            perturbed_tensor = perturbed_batch[i]
            adv_pred_class = adv_pred_classes[i]

            # Calculate quality metrics
            original_for_psnr = (original_tensor * 255).clamp(0, 255)
            perturbed_for_psnr = (perturbed_tensor * 255).clamp(0, 255)
            psnr_score = psnr_evaluator.evaluate(
                original_for_psnr.unsqueeze(0), perturbed_for_psnr.unsqueeze(0)
            )
            ssim_score = sim_evaluator.evaluate(
                original_for_psnr.unsqueeze(0), perturbed_for_psnr.unsqueeze(0)
            )
            ergas_score = ergas_evaluator.evaluate(
                original_for_psnr.unsqueeze(0), perturbed_for_psnr.unsqueeze(0)
            )

            # Save adversarial image
            adv_pil = tensor_to_pil(perturbed_tensor, dataset_name)
            img_filename = (
                f"adv_{dataset_name}_{attack_name}"
                f"_model{model_name.replace('_','-')}"
                f"_src{source_class}_tgt{target_class}_idx{dataset_index}.png"
            )
            img_path = os.path.join(image_output_dir, img_filename)
            adv_pil.save(img_path)

            # Create metadata row
            metadata_row = {
                "model": model_name,
                "attack": attack_name,
                "first_success_iter": first_success_iter if success else None,
                "iterations": iterations,
                "true_class": source_class,
                "target_class": target_class,
                "original_pred_class": orig_pred_class,
                "adversarial_pred_class": int(adv_pred_class),
                "first_success_prob_distribution": (
                    first_success_output if success else None
                ),
                "final_prob_distribution": final_output,
                "dataset_index": dataset_index,
                "attack_successful": success,
                "psnr_score": psnr_score,
                "ssim_score": ssim_score,
                "ergas_score": ergas_score,
                "adversarial_image_path": img_path,
            }
            metadata_results.append(metadata_row)

        except Exception as e:
            print(f"[Proc {process_id} Error] Failed processing result {i}: {e}")
            metadata_results.append(
                {
                    "model": model_name,
                    "attack": attack_name,
                    "iterations": iterations,
                    "true_class": batch_item.get("source_class", -1),
                    "target_class": batch_item.get("target_class", -1),
                    "dataset_index": batch_item.get("dataset_index", -1),
                    "original_pred_class": -1,
                    "adversarial_pred_class": -1,
                    "attack_successful": None,
                    "adversarial_image_path": None,
                    "error": str(e),
                }
            )

    del model
    if device_str == "cuda":
        torch.cuda.empty_cache()

    print(
        f"[Proc {process_id}] Completed batch processing: {len(metadata_results)} results"
    )
    return metadata_results


def run_pipeline(config: GenerationConfig):
    """Sets up and runs the parallel image generation process using GenerationConfig."""

    print(f"Pipeline using device type: {config.device}")

    if config.device == "cuda" and config.parallel_processes > 1:
        num_gpus = torch.cuda.device_count()
        if num_gpus < config.parallel_processes:
            print(
                f"Warning: Requesting {config.parallel_processes} parallel CUDA processes but only {num_gpus} GPU(s) detected. Processes might share GPUs, potentially causing slowdowns or memory issues."
            )
        elif num_gpus > config.parallel_processes:
            print(
                f"Info: {num_gpus} GPUs detected, but only using {config.parallel_processes} parallel processes."
            )

    dataset_instance = DatasetRegistry.get_dataset_instance(config.datasets[0])
    shared_samples = {}
    for cls in range(10):
        indices = dataset_instance.get_indices_from_class(
            cls, train=False, num_images=config.num_images_per_class
        )
        shared_samples[cls] = indices

    # Create configurations for each model-attack-epsilon combination
    param_combinations = list(
        itertools.product(
            config.models,
            config.datasets,
            config.attacks,
            config.epsilons,
        )
    )

    # Calculate total number of adversarial examples that will be generated
    total_examples_per_config = (
        config.num_images_per_class * 10 * 9
    )  # num_images * num_classes * (num_classes - 1 targets)
    total_examples = len(param_combinations) * total_examples_per_config
    print(
        f"Will generate {total_examples} adversarial examples across {len(param_combinations)} model-attack configurations"
    )
    print(
        f"Each configuration processes {total_examples_per_config} examples in a single batch"
    )

    experiment_configs = []
    for i, (model, dataset, attack, eps) in enumerate(param_combinations):
        exp_config = {
            "model_name": model,
            "dataset_name": dataset,
            "attack_name": attack,
            "epsilon": eps,
            "alpha": config.alpha,
            "iterations": config.iterations,
            "num_images_per_class": config.num_images_per_class,
            "seed": config.seed,
            "device": config.device,
            "image_output_dir": config.image_output_dir,
            "process_id": i,
            "shared_indices": shared_samples,
        }
        experiment_configs.append(exp_config)

    num_configs = len(experiment_configs)
    print(
        f"Generated {num_configs} batch processing configurations (grouped by model + attack)."
    )
    if not experiment_configs:
        print("No configurations to run. Exiting.")
        return

    os.makedirs(config.image_output_dir, exist_ok=True)
    print(f"Ensured base image output directory exists: {config.image_output_dir}")

    num_workers = min(config.parallel_processes, num_configs, cpu_count())
    print(f"Starting batch generation using {num_workers} parallel processes...")
    print(
        f"Each process will handle one model-attack configuration and process all examples for that configuration in a single batch."
    )

    all_metadata = []
    with Pool(processes=num_workers) as pool:
        results_iterator = pool.imap(run_single_generation, experiment_configs)
        for metadata_list in tqdm(
            results_iterator, total=num_configs, desc="Processing Batches"
        ):
            if metadata_list:
                all_metadata.extend(metadata_list)

    if not all_metadata:
        print("Warning: No metadata was generated across all processes.")
        return

    print(
        f"\nCollected metadata for {len(all_metadata)} generated adversarial examples."
    )
    print(
        f"Successfully processed {len([m for m in all_metadata if m.get('attack_successful')])} successful attacks."
    )
    metadata_df = pd.DataFrame(all_metadata)

    if config.metadata_output_path:
        output_dir = os.path.dirname(config.metadata_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        try:
            metadata_df.to_csv(config.metadata_output_path, index=False)
            print(f"Metadata saved successfully to: {config.metadata_output_path}")
        except Exception as e:
            print(f"Error saving metadata CSV to {config.metadata_output_path}: {e}")
            print("Displaying metadata instead:")
            print(metadata_df.head())
    else:
        print("\n--- Combined Metadata (Head) ---")
        print(metadata_df.head())

    print(f"\nAdversarial images saved in directory: {config.image_output_dir}")
    print("Pipeline finished.")


if __name__ == "__main__":
    # Use the argument parser from config module
    parser = create_argument_parser()
    args = parser.parse_args()

    # Convert arguments to GenerationConfig
    config = parse_args_to_config(args)

    # Validate the configuration
    validate_configuration(config)

    # Run the pipeline with the validated configuration
    run_pipeline(config)
