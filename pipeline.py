import torch
import os
import pandas as pd
import numpy as np
import itertools
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

from src.utils.randomness import set_seed
from src.utils.torch_util import tensor_to_pil
from src.iqa import *


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
            f"[Proc {process_id} Error] Failed to load model/dataset for config {config}: {e}")
        return []

    attack_func = AttackRegistry.get_attack_function(attack_name)

    try:
        num_classes = len(dataset_instance.labels)
    except AttributeError:
        print(
            f"Warning: Dataset {dataset_name} has no 'labels' attribute. Assuming 10 classes.")
        num_classes = 10

    batch_data = []

    original_tensors_for_prediction = []

    for source_class in range(num_classes):
        try:
            samples = config["shared_images"][source_class]
        except Exception as e:
            print(
                f"[Proc {process_id} Error] Failed getting samples for class {source_class}: {e}")
            continue

        if not samples:
            continue

        for sample in samples:
            if not isinstance(sample, dict) or "index" not in sample or "tensor" not in sample:
                print(
                    f"[Proc {process_id} Warning] Skipping invalid sample format for source class {source_class}.")
                continue

            original_tensor_cpu = sample["tensor"]
            dataset_index = sample["index"]
            original_tensor = original_tensor_cpu.to(device)

            for target_class in range(num_classes):
                if target_class == source_class:
                    continue

                batch_data.append({
                    "original_tensor": original_tensor.clone().detach().squeeze(0),
                    "source_class": source_class,
                    "target_class": target_class,
                    "dataset_index": dataset_index,
                })
                original_tensors_for_prediction.append(
                    original_tensor.clone().detach().squeeze(0))

    if not batch_data:
        print(
            f"[Proc {process_id} Warning] No valid samples found for processing.")
        return []

    print(f"[Proc {process_id}] Getting original predictions in batch for {len(original_tensors_for_prediction)} images...")
    original_batch = torch.stack(original_tensors_for_prediction)

    try:
        with torch.no_grad():

            orig_logits = model(original_batch)
            orig_pred_classes = orig_logits.argmax(
                1).cpu().numpy()
    except Exception as e:
        print(
            f"[Proc {process_id} Error] Failed batch original prediction: {e}")

        orig_pred_classes = []
        for tensor in original_tensors_for_prediction:
            try:
                with torch.no_grad():
                    pred = model(tensor.unsqueeze(0)).argmax(1).item()
                    orig_pred_classes.append(pred)
            except Exception:
                orig_pred_classes.append(-1)
        orig_pred_classes = np.array(orig_pred_classes)

    batch_tensors = torch.stack([item["original_tensor"]
                                for item in batch_data])
    batch_targets = [item["target_class"] for item in batch_data]

    attack_config = AttackConfig(
        name=attack_name, epsilon=epsilon, alpha=alpha, iterations=iterations)
    attack_kwargs = attack_config.get_attack_kwargs()
    attack_kwargs["break_early"] = True

    print(f"[Proc {process_id}] Processing batch of {len(batch_data)} examples for {model_name} + {attack_name}")

    try:

        (perturbed_batch, success_list, first_success_iter_list,
         first_success_output_list, final_output_list) = attack_func(
            model, batch_tensors, batch_targets, **attack_kwargs)
        perturbed_batch = perturbed_batch.detach()

        with torch.no_grad():
            adv_logits = model(perturbed_batch)
            adv_pred_classes = adv_logits.argmax(
                1).cpu().numpy()

    except Exception as e:
        print(f"[Proc {process_id} Error] Failed during batch attack: {e}")
        return []

    metadata_results = []

    for i, (batch_item, success, first_success_iter, first_success_output, final_output) in enumerate(
        zip(batch_data, success_list, first_success_iter_list,
            first_success_output_list, final_output_list)):

        try:
            source_class = batch_item["source_class"]
            target_class = batch_item["target_class"]
            dataset_index = batch_item["dataset_index"]
            original_tensor = batch_item["original_tensor"]
            perturbed_tensor = perturbed_batch[i]

            orig_pred_class = int(orig_pred_classes[i])
            adv_pred_class = int(adv_pred_classes[i])

            original_for_psnr = (original_tensor * 255).clamp(0, 255)
            perturbed_for_psnr = (perturbed_tensor * 255).clamp(0, 255)

            try:
                psnr_score = psnr_evaluator.evaluate(
                    original_for_psnr.unsqueeze(0), perturbed_for_psnr.unsqueeze(0))
                ssim_score = sim_evaluator.evaluate(
                    original_for_psnr.unsqueeze(0), perturbed_for_psnr.unsqueeze(0))
                ergas_score = ergas_evaluator.evaluate(
                    original_for_psnr.unsqueeze(0), perturbed_for_psnr.unsqueeze(0))

                if torch.is_tensor(psnr_score):
                    psnr_score = psnr_score.item()
                if torch.is_tensor(ssim_score):
                    ssim_score = ssim_score.item()
                if torch.is_tensor(ergas_score):
                    ergas_score = ergas_score.item()

            except Exception as metric_error:
                print(
                    f"[Proc {process_id} Warning] Error calculating metrics for image {i}: {metric_error}")
                psnr_score = ssim_score = ergas_score = -1

            try:
                adv_pil = tensor_to_pil(perturbed_tensor, dataset_name)
                img_filename = (
                    f"adv_{dataset_name}_{attack_name}"
                    f"_model{model_name.replace('_', '-')}"
                    f"_src{source_class}_tgt{target_class}_idx{dataset_index}.png"
                )
                img_path = os.path.join(image_output_dir, img_filename)
                adv_pil.save(img_path)
            except Exception as save_error:
                print(
                    f"[Proc {process_id} Warning] Error saving image {i}: {save_error}")
                img_path = None

            metadata_row = {
                "model": model_name,
                "attack": attack_name,
                "first_success_iter": first_success_iter if success else None,
                "iterations": iterations,
                "true_class": source_class,
                "target_class": target_class,
                "original_pred_class": orig_pred_class,
                "adversarial_pred_class": adv_pred_class,
                "first_success_prob_distribution": (first_success_output if success else None),
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
            print(
                f"[Proc {process_id} Error] Failed processing result {i}: {e}")
            metadata_results.append({
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
            })

    del model
    if device_str == "cuda":
        torch.cuda.empty_cache()

    print(
        f"[Proc {process_id}] Completed batch processing: {len(metadata_results)} results")
    return metadata_results


def run_pipeline(config: GenerationConfig):
    """
    Optimized version of run_pipeline with model caching and reduced redundancy.
    """
    print(f"Pipeline using device type: {config.device}")

    if config.device == "cuda" and config.parallel_processes > 1:
        num_gpus = torch.cuda.device_count()
        if num_gpus < config.parallel_processes:
            print(
                f"Warning: Requesting {config.parallel_processes} parallel CUDA processes but only {num_gpus} GPU(s) detected.")

    dataset_instance = DatasetRegistry.get_dataset_instance(config.datasets[0])
    num_classes = len(dataset_instance.labels)

    shared_images = {}
    print("Loading shared images...")
    for cls in range(num_classes):
        indices = dataset_instance.get_indices_from_class(
            cls, train=False, num_images=config.num_images_per_class)
        shared_images[cls] = []
        for idx in indices:
            sample = dataset_instance.get_by_index(idx, train=False)
            if isinstance(sample, dict) and "tensor" in sample:
                shared_images[cls].append(
                    {"tensor": sample["tensor"], "index": sample["index"]})

    param_combinations = list(itertools.product(
        config.models, config.datasets, config.attacks, config.epsilons))

    configs_by_model = {}
    for i, (model, dataset, attack, eps) in enumerate(param_combinations):
        if model not in configs_by_model:
            configs_by_model[model] = []

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
            "shared_images": shared_images,
        }
        configs_by_model[model].append(exp_config)

    total_examples_per_config = config.num_images_per_class * \
        num_classes * (num_classes - 1)
    total_examples = len(param_combinations) * total_examples_per_config
    print(
        f"Will generate {total_examples} adversarial examples across {len(param_combinations)} configurations")

    os.makedirs(config.image_output_dir, exist_ok=True)

    all_metadata = []

    for model_name, model_configs in configs_by_model.items():
        print(
            f"\nProcessing {len(model_configs)} configurations for model: {model_name}")

        try:
            model = ModelRegistry.load_model(
                model_name, torch.device(config.device))
            dataset_instance = DatasetRegistry.get_dataset_instance(
                config.datasets[0])

            for cfg in model_configs:
                cfg['_cached_model'] = model
                cfg['_cached_dataset'] = dataset_instance

        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue

        for cfg in tqdm(model_configs, desc=f"Processing {model_name}"):
            metadata = run_single_generation(cfg)
            if metadata:
                all_metadata.extend(metadata)

        del model
        if config.device == "cuda":
            torch.cuda.empty_cache()

    if not all_metadata:
        print("Warning: No metadata was generated.")
        return

    print(
        f"\nCollected metadata for {len(all_metadata)} generated adversarial examples.")
    print(
        f"Successfully processed {len([m for m in all_metadata if m.get('attack_successful')])} successful attacks.")

    metadata_df = pd.DataFrame(all_metadata)

    if config.metadata_output_path:
        output_dir = os.path.dirname(config.metadata_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        try:
            metadata_df.to_csv(config.metadata_output_path, index=False)
            print(
                f"Metadata saved successfully to: {config.metadata_output_path}")
        except Exception as e:
            print(
                f"Error saving metadata CSV to {config.metadata_output_path}: {e}")
            print("Displaying metadata instead:")
            print(metadata_df.head())
    else:
        print("\n--- Combined Metadata (Head) ---")
        print(metadata_df.head())

    print(
        f"\nAdversarial images saved in directory: {config.image_output_dir}")
    print("Pipeline finished.")


if __name__ == "__main__":
    start_time = time.time()

    parser = create_argument_parser()
    args = parser.parse_args()

    config = parse_args_to_config(args)

    validate_configuration(config)

    run_pipeline(config)

    end_time = time.time()
    total_time = end_time - start_time
    print(
        f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
