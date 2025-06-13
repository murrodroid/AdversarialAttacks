from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import itertools
import torch
import time
import os
import wandb

from src.datasets.dataset_base import AdvDataset
from src.utils.randomness import set_seed
from src.utils.torch_util import tensor_to_pil, unnormalize_tensor, normalize_tensor
from src.iqa import ERGAS, PSNR, SSIM


from config import (
    ModelRegistry,
    DatasetRegistry,
    AttackRegistry,
    AttackConfig,
    GenerationConfig,
    create_argument_parser,
    get_config,
    parse_args_to_config,
    validate_configuration,
    create_wandb_config,
)

ssim_evaluator = SSIM()


def run_single_generation(generation_config, attack_config):
    set_seed(generation_config["seed"] + generation_config.get("process_id", 0))

    requested_device = generation_config["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    model = generation_config["_cached_model"]
    attack_function = AttackRegistry.get_attack_function(generation_config["attack_name"])
    attack_parameters = attack_config.get_attack_kwargs()
    attack_parameters["break_early"] = True

    input_batch = generation_config["batch_cpu"]
    if device.type == "cuda":
        input_batch = input_batch.to(device, non_blocking=True)
    else:
        input_batch = input_batch.to(device)

    if attack_config.needs_unnormalized_tensors():
        attack_input_batch = unnormalize_tensor(input_batch, generation_config["dataset_name"])
    else:
        attack_input_batch = input_batch

    with torch.no_grad():
        original_outputs = model(input_batch)
        original_predictions = original_outputs.argmax(1).cpu().numpy()
        original_probs = torch.softmax(original_outputs, dim=1).cpu().numpy()

    (
        perturbed_images,
        attack_success,
        first_success_iteration,
        first_output,
        final_output,
    ) = attack_function(
        model,
        attack_input_batch,
        [target for (_, target, _) in generation_config["meta"]],
        **attack_parameters,
    )
    perturbed_images = perturbed_images.detach()

    if attack_config.needs_unnormalized_tensors():
        perturbed_normalized = normalize_tensor(perturbed_images, generation_config["dataset_name"])
    else:
        perturbed_normalized = perturbed_images

    with torch.no_grad():
        adversarial_outputs = model(perturbed_normalized)
        adversarial_predictions = adversarial_outputs.argmax(1).cpu().numpy()
        adversarial_probs = torch.softmax(adversarial_outputs, dim=1).cpu().numpy()

    psnr_scores = PSNR.evaluate(input_batch, perturbed_normalized).cpu().numpy()
    ssim_scores = ssim_evaluator.evaluate(input_batch, perturbed_normalized).cpu().numpy()
    ergas_scores = ERGAS.evaluate(input_batch, perturbed_normalized).cpu().numpy()

    if attack_config.needs_unnormalized_tensors():
        images_to_save = normalize_tensor(perturbed_images, generation_config["dataset_name"])
    else:
        images_to_save = perturbed_images

    output_paths = [
        f"{generation_config['image_output_dir']}/adv_{generation_config['attack_name']}_src{source}_tgt{target}_idx{index}.png"
        for source, target, index in generation_config["meta"]
    ]
    if generation_config['should_save_images']:
        with ThreadPoolExecutor() as executor:
            for i, (_, _, _) in enumerate(generation_config["meta"]):
                adversarial_image = images_to_save[i]
                output_path = output_paths[i]
                executor.submit(
                    lambda img=adversarial_image, path=output_path: tensor_to_pil(
                        img, generation_config["dataset_name"]
                    ).save(path)
                )

    generation_results = []
    for i, (source_class, target_class, image_index) in enumerate(generation_config["meta"]):
        generation_results.append(
            {
                "model": generation_config["model_name"],
                "attack": generation_config["attack_name"],
                "true_class": source_class,
                "target_class": target_class,
                "original_pred_class": int(original_predictions[i]),
                "adversarial_pred_class": int(adversarial_predictions[i]),
                "first_success_iter": (first_success_iteration[i] if attack_success[i] else None),
                "attack_successful": int(adversarial_predictions[i]) == target_class,
                "psnr_score": float(psnr_scores[i]),
                "ssim_score": float(ssim_scores[i]),
                "ergas_score": float(ergas_scores[i]),
                "adversarial_image_path": output_paths[i],
                "original_probs": original_probs[i].tolist(),
                "adversarial_probs": adversarial_probs[i].tolist(),
                "dataset_index": image_index,
            }
        )
    return generation_results


def preprocess_batches(
    dataset,
    num_classes,
    num_images_per_class,
    batch_size,
    pairing_mode="all_targets",
    workers=8,
):
    """Create batches directly from cached tensors - simplified and fast"""
    ds = AdvDataset(dataset, num_classes, num_images_per_class, pairing_mode)

    print("Creating batches from cached tensors...")
    batches = []

    for i in range(0, len(ds.samples), batch_size):
        batch_samples = ds.samples[i : i + batch_size]

        batch_tensors = [ds.cached_tensors[idx] for _, _, idx in batch_samples]
        batch_tensor = torch.stack(batch_tensors)

        batches.append({"batch_cpu": batch_tensor, "meta": batch_samples})

    print(f"Created {len(batches)} batches with batch size {batch_size}")
    return batches


def run_pipeline(config: GenerationConfig):
    os.makedirs(config.image_output_dir, exist_ok=True)

    wandb_cfg = create_wandb_config(config)
    run = wandb.init(
        project=wandb_cfg['project'],
        name=wandb_cfg['name'],
        entity=wandb_cfg['entity'],
        mode=wandb_cfg['mode'],
        job_type=wandb_cfg['job_type'],
        tags=wandb_cfg['tags'],
    )
    step = 0

    dataset = DatasetRegistry.get_dataset_instance(config.dataset)
    num_classes = len(dataset.labels)
    all_results = []

    preprocessing_start = time.time()
    batches = preprocess_batches(
        dataset,
        num_classes,
        config.num_images_per_class,
        config.batch_size,
        config.pairing_mode,
    )
    preprocessing_time = time.time() - preprocessing_start
    print(f"\nPreprocessing completed in {preprocessing_time:.2f} seconds")

    generation_start = time.time()
    for model_name, attack_name in itertools.product(config.models, config.attacks):
        model = ModelRegistry.load_model(model_name, torch.device(config.device)).eval()
        attack_config = AttackConfig(
            attack_name,
            config.epsilon,
            config.alpha,
            config.iterations,
        )

        for batch in tqdm(batches, desc=f"Generating attacks for {model_name}/{attack_name}"):
            if config.device == "cuda":
                batch["batch_cpu"] = batch["batch_cpu"].to(config.device, non_blocking=True)
            else:
                batch["batch_cpu"] = batch["batch_cpu"].to(config.device)

            generation_config = {
                "model_name": model_name,
                "dataset_name": config.dataset,
                "attack_name": attack_name,
                "epsilon": config.epsilon,
                "alpha": config.alpha,
                "iterations": config.iterations,
                "seed": config.seed,
                "device": config.device,
                "image_output_dir": config.image_output_dir,
                "process_id": 0,
                "_cached_model": model,
                "should_save_images": config.should_save_images,
                **batch,
            }

            batch_results = run_single_generation(generation_config, attack_config)
            all_results.extend(batch_results)

            wandb_df = pd.DataFrame(batch_results)
            run.log(
            {
            "model": model,
            "attack": attack_name,
            "success_rate": wandb_df.attack_successful.mean(),
            "mean_psnr": wandb_df.psnr_score.mean(),
            "mean_ssim": wandb_df.ssim_score.mean(),
            "mean_ergas": wandb_df.ergas_score.mean(),
            "mean_first_iter": wandb_df.first_success_iter.dropna().mean(),
            },
            step=step,
            )

            if config.device == "cuda":
                torch.cuda.empty_cache()

        del model
        if config.device == "cuda":
            torch.cuda.empty_cache()

    generation_time = time.time() - generation_start
    print(f"Generated {len(all_results)} attacks in {generation_time:.2f} seconds")

    results_df = pd.DataFrame(all_results)
    if config.metadata_output_path:
        os.makedirs(os.path.dirname(config.metadata_output_path), exist_ok=True)
        results_df.to_csv(config.metadata_output_path, index=False)
    else:
        print(results_df.head())


if __name__ == "__main__":
    start_time = time.time()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    config = get_config("imagenet20", num_images=100, pairing_mode="random_target", should_save_images=False)

    validate_configuration(config)

    run_pipeline(config)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
