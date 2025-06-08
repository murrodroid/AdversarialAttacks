from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import itertools
import torch
import time
import os

from src.utils.randomness import set_seed
from src.utils.torch_util import tensor_to_pil
from src.iqa import ERGAS, PSNR, SSIM

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


def run_single_generation(generation_config):
    set_seed(generation_config["seed"] +
             generation_config.get("process_id", 0))
    device = torch.device(generation_config["device"])
    model = generation_config["_cached_model"]
    attack_function = AttackRegistry.get_attack_function(
        generation_config["attack_name"])
    attack_parameters = AttackConfig(
        generation_config["attack_name"],
        generation_config["epsilon"],
        generation_config["alpha"],
        generation_config["iterations"],
    ).get_attack_kwargs()
    attack_parameters["break_early"] = True

    input_batch_cpu = generation_config["batch_cpu"].pin_memory()
    input_batch = input_batch_cpu.to(device, non_blocking=True)

    with torch.no_grad():
        original_predictions = model(input_batch).argmax(1).cpu().numpy()

    perturbed_images, attack_success, first_success_iteration, first_output, final_output = attack_function(
        model, input_batch, [target for (_, target, _) in generation_config["meta"]], **attack_parameters
    )
    perturbed_images = perturbed_images.detach()

    with torch.no_grad():
        adversarial_predictions = model(
            perturbed_images).argmax(1).cpu().numpy()

    ssim_evaluator = SSIM()

    psnr_scores = PSNR.evaluate(input_batch, perturbed_images).cpu().numpy()
    ssim_scores = ssim_evaluator.evaluate(
        input_batch, perturbed_images).cpu().numpy()
    ergas_scores = ERGAS.evaluate(input_batch, perturbed_images).cpu().numpy()

    output_paths = [
        f"{generation_config['image_output_dir']}/adv_{generation_config['attack_name']}_src{source}_tgt{target}_idx{index}.png"
        for source, target, index in generation_config["meta"]
    ]

    with ThreadPoolExecutor() as executor:
        for i, (_, _, _) in enumerate(generation_config["meta"]):
            adversarial_image = perturbed_images[i]
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
                "first_success_iter": first_success_iteration[i] if attack_success[i] else None,
                "attack_successful": bool(attack_success[i]),
                "psnr_score": float(psnr_scores[i]),
                "ssim_score": float(ssim_scores[i]),
                "ergas_score": float(ergas_scores[i]),
                "adversarial_image_path": output_paths[i],
            }
        )
    return generation_results


def run_pipeline(config: GenerationConfig):
    os.makedirs(config.image_output_dir, exist_ok=True)

    dataset_caches = {}
    for dataset_name in config.datasets:
        dataset = DatasetRegistry.get_dataset_instance(dataset_name)
        num_classes = len(dataset.labels)
        images = []
        metadata = []
        for class_idx in range(num_classes):
            for image_idx in dataset.get_indices_from_class(
                class_idx, train=False, num_images=config.num_images_per_class
            ):
                sample = dataset.get_by_index(image_idx, train=False)
                image = sample["tensor"].squeeze(0)
                for target_class in range(num_classes):
                    if target_class != class_idx:
                        images.append(image)
                        metadata.append((class_idx, target_class, image_idx))
        dataset_caches[dataset_name] = {
            "batch_cpu": torch.stack(images), "meta": metadata}

    experiment_groups = {}
    for process_id, (model_name, dataset_name, attack_name, epsilon) in enumerate(
        itertools.product(config.models, config.datasets,
                          config.attacks, config.epsilons)
    ):
        experiment_groups.setdefault((model_name, dataset_name), []).append(
            {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "attack_name": attack_name,
                "epsilon": epsilon,
                "alpha": config.alpha,
                "iterations": config.iterations,
                "seed": config.seed,
                "device": config.device,
                "image_output_dir": config.image_output_dir,
                "process_id": process_id,
                **dataset_caches[dataset_name],
            }
        )

    all_results = []
    for (model_name, dataset_name), experiment_configs in experiment_groups.items():
        model = ModelRegistry.load_model(
            model_name, torch.device(config.device)).eval()
        for exp_config in experiment_configs:
            exp_config["_cached_model"] = model
        for exp_config in tqdm(experiment_configs, desc=f"{model_name}/{dataset_name}"):
            all_results.extend(run_single_generation(exp_config))
        del model
        if config.device == "cuda":
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results)
    if config.metadata_output_path:
        os.makedirs(os.path.dirname(
            config.metadata_output_path), exist_ok=True)
        results_df.to_csv(config.metadata_output_path, index=False)
    else:
        print(results_df.head())


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
