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


def run_single_generation(cfg):
    set_seed(cfg["seed"] + cfg.get("process_id", 0))
    device = torch.device(cfg["device"])
    model = cfg["_cached_model"]
    attack_fn = AttackRegistry.get_attack_function(cfg["attack_name"])
    kw = AttackConfig(cfg["attack_name"], cfg["epsilon"],
                      cfg["alpha"], cfg["iterations"]).get_attack_kwargs()
    kw["break_early"] = True

    batch_cpu = cfg["batch_cpu"].pin_memory()
    batch = batch_cpu.to(device, non_blocking=True)

    with torch.no_grad():
        orig_preds = model(batch).argmax(1).cpu().numpy()

    perturbed, success, first_it, first_out, final_out = attack_fn(
        model, batch, [t for (_, t, _) in cfg["meta"]], **kw
    )
    perturbed = perturbed.detach()

    with torch.no_grad():
        adv_preds = model(perturbed).argmax(1).cpu().numpy()

    ssim_eval = SSIM()

    psnr = PSNR.evaluate(batch, perturbed).cpu().numpy()
    ssim = ssim_eval.evaluate(batch, perturbed).cpu().numpy()
    ergas = ERGAS.evaluate(batch, perturbed).cpu().numpy()

    paths = [
        f"{cfg['image_output_dir']}/adv_{cfg['attack_name']}_src{src}_tgt{tgt}_idx{idx}.png"
        for src, tgt, idx in cfg["meta"]
    ]

    with ThreadPoolExecutor() as ex:
        for i, (_, _, _) in enumerate(cfg["meta"]):
            img = perturbed[i]
            path = paths[i]
            ex.submit(lambda img=img, path=path: tensor_to_pil(
                img, cfg["dataset_name"]).save(path))

    results = []
    for i, (src, tgt, idx) in enumerate(cfg["meta"]):
        results.append({
            "model": cfg["model_name"],
            "attack": cfg["attack_name"],
            "true_class": src,
            "target_class": tgt,
            "original_pred_class": int(orig_preds[i]),
            "adversarial_pred_class": int(adv_preds[i]),
            "first_success_iter": first_it[i] if success[i] else None,
            "attack_successful": bool(success[i]),
            "psnr_score": float(psnr[i]),
            "ssim_score": float(ssim[i]),
            "ergas_score": float(ergas[i]),
            "adversarial_image_path": paths[i],
        })
    return results


def run_pipeline(config: GenerationConfig):
    os.makedirs(config.image_output_dir, exist_ok=True)

    caches = {}
    for ds_name in config.datasets:
        ds = DatasetRegistry.get_dataset_instance(ds_name)
        K = len(ds.labels)
        imgs = []
        meta = []
        for c in range(K):
            for idx in ds.get_indices_from_class(c, train=False, num_images=config.num_images_per_class):
                sample = ds.get_by_index(idx, train=False)
                img = sample["tensor"].squeeze(0)
                for t in range(K):
                    if t != c:
                        imgs.append(img)
                        meta.append((c, t, idx))
        caches[ds_name] = {
            "batch_cpu": torch.stack(imgs),
            "meta": meta
        }

    groups = {}
    for i, (m, d, a, e) in enumerate(itertools.product(
        config.models, config.datasets, config.attacks, config.epsilons
    )):
        groups.setdefault((m, d), []).append({
            "model_name": m,
            "dataset_name": d,
            "attack_name": a,
            "epsilon": e,
            "alpha": config.alpha,
            "iterations": config.iterations,
            "seed": config.seed,
            "device": config.device,
            "image_output_dir": config.image_output_dir,
            "process_id": i,
            **caches[d]
        })

    all_metadata = []
    for (model_name, ds_name), cfgs in groups.items():
        model = ModelRegistry.load_model(
            model_name, torch.device(config.device)).eval()
        for c in cfgs:
            c["_cached_model"] = model
        for c in tqdm(cfgs, desc=f"{model_name}/{ds_name}"):
            all_metadata.extend(run_single_generation(c))
        del model
        if config.device == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_metadata)
    if config.metadata_output_path:
        os.makedirs(os.path.dirname(
            config.metadata_output_path), exist_ok=True)
        df.to_csv(config.metadata_output_path, index=False)
    else:
        print(df.head())


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
