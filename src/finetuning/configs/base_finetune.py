from pathlib import Path
from datetime import datetime
import argparse

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

models = ['mobilenet','resnet','swin']
output_dim = 20

cfg = dict(
    model = 'resnet', # default (hopefully never used by default...)
    output_dim        = 20,

    epochs            = 5,
    batch_size        = 32,
    learning_rate     = 0.001,
    weight_decay      = 1e-4,
    lr_scheduler      = "cosine",
    width_mult        = 1.5,  

    workers           = 8,
    amp               = True,
    save_dir          = Path("/zhome/0e/9/205681/AdversarialAttacks/") / f"{model}{output_dim}-{run_id}",

    dataset_root        = Path(f"/zhome/0e/9/205681/AdversarialAttacks/data/{model}{output_dim}"),
    dataset_name        = "imagenet20",
    adversarial_training = False,  
)
wandb_cfg = dict(
    project="adversarialAttacks",
    entity=None,
    mode="online",
)

runs_root   = Path("/zhome/0e/9/205681/AdversarialAttacks/finetune_results/base_finetune")     # top-level folder
# run_dir     = runs_root / run_name
# ckpt_dir    = run_dir / "checkpoints"
# reports_dir = run_dir / "reports"
# run_dir.mkdir(parents=True, exist_ok=True)
# ckpt_dir.mkdir(exist_ok=True)
# reports_dir.mkdir(exist_ok=True)

def create_argument_parser_cfg(default_config: dict = cfg):
    parser = argparse.ArgumentParser(
        description="Finetuning parser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--adv", action="store_true")
    parser.add_argument("--all_layers", action="store_true")
    parser.add_argument("--hpc", action="store_true")
    parser.add_argument("--model", type=str, choices=models, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--robust_weights", type=str, default=None)
    args = parser.parse_args()

    cfg_out = default_config.copy()

    # basic flags -----------------------------------------------------------
    cfg_out["adversarial_training"] = args.adv
    cfg_out["finetune_all_layers"]  = args.all_layers
    cfg_out["using_hpc"]            = args.hpc
    if args.hpc:
        cfg_out["batch_size"] = 1024

    # model-specific paths --------------------------------------------------
    cfg_out["model_name"] = args.model
    cfg_out["dataset_root"] = Path(
        f"/zhome/0e/9/205681/AdversarialAttacks/data/{args.model}{cfg_out['output_dim']}"
    )
    root_dir = (
        Path("/zhome/0e/9/205681/AdversarialAttacks")
        if args.hpc
        else Path("./checkpoints")
    )
    cfg_out["save_dir"] = root_dir / f"{args.model}{cfg_out['output_dim']}-{run_id}"

    # epochs / optional weights --------------------------------------------
    if args.epochs:
        cfg_out["epochs"] = args.epochs
    if args.robust_weights:
        cfg_out["robust_weights"] = args.robust_weights

    # keep wandb_cfg consistent --------------------------------------------
    wandb_cfg["run_name"] = f"{args.model}{cfg_out['output_dim']}_{run_id}"

    return cfg_out