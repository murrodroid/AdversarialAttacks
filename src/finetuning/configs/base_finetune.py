from pathlib import Path
from datetime import datetime
import argparse

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

models = ['mobilenet','resnet','swin']


# ----- input -----
model = models[1] 
output_dim = 20
# ----- ----- -----

cfg = dict(
    output_dim        = output_dim,

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
    run_name=f"{model}20_{run_id}",
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

    parser.add_argument("--adv",        action="store_true")
    parser.add_argument("--all_layers", action="store_true")
    parser.add_argument("--hpc",        action="store_true")
    parser.add_argument("--model",      type=str, choices=models,required=True)
    parser.add_argument("--epochs",     type=int)

    args = parser.parse_args()

    cfg_out = default_config.copy()
    if args.adv is not None:
        cfg_out["adversarial_training"] = args.adv
    if args.all_layers is not None:
        cfg_out["finetune_all_layers"] = args.all_layers
    if args.hpc is not None:
        cfg_out["using_hpc"] = args.hpc
    if args.model:
        cfg_out["model_name"] = args.model
        if args.hpc:
            cfg_out["save_dir"] = Path("/zhome/0e/9/205681/AdversarialAttacks") / f"{args.model}{cfg_out['output_dim']}-{run_id}"
        else:
            cfg_out["save_dir"] = Path("./checkpoints") / f"{args.model}{cfg_out['output_dim']}-{run_id}"
    if args.epochs:
        cfg_out["epochs"] = args.epochs

    return cfg_out