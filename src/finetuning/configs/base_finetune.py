from pathlib import Path
from datetime import datetime

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

#eksempel til resnet50 finetuning til imagenet100
config = dict(
    training = dict(
        model_name        = "resnet",
        output_dim        = 100,
        finetune_all_layers = False,

        epochs            = 128,
        batch_size        = 256,
        learning_rate     = 0.001,
        weight_decay      = 1e-4,
        lr_scheduler      = "cosine",

        workers           = 8,
        amp               = True,
        save_dir          = Path("checkpoints") / f"resnet50-{run_id}",

        train_root        = Path("/datasets/imagenet/train"),
        val_root          = Path("/datasets/imagenet/val"),
    ),
    wandb = dict(
        project = "adversarialAttacks",
        entity  = None,            
        mode    = "online", 
    )
)

run_name  = f"{config['model_name']}_{run_id}"
runs_root   = Path("finetuning/base_finetune")     # top-level folder
run_dir     = runs_root / run_name          
ckpt_dir    = run_dir / "checkpoints"
reports_dir = run_dir / "reports"
run_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(exist_ok=True)
reports_dir.mkdir(exist_ok=True)