from pathlib import Path
from datetime import datetime

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

models = ['mobilenet','resnet','swin']

model = models[2] # input


train_cfg = dict(
    model_name        = model,
    output_dim        = 100,
    finetune_all_layers = False,

    epochs            = 5,
    batch_size        = 64,
    learning_rate     = 0.001,
    weight_decay      = 1e-4,
    lr_scheduler      = "cosine",

    workers           = 8,
    amp               = True,
    save_dir          = Path("checkpoints") / f"{model}100-{run_id}",

    dataset_root        = Path(f"/data/{model}100"),
)
wandb_cfg = dict(
    project  = "adversarialAttacks",
    entity   = None,   
    mode     = "online", 
    run_name = f"{model}100_{run_id}",
)

runs_root   = Path("finetune_results/base_finetune")     # top-level folder
# run_dir     = runs_root / run_name          
# ckpt_dir    = run_dir / "checkpoints"
# reports_dir = run_dir / "reports"
# run_dir.mkdir(parents=True, exist_ok=True)
# ckpt_dir.mkdir(exist_ok=True)
# reports_dir.mkdir(exist_ok=True)