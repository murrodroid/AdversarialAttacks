from pathlib import Path
from datetime import datetime

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

models = ['mobilenet','resnet','swin']


# ----- input -----
model = models[0] 
output_dim = 20
# ----- ----- -----

train_cfg = dict(
    using_hpc = False,

    model_name        = model,
    output_dim        = output_dim,
    finetune_all_layers = False,

    epochs            = 8,
    batch_size        = 256,
    learning_rate     = 0.001,
    weight_decay      = 1e-4,
    lr_scheduler      = "cosine",

    workers           = 8,
    amp               = True,
    save_dir          = Path("checkpoints") / f"{model}{output_dim}-{run_id}",

    dataset_root        = Path(f"/data/{model}{output_dim}"),
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