from src.finetuning.base import finetune
from src.finetuning.configs.base_finetune import train_cfg,wandb_cfg
from src.datasets.imagenet import create_imagenet100_loaders
from src.models.get_model import get_model


model_name = train_cfg["model_name"]
model = get_model(model_name)
def main() -> None:
    """Run finetuning for the configured model."""
    model = get_model(model_name)
    train_loader,val_loader = create_imagenet100_loaders(
        batch_size=train_cfg["batch_size"],
        workers=train_cfg["workers"],
        )
    finetune(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=train_cfg,
        wandb_cfg=wandb_cfg
    )

if __name__ == "__main__":
    main()