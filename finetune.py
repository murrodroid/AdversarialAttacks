from src.finetuning.base import finetune
from src.finetuning.configs.base_finetune import config
from src.finetuning.dataloaders import create_imagenet100_loaders
from src.models.get_model import get_model


model_name = 'resnet'

model = get_model(model_name)
train_loader,val_loader = create_imagenet100_loaders(model_name=model_name,batch_size=config['training']['batch_size'],workers=config['training']['workers'])

def main() -> None:
    """Run finetuning for the configured model."""
    model_name = config["training"]["model_name"]

    model = get_model(model_name)
    train_loader, val_loader = create_imagenet100_loaders(
        model_name=model_name,
        batch_size=config["training"]["batch_size"],
        workers=config["training"]["workers"],
    )

    finetune(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=config,
    )


if __name__ == "__main__":
    main()