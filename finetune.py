import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from torch.amp import autocast
from src.finetuning.base import finetune
from src.finetuning.configs.base_finetune import train_cfg, wandb_cfg
from src.datasets.imagenet import create_imagenet20_loaders
from src.models.get_model import get_finetuned_model
from src.utils.torch_util import getDevice


model_name = train_cfg["model_name"]


def evaluate_test_set(model, test_loader, train_cfg):
    """Evaluate the model on the test set and return accuracy."""
    device = getDevice()
    model.eval()

    test_correct = test_total = 0
    test_loss = 0.0
    loss_fn = nn.CrossEntropyLoss().to(device)

    print("Evaluating on test set...")
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=train_cfg.get("amp", True)):
                preds = model(x)
                loss = loss_fn(preds, y)

            test_loss += loss.item() * y.size(0)
            test_correct += preds.argmax(1).eq(y).sum().item()
            test_total += y.size(0)

    test_accuracy = test_correct / test_total
    test_loss = test_loss / test_total

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    return test_accuracy, test_loss


def save_test_results(test_accuracy, test_loss, train_cfg, wandb_cfg):
    """Save test results to a JSON file."""
    # Create results directory
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)

    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Prepare results data
    results = {
        "timestamp": timestamp,
        "model_name": train_cfg["model_name"],
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "training_config": {
            "epochs": train_cfg["epochs"],
            "batch_size": train_cfg["batch_size"],
            "learning_rate": train_cfg["learning_rate"],
            "weight_decay": train_cfg["weight_decay"],
            "finetune_all_layers": train_cfg["finetune_all_layers"],
            "output_dim": train_cfg["output_dim"],
        },
        "wandb_run_name": wandb_cfg["run_name"],
    }

    # Save to JSON file
    results_file = (
        results_dir / f"test_results_{train_cfg['model_name']}_{timestamp}.json"
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Test results saved to: {results_file}")
    return results_file


def main() -> None:
    """Run finetuning for the configured model and evaluate on test set."""
    model = get_finetuned_model(model_name, cfg={"output_dim": 20})

    # Create train, validation, and test loaders
    train_loader, val_loader, test_loader = create_imagenet20_loaders(
        batch_size=train_cfg["batch_size"],
        workers=train_cfg["workers"],
        train_cfg=train_cfg,
        include_test=True,  # Include test loader
        random_seed=42,  # Ensure reproducible splits
    )

    print("Starting finetuning")
    finetune(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=train_cfg,
        wandb_cfg=wandb_cfg
    )

    # Load the best model checkpoint for test evaluation
    print("\nLoading best model for test evaluation...")
    checkpoint_path = train_cfg["save_dir"] / f"{train_cfg['model_name']}.pt"

    if checkpoint_path.exists():
        # Load the saved state dict
        state_dict = torch.load(checkpoint_path, map_location=getDevice())
        model.load_state_dict(state_dict)
        print(f"Loaded best model from: {checkpoint_path}")
    else:
        print("Warning: No checkpoint found, using current model state")

    # Evaluate on test set
    test_accuracy, test_loss = evaluate_test_set(model, test_loader, train_cfg)

    # Save test results
    results_file = save_test_results(test_accuracy, test_loss, train_cfg, wandb_cfg)

    print(f"\nFinal Test Results:")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()
