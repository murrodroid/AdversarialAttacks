import os, torch, wandb
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from src.utils.torch_util import getDevice
from src.finetuning.configs.base_finetune import train_cfg, wandb_cfg

def _replace_head(model, name: str, cfg: dict):
    if name == "resnet":
        model.fc = nn.Linear(model.fc.in_features, cfg['output_dim'])

    elif name == "mobilenet":
        model.classifier[3] = nn.Linear(model.classifier[3].in_features,
                                        cfg['output_dim'])

    elif name == "swin":
        model.head = nn.Linear(model.head.in_features, cfg['output_dim'])
    
    return model

def _build_opt(name, params, cfg):
    if name == "resnet":
        return optim.SGD(params, lr=cfg['learning_rate'],
                         momentum=0.9, weight_decay=cfg['weight_decay'])
    if name == "mobilenet":
        return optim.RMSprop(params, lr=cfg['learning_rate'], alpha=0.9,
                             momentum=0.9, eps=0.0316, weight_decay=cfg['weight_decay'])
    if name == "swin":
        return optim.AdamW(params, lr=cfg['learning_rate'], betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=cfg['weight_decay'])
    raise ValueError(f"unknown model {name}")

def _freeze_backbone(model, name):
    for p in model.parameters():
        p.requires_grad = False
    if name == "resnet":
        for p in model.fc.parameters(): p.requires_grad = True
    elif name == "mobilenet":
        for p in model.classifier[3].parameters(): p.requires_grad = True
    elif name == "swin":
        for p in model.head.parameters(): p.requires_grad = True
    return model

def finetune(model, train_loader, val_loader, train_cfg: dict, wandb_cfg: dict):
    rank         = int(os.getenv("LOCAL_RANK", 0))
    distributed  = int(os.getenv("WORLD_SIZE", 1)) > 1
    device       = getDevice()
    model        = _replace_head(model, train_cfg["model_name"], train_cfg).to(device)
    if not train_cfg["finetune_all_layers"]:
        model = _freeze_backbone(model, train_cfg["model_name"])
    if distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)
        model = DDP(model, device_ids=[rank])
    opt     = _build_opt(train_cfg["model_name"], (p for p in model.parameters() if p.requires_grad), train_cfg)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_cfg["epochs"])
    scaler  = GradScaler(enabled=train_cfg.get("amp", True))
    loss_fn = nn.CrossEntropyLoss().to(device)

    run = wandb.init(
        project = wandb_cfg["project"],
        entity  = wandb_cfg.get("entity"),
        name    = wandb_cfg["run_name"],
        mode    = wandb_cfg.get("mode", "online") if rank == 0 else "disabled",
        config  = {**train_cfg, **wandb_cfg},
    )

    wandb.watch(model,log='all' if rank == 0 else None, log_freq=100)

    best = 0.0
    for epoch in range(train_cfg["epochs"]):
        sampler = getattr(train_loader, "sampler", None)
        if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
            sampler.set_epoch(epoch)

        model.train()
        tr_loss = tr_correct = tr_total = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=train_cfg.get("amp", True)):
                preds = model(x)
                loss  = loss_fn(preds, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_loss    += loss.item() * y.size(0)
            tr_correct += preds.argmax(1).eq(y).sum().item()
            tr_total   += y.size(0)

        # val acc
        sched.step()
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with autocast(enabled=train_cfg.get("amp", True)):
                    preds = model(x)
                    loss  = loss_fn(preds, y)
                val_loss   += loss.item() * y.size(0)
                val_correct += preds.argmax(1).eq(y).sum().item()
                val_total   += y.size(0)

        # wandb stats & save best model
        if rank == 0:
            epoch_stats = {
                "epoch"      : epoch,
                "lr"         : sched.get_last_lr()[0],
                "train_loss" : tr_loss / tr_total,
                "train_acc"  : tr_correct / tr_total,
                "val_loss"   : val_loss / val_total,
                "val_acc"    : val_correct / val_total,
            }
            wandb.log(epoch_stats)
            if epoch_stats["val_acc"] > best:
                best = epoch_stats["val_acc"]
                train_cfg["save_dir"].mkdir(parents=True, exist_ok=True)
                torch.save(model.module.state_dict() if distributed else model.state_dict(),
                           train_cfg["save_dir"] / "best.pt")
    # end
    if rank == 0: wandb.finish()
    if distributed: dist.destroy_process_group()