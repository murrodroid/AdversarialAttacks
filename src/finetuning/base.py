import os, torch, wandb
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from src.utils.torch_util import getDevice

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

def finetune(model, train_loader, val_loader, cfg: dict):
    train_cfg = cfg['training']
    wandb_cfg = cfg['wandb']
    model_name = train_cfg['model_name']
    rank = int(os.getenv("LOCAL_RANK", 0))
    model = _replace_head(model, model_name, train_cfg)
    device = getDevice()

    if not (torch.cuda.is_available() or getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError('cuda/mps not available, aint no way we finetunin on cpu bruh')

    distributed = int(os.getenv("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)
    
    if not train_cfg["finetune_all_layers"]: model = _freeze_backbone(model, model_name)
    
    params = filter(lambda p: p.requires_grad, model.parameters())
    opt = _build_opt(model_name,params,train_cfg)

    model = model.to(device)
    if distributed: model = DDP(model, device_ids=[rank])

    run = wandb.init(
        project = wandb_cfg["project"],
        entity  = wandb_cfg.get("entity"),
        mode    = "online" if rank == 0 else "disabled",
        config  = train_cfg,
    )

    criterion = nn.CrossEntropyLoss().to(device)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_cfg["epochs"])
    use_amp = train_cfg.get('amp',True)
    scaler = GradScaler(enabled=use_amp)

    best = 0.0
    for epoch in range(train_cfg['epochs']):
        sampler = getattr(train_loader, "sampler", None)
        if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
            sampler.set_epoch(epoch)

        model.train()

        # training
        for x, y in train_loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        model.eval()
        correct = total = 0

        # val acc
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                with autocast(enabled=use_amp):
                    pred = model(x).argmax(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        acc = correct / total

        # wandb stats & save best model
        if rank == 0:
            wandb.log({"epoch": epoch, "val_acc": acc})
            if acc > best:
                best = acc
                os.makedirs(train_cfg["save_dir"], exist_ok=True)
                state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save(state, os.path.join(train_cfg["save_dir"], "best.pt"))
    
    # end
    if rank == 0: wandb.finish()
    if distributed: dist.destroy_process_group()