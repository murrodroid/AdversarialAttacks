import os, torch, wandb
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

def _replace_head_define_opt(model, name: str, cfg: dict):
    if name == "resnet":
        model.fc = nn.Linear(model.fc.in_features, cfg['output_dim'])
        opt = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['lr'], momentum=0.9, weight_decay=cfg['weight_decay']
        )

    elif name == "mobilenet":
        model.classifier[3] = nn.Linear(model.classifier[3].in_features,
                                        cfg['output_dim'])
        opt = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['lr'], alpha=0.9, momentum=0.9,
            eps=0.0316, weight_decay=cfg['weight_decay']
        )

    elif name == "swin":
        model.head = nn.Linear(model.head.in_features, cfg['output_dim'])
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['lr'], betas=(0.9, 0.999),
            weight_decay=cfg['weight_decay'], eps=1e-8
        )
    
    return model, opt

def _freeze_backbone(m, name):
    for p in m.parameters():
        p.requires_grad = False
    if name == "resnet":
        for p in m.fc.parameters(): p.requires_grad = True
    elif name == "mobilenet":
        for p in m.classifier[3].parameters(): p.requires_grad = True
    elif name == "swin":
        for p in m.head.parameters(): p.requires_grad = True
    return m

def finetune(model, model_name, train_loader, val_loader, cfg):
    params = cfg['params']
    wandbc = cfg['wandb']
    rank = int(os.environ["LOCAL_RANK"])
    
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)

    model, opt = _replace_head_define_opt(model, model_name, params["output_dim"])

    if not params["finetune_all_layers"]:
        model = _freeze_backbone(model, model_name)

    model = model.cuda() if torch.cuda.is_available() else AssertionError('cuda not available, aint no way we finetunin on cpu bruh')
    model = DDP(model, device_ids=[rank])

    wandb.init(project=wandbc["project"], config=wandbc,
               mode="online" if rank == 0 else "disabled")
    
    criterion = nn.CrossEntropyLoss().cuda()
    
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=params["epochs"])
    scaler = GradScaler()
    best = 0.0

    for epoch in range(params['epochs']):
        train_loader.sampler.set_epoch(epoch)
        model.train()

        # training
        for x, y in train_loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast():
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
                with autocast():
                    pred = model(x).argmax(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        acc = correct / total

        # stats & save best model
        if rank == 0:
            wandb.log({"epoch": epoch, "val_acc": acc})
            if acc > best:
                best = acc
                torch.save(model.module.state_dict(),
                           os.path.join(params["save_dir"], "best.pt"))
    wandb.finish()
    dist.destroy_process_group()
