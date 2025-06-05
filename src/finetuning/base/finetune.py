import os, torch, wandb
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

def _replace_head(m, name, k):
    if name == "resnet":
        m.fc = nn.Linear(m.fc.in_features, k)
    elif name == "mobilenet":
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, k)
    elif name == "swin":
        m.head = nn.Linear(m.head.in_features, k)
    return m

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

def finetune(model, model_name, train_loader, val_loader, config):
    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    model = _replace_head(model, model_name, config["output_dim"])

    if not config["finetune_all_layers"]:
        model = _freeze_backbone(model, model_name)

    model = model.cuda()
    model = DDP(model, device_ids=[rank])

    wandb.init(project=config["project"], config=config,
               mode="online" if rank == 0 else "disabled")
    
    criterion = nn.CrossEntropyLoss().cuda()
    optim_params = filter(lambda p: p.requires_grad, model.parameters())

    opt = optim.SGD(optim_params, lr=config["lr"],
                    momentum=0.9, weight_decay=config["weight_decay"])
    
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config["epochs"])
    scaler, best = GradScaler(), 0

    for epoch in range(config["epochs"]):
        train_loader.sampler.set_epoch(epoch)
        model.train()
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
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                with autocast():
                    pred = model(x).argmax(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        acc = correct / total
        if rank == 0:
            wandb.log({"epoch": epoch, "val_acc": acc})
            if acc > best:
                best = acc
                torch.save(model.module.state_dict(),
                           os.path.join(config["save_dir"], "best.pt"))
    wandb.finish()
    dist.destroy_process_group()
