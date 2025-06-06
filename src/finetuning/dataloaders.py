from ..datasets.imagenet import ImageNet100
from .configs.base_finetune import config

from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torchvision.models import (
    ResNet50_Weights, MobileNet_V3_Large_Weights, Swin_T_Weights)

train_cfg = config['training']

_BACKBONE_WEIGHTS = dict(
    resnet    = ResNet50_Weights.DEFAULT,
    mobilenet = MobileNet_V3_Large_Weights.DEFAULT,
    swin      = Swin_T_Weights.DEFAULT,
)

def create_imagenet100_loaders(model_name: str,
                               dataset_dir: str = train_cfg['dataset_root'],
                               batch_size: int = 32,
                               workers: int = 8):

    ws   = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank()        if dist.is_initialized() else 0

    weights     = _BACKBONE_WEIGHTS[model_name]
    tfm_train   = weights.transforms(training=True)
    tfm_val     = weights.transforms(training=False)

    ds_train = ImageNet100(dataset_dir, split="train")
    ds_val   = ImageNet100(dataset_dir, split="validation")

    ds_train.transform, ds_val.transform = tfm_train, tfm_val

    smp_train = DistributedSampler(ds_train, ws, rank, shuffle=True) if ws > 1 else None
    smp_val   = DistributedSampler(ds_val,   ws, rank, shuffle=False) if ws > 1 else None

    per_gpu_bs = batch_size // ws

    loader_kwargs = dict(batch_size=per_gpu_bs,
                         num_workers=workers,
                         pin_memory=True,
                         drop_last=True)

    train_loader = DataLoader(ds_train, sampler=smp_train,
                              shuffle=(smp_train is None), **loader_kwargs)
    val_loader   = DataLoader(ds_val,   sampler=smp_val,
                              shuffle=False,               **loader_kwargs)

    return train_loader, val_loader
