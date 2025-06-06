import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]   # …/src
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finetuning.base import finetune
from finetuning.configs.base_finetune import *
from finetuning.dataloaders import create_imagenet100_loaders
from models.get_model import get_model


model_name = 'resnet'

model = get_model(model_name)
train_loader,val_loader = create_imagenet100_loaders(model_name=model_name,batch_size=config['training']['batch_size'],workers=config['training']['workers'])

finetune(model=model,model_name=model_name,train_loader=train_loader,val_loader=val_loader,cfg=config)