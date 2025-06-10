from pathlib import Path
from src.utils.torch_util import save_compressed


mobilenet = 'checkpoints/mobilenet100-20250609-232628/mobilenet.pt'
resnet = 'checkpoints/resnet100-20250609-225658/resnet.pt'
swin = 'checkpoints/swin100-20250609-235538/swin.pt'

model_dirs = [mobilenet,resnet,swin]

for dir in model_dirs: save_compressed(Path(dir))