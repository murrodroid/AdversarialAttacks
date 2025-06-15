import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights
from src.finetuning.shifted_window_attention import ShiftedWindowAttention

def build_custom_swin_tiny_with_temp(temperature=1.5, num_classes=100):
    model = swin_t(weights=Swin_T_Weights.DEFAULT)

    for stage in model.features:
        for block in getattr(stage, 'blocks', []):
            block.attn = ShiftedWindowAttention(
                dim=block.attn.qkv.in_features,
                window_size=block.attn.window_size,
                shift_size=block.attn.shift_size,
                num_heads=block.attn.num_heads,
                attention_dropout=block.attn.attn_drop.p,
                dropout=block.attn.proj_drop.p,
                temperature=temperature,
            )

    model.head = nn.Linear(model.head.in_features, num_classes)
    return model
