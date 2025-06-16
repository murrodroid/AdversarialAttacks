import torch.nn as nn
from src.finetuning.shifted_window_attention import ShiftedWindowAttention

def robust_swin(finetuned_model,temperature=1.5, num_classes=100):
    for stage in finetuned_model.features:
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
    finetuned_model.head = nn.Linear(finetuned_model.head.in_features, num_classes)
    return finetuned_model
