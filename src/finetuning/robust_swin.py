import torch.nn as nn
from src.finetuning.shifted_window_attention import ShiftedWindowAttention
from torchvision.models import swin_t

def robust_swin(temperature=1.5, num_classes=20):
    model = swin_t(weights=None)
    
    # More robust way to access all Swin blocks
    def replace_attention_in_stage(stage):
        if hasattr(stage, '__iter__') and not isinstance(stage, nn.Module):
            # If it's a sequence/list, iterate through it
            for substage in stage:
                replace_attention_in_stage(substage)
        elif hasattr(stage, 'children'):
            # If it's a module with children, check for blocks
            for child in stage.children():
                replace_attention_in_stage(child)
        elif hasattr(stage, 'attn'):
            # This is likely a SwinTransformerBlock
            stage.attn = ShiftedWindowAttention(
                dim=stage.attn.qkv.in_features,
                window_size=stage.attn.window_size,
                shift_size=stage.attn.shift_size,
                num_heads=stage.attn.num_heads,
                attention_dropout=stage.attn.attn_drop.p,
                dropout=stage.attn.proj_drop.p,
                temperature=temperature,
            )
    
    # Apply attention replacement to all stages
    replace_attention_in_stage(model.features)
    
    # Only replace head if num_classes is different from default
    if num_classes != 1000:
        model.head = nn.Linear(model.head.in_features, num_classes)
    
    return model
