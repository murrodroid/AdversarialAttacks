import torch
from torchvision.models import mobilenet_v3_large
from src.finetuning.base import _replace_head

def robust_mobilenet(width_mult=1.5, stem_kernel_size=7):
    # increase the number of channels throughout the model by 1.25
    # increase the size of kernel of the stem from initial 16 channel to 7x7
    model = mobilenet_v3_large(weights=None, width_mult=width_mult)
    print("model imported")

    model = _replace_head(model, "mobilenet", {"output_dim": 20})
    print("head replaced with output_dim=20")

    old_stem_conv = model.features[0][0]
    print("old stem conv:", old_stem_conv)
    
    new_padding = (stem_kernel_size - 1) // 2
    print("new padding:", new_padding)

    model.features[0][0] = torch.nn.Conv2d(
        in_channels=old_stem_conv.in_channels,
        out_channels=old_stem_conv.out_channels,
        kernel_size=(stem_kernel_size, stem_kernel_size),
        stride=old_stem_conv.stride,
        padding=(new_padding, new_padding),
        bias=old_stem_conv.bias
    )
    
    print("new stem conv:", model.features[0][0])
    return model
    