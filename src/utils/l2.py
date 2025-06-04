import torch as torch


def get_activation(model,layer_name):
    activation_dict = {}
    def hook_fn(module, input, output):
        activation_dict[layer_name] = output.detach()
    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(hook_fn)
    return activation_dict

def L2(source_image,pertrubed_image):
    l2_distance = torch.norm(source_image - pertrubed_image, p=2)
    return l2_distance

