import torch.nn as nn

ACTIVATION_CLS = {
    'silu': nn.SiLU,
    'swish': nn.SiLU,
    'mish': nn.Mish,
    'gelu': nn.GELU,
    'relu': nn.ReLU,
    'lrelu': nn.LeakyReLU,
}


def get_activation_cls(name: str):
    name = name.lower()
    return ACTIVATION_CLS[name]
