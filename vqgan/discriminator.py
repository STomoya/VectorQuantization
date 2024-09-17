import torch
import torch.nn as nn

from vq.activation import get_activation_cls
from vq.config import ConfigMixin


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        activation: str = 'lrelu',
        groups: int = 32,
        residual_scale_init: float = 0.001,
    ):
        super().__init__()
        act_kwargs = {'negative_slope': 0.2} if activation == 'lrelu' else {}

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.activation = get_activation_cls(activation)(**act_kwargs)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, kernel_size // 2)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

        self.gamma = nn.Parameter(torch.ones(1) * residual_scale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x)

        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)

        x = x * self.gamma + skip
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 32, activation: str = 'lrelu'):
        super().__init__()
        act_kwargs = {'negative_slope': 0.2} if activation == 'lrelu' else {}

        self.norm = nn.GroupNorm(groups, in_channels)
        self.activation = get_activation_cls(activation)(**act_kwargs)
        self.down = nn.Conv2d(in_channels, out_channels, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.activation(x)
        x = self.down(x)
        return x


class Discriminator(nn.Module, ConfigMixin):
    """Discriminator."""

    def __init__(
        self,
        in_channels: int,
        block_out_channels: tuple[int, ...] = (64, 256, 256, 512),  # type: ignore
        layers_per_block: int = 2,
        norm_groups: int = 32,
        activation: str = 'swish',
        residual_scale_init: float = 1e-3,
    ):
        super().__init__()

        self.input = nn.Conv2d(in_channels, block_out_channels[0], 7, 1, 3)

        channels = [block_out_channels[0], *block_out_channels]
        in_channels = channels[:-1]
        out_channels = channels[1:]

        self.blocks = nn.Sequential()
        for i, (ich, och) in enumerate(zip(in_channels, out_channels)):  # noqa: B905
            if i != 0:
                self.blocks.append(Downsample(ich, och, norm_groups, activation))
                ich = och  # noqa: PLW2901
            for _ in range(layers_per_block):
                self.blocks.append(
                    ResBlock(
                        ich,
                        och,
                        3,
                        activation,
                        norm_groups,
                        residual_scale_init,
                    )
                )
                ich = och  # noqa: PLW2901

        self.output = nn.Sequential(
            nn.GroupNorm(norm_groups, och),
            get_activation_cls(activation)(**({'negative_slope': 0.2} if activation == 'lrelu' else {})),
            nn.Conv2d(och, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.blocks(x)
        x = self.output(x)
        return x
