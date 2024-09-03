import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F

from vq.activation import get_activation_cls
from vq.vq import VectorQuantizer


class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()

        self.eps = eps

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = x * self.weight
        else:
            x = x.to(input_dtype)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dims: int,
        heads: int = 8,
        head_dim: int = 64,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        out_bias: bool = True,
        residual_scale_init: float | None = 1.0,
        use_rms_norm: bool = False,
        eps: float = 1e-6,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = heads
        self.head_dim = head_dim
        self.scale = self.head_dim**-0.5
        self.inner_dims = heads * head_dim

        self.fused_attn = hasattr(F, 'scaled_dot_product_attention')
        NormCls = RMSNorm if use_rms_norm else nn.LayerNorm

        self.input_norm = NormCls(dims, eps=eps)
        self.to_qkv = nn.Linear(dims, self.inner_dims * 3, bias=qkv_bias)
        self.q_norm = NormCls(self.inner_dims, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = NormCls(self.inner_dims, eps=eps) if qk_norm else nn.Identity()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.to_out = nn.Linear(self.inner_dims, dims, bias=out_bias)
        self.out_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.gamma = nn.Parameter(torch.ones(1) * residual_scale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x

        B, C, height, width = x.shape
        N = height * width
        x = x.view(B, C, N).transpose(1, 2)

        x = self.input_norm(x)

        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.inner_dims)
        x = self.to_out(x)
        x = self.out_dropout(x)

        x = x.transpose(1, 2).view(B, C, height, width).contiguous()

        x = skip + self.gamma * x
        return x


class ResNetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = 'swish',
        residual_scale_init: float | None = 1.0,
        groups: int = 32,
        eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.act = get_activation_cls(activation)()

        self.gamma = nn.Parameter(torch.ones(1) * residual_scale_init)

        self.skip_conv = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip_conv(x)

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)

        x = skip + x * self.gamma

        return x


class Downsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = True,
        out_channels: int | None = None,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        stride = 2
        out_channels = out_channels or channels

        if use_conv:
            self.conv = nn.Conv2d(channels, out_channels, kernel_size, stride, kernel_size // 2, bias=bias)
        else:
            assert (
                channels == out_channels
            ), f'"channels" ({channels}) must be the same as "out_channels" ({out_channels}) when "use_conv=False"'
            self.conv = nn.AvgPool2d(stride, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x


class Upsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: int | None = None,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        out_channels = out_channels or channels
        self.conv = (
            nn.Conv2d(channels, out_channels, kernel_size, 1, kernel_size // 2, bias=bias)
            if use_conv
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if x.shape[0] >= 64:
            x = x.contiguous()

        x = F.interpolate(x, scale_factor=2.0, mode='nearest')

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            x = x.to(dtype)

        x = self.conv(x)

        return x


class Block2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        activation: str = 'swish',
        residual_scale_init: float | None = 1.0,
        norm_groups: int = 32,
        eps: float = 1e-6,
        dropout: float = 0.0,
        is_up: bool = False,
    ):
        super().__init__()

        blocks = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            blocks.append(
                ResNetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    activation=activation,
                    residual_scale_init=residual_scale_init,
                    groups=norm_groups,
                    eps=eps,
                    dropout=dropout,
                )
            )
        blocks.append(Upsample2D(out_channels, use_conv=True) if is_up else Downsample2D(out_channels, use_conv=True))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return x


class MidBlock2D(nn.Module):
    def __init__(
        self,
        channels: int,
        num_layers: int = 1,
        activation: str = 'swish',
        attn_heads: int = 8,
        attn_head_dims: int | None = 64,
        attn_qkv_bias: bool = False,
        attn_qk_norm: bool = False,
        attn_rms_norm: bool = False,
        residual_scale_init: float | None = 1.0,
        norm_groups: int = 32,
        eps: float = 1e-6,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        blocks = [
            ResNetBlock2D(
                in_channels=channels,
                out_channels=channels,
                activation=activation,
                residual_scale_init=residual_scale_init,
                groups=norm_groups,
                eps=eps,
                dropout=dropout,
            )
        ]
        for _ in range(num_layers):
            blocks.append(
                Attention(
                    dims=channels,
                    heads=attn_heads,
                    head_dim=attn_head_dims,
                    qkv_bias=attn_qkv_bias,
                    qk_norm=attn_qk_norm,
                    residual_scale_init=residual_scale_init,
                    use_rms_norm=attn_rms_norm,
                    eps=eps,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                )
            )
            blocks.append(
                ResNetBlock2D(
                    in_channels=channels,
                    out_channels=channels,
                    activation=activation,
                    residual_scale_init=residual_scale_init,
                    groups=norm_groups,
                    eps=eps,
                    dropout=dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_out_channels: tuple[int, ...] = (64, 256, 256, 512),  # type: ignore
        layers_per_block: int = 2,
        norm_groups: int = 32,
        activation: str = 'swish',
        residual_scale_init: float = 1e-3,
    ):
        super().__init__()

        self.input = nn.Conv2d(in_channels, block_out_channels[0], 3, 1, 1)

        blocks = []
        block_io_chs = [*block_out_channels, block_out_channels[-1]]
        block_i_chs = block_io_chs[:-1]
        block_o_chs = block_io_chs[1:]

        for ich, och in zip(block_i_chs, block_o_chs):  # noqa: B905
            blocks.append(
                Block2D(
                    ich,
                    och,
                    layers_per_block,
                    activation=activation,
                    residual_scale_init=residual_scale_init,
                    norm_groups=norm_groups,
                    is_up=False,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        self.midblock = MidBlock2D(
            och,
            layers_per_block,
            activation=activation,
        )

        self.output = nn.Sequential(
            nn.GroupNorm(norm_groups, och, eps=1e-6),
            get_activation_cls(activation)(),
            nn.Conv2d(och, out_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.blocks(x)
        x = self.midblock(x)
        x = self.output(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_out_channels: tuple[int, ...] = (64, 256, 256, 512),  # type: ignore
        layers_per_block: int = 2,
        norm_groups: int = 32,
        activation: str = 'swish',
        residual_scale_init: float = 1e-3,
    ):
        super().__init__()
        block_out_channels = list(reversed(block_out_channels))

        self.input = nn.Conv2d(in_channels, block_out_channels[0], 3, 1, 1)

        blocks = []
        block_io_chs = [*block_out_channels, block_out_channels[-1]]
        block_i_chs = block_io_chs[:-1]
        block_o_chs = block_io_chs[1:]

        self.midblock = MidBlock2D(
            block_out_channels[0],
            layers_per_block,
            activation=activation,
        )

        for ich, och in zip(block_i_chs, block_o_chs):  # noqa: B905
            blocks.append(
                Block2D(
                    ich,
                    och,
                    layers_per_block,
                    activation=activation,
                    residual_scale_init=residual_scale_init,
                    norm_groups=norm_groups,
                    is_up=True,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        self.output = nn.Sequential(
            nn.GroupNorm(norm_groups, och, eps=1e-6),
            get_activation_cls(activation)(),
            nn.Conv2d(och, out_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.midblock(x)
        x = self.blocks(x)
        x = self.output(x)
        return x


class VQVAE(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 16384,
        embed_dim: int = 4,
        block_out_channels: tuple[int, ...] = (64, 128, 256, 256),  # type: ignore
        layers_per_block: int = 2,
        activation: str = 'swish',
        image_channels: int = 3,
        residual_scale_init: float = 0.001,
        latent_dim: int | None = None,
    ):
        super().__init__()

        latent_dim = latent_dim or embed_dim

        self.encoder = Encoder(
            image_channels,
            latent_dim,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            activation=activation,
            residual_scale_init=residual_scale_init,
        )

        self.pre_vq_conv = nn.Conv2d(latent_dim, embed_dim, 1)
        self.quantize = VectorQuantizer(num_embeddings=num_embeddings, embed_dim=embed_dim)
        self.pos_vq_conv = nn.Conv2d(embed_dim, latent_dim, 1)

        self.decoder = Decoder(
            latent_dim,
            image_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            activation=activation,
            residual_scale_init=residual_scale_init,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.pre_vq_conv(x)
        return x

    def decode(self, x: torch.Tensor, return_loss: bool = True) -> torch.Tensor:
        x, loss = self.quantize(x)
        x = self.pos_vq_conv(x)
        x = self.decoder(x)

        if return_loss:
            return x, loss
        return x

    def forward(self, x: torch.Tensor, return_loss: bool = True) -> torch.Tensor:
        x = self.encode(x)
        x = self.decode(x, return_loss)
        return x


if __name__ == '__main__':
    model = VQVAE()
    input = torch.randn(4, 3, 256, 256)
    print(model(input, return_loss=False).size())
