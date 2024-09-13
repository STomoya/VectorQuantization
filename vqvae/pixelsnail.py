from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vq.config import ConfigMixin


class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.utils.parametrizations.weight_norm(self)


class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.utils.parametrizations.weight_norm(self)


class DownShiftConv(Conv2d):
    def forward(self, x: torch.Tensor):
        kh, kw = self.kernel_size
        x = F.pad(x, (kw // 2, kw // 2, kh - 1, 0))
        x = super().forward(x)
        return x


class DownRightShiftConv(Conv2d):
    def forward(self, x: torch.Tensor):
        kh, kw = self.kernel_size
        x = F.pad(x, (kw - 1, 0, kh - 1, 0))
        x = super().forward(x)
        return x


def shift_down(input, size=1):
    return F.pad(input, [0, 0, size, 0])[:, :, : input.shape[2], :]


def shift_right(input, size=1):
    return F.pad(input, [size, 0, 0, 0])[:, :, :, : input.shape[3]]


def concat_elu(x: torch.Tensor) -> torch.Tensor:
    x = F.elu(torch.cat([x, -x], dim=1))
    return x


class GatedResidual(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dropout: float = 0.0,
        aux_channels: int = 0,
        causal: bool = False,
        mid_channels: int | None = None,
    ):
        super().__init__()

        mid_channels = mid_channels or channels

        ConvCls = DownShiftConv if causal else Conv2d
        kernel_size = (kernel_size // 2 + 1, kernel_size) if causal else kernel_size

        self.conv1 = ConvCls(channels * 2, mid_channels, kernel_size)

        if aux_channels > 0:
            self.aux_conv = Conv2d(aux_channels * 2, mid_channels, 1)
        else:
            self.aux_conv = None

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.conv2 = ConvCls(mid_channels * 2, channels * 2, kernel_size)

        self.gate = nn.GLU(dim=1)

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None):
        skip = x

        x = concat_elu(x)
        x = self.conv1(x)

        if self.aux_conv:
            aux = concat_elu(aux)
            aux = self.aux_conv(aux)
            x = x + aux

        x = concat_elu(x)
        x = self.dropout(x)
        x = self.conv2(x)

        x = self.gate(x)
        x = skip + x

        return x


class CausalAttention(nn.Module):
    def __init__(
        self,
        dims: int,
        q_dims: int,
        k_dims: int,
        qkv_bias: bool = False,
        dropout: float = 0.0,
        num_heads: int = 8,
    ):
        super().__init__()
        assert dims % num_heads == 0

        self.q_dims = q_dims
        self.k_dims = self.v_dims = k_dims

        self.num_heads = num_heads
        self.head_dims = dims // self.num_heads
        self.inner_dims = self.head_dims * self.num_heads
        self.scale = self.head_dims**-0.5

        self.to_q = Conv2d(self.q_dims, self.inner_dims, 1, bias=qkv_bias)
        self.to_kv = Conv2d(self.k_dims, self.inner_dims * 2, 1, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        self.use_flash = hasattr(F, 'scaled_dot_product_attention')

    @staticmethod
    @lru_cache(maxsize=32)
    def causal_mask(size):
        shape = [size, size]
        mask = np.tril(np.ones(shape)).astype(np.uint8)
        return torch.from_numpy(mask).unsqueeze(0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, force_native: bool = False):
        B, _, H, W = q.size()
        q = self.to_q(q)
        kv = self.to_kv(k)
        k, v = kv.chunk(2, dim=1)

        q = q.reshape(B, self.num_heads, self.head_dims, H * W).transpose(-1, -2)
        k = k.reshape(B, self.num_heads, self.head_dims, H * W).transpose(-1, -2)
        v = v.reshape(B, self.num_heads, self.head_dims, H * W).transpose(-1, -2)

        if not force_native and self.use_flash:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-1, -2)
            mask = self.causal_mask(H * W)
            attn = attn.masked_fill(mask.to(attn) == 0, float('-inf'))
            attn = attn.softmax(-1)
            attn = self.dropout(attn)
            x = attn @ v

        x = x.transpose(-1, -2).reshape(B, -1, H, W)
        return x


class Block(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        num_res_blocks: int = 5,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        num_heads: int = 8,
    ):
        super().__init__()

        layers = []
        for _ in range(num_res_blocks):
            layers.append(
                GatedResidual(
                    channels=channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    causal=True,
                )
            )
        self.resblocks = nn.Sequential(*layers)

        self.pre_to_q = GatedResidual(channels=channels + 2, kernel_size=1, dropout=dropout)

        self.pre_to_k = GatedResidual(channels=channels * 2 + 2, kernel_size=1, dropout=dropout)

        self.attn = CausalAttention(
            dims=channels,
            q_dims=channels + 2,
            k_dims=channels * 2 + 2,
            qkv_bias=qkv_bias,
            dropout=attn_dropout,
            num_heads=num_heads,
        )

        self.out = GatedResidual(channels=channels, kernel_size=1, dropout=dropout, aux_channels=channels)

    def forward(self, x: torch.Tensor, bg: torch.Tensor):
        res = self.resblocks(x)

        q = self.pre_to_q(torch.cat([x, bg], dim=1))
        k = self.pre_to_k(torch.cat([x, res, bg], dim=1))
        attn = self.attn(q, k)

        x = self.out(x, attn)

        return x


class PixelSNAIL(nn.Module, ConfigMixin):
    def __init__(
        self,
        num_embeddings: int,
        channels: int,
        image_dims: tuple[int, int],
        kernel_size: int = 3,
        num_blocks: int = 12,
        num_res_blocks: int = 5,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        qkv_bias: bool = False,
        num_heads: int = 8,
    ):
        super().__init__()
        H, W = image_dims

        background_v = torch.linspace(-1.0, 1.0, W)
        background_h = torch.linspace(-1.0, 1.0, H)
        background_h, background_v = torch.meshgrid(background_h, background_v, indexing='ij')
        background = torch.stack([background_h, background_v], dim=0).unsqueeze(0)
        self.register_buffer('background', background)

        self.down_conv = DownShiftConv(num_embeddings, channels, (kernel_size // 2, kernel_size))
        self.downright_conv = DownRightShiftConv(num_embeddings, channels, ((kernel_size + 1) // 2, kernel_size // 2))

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                Block(
                    channels=channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    num_res_blocks=num_res_blocks,
                    qkv_bias=qkv_bias,
                    attn_dropout=attn_dropout,
                    num_heads=num_heads,
                )
            )

        self.out = nn.Sequential(nn.ELU(), Conv2d(channels, num_embeddings, 1))
        self.num_embeddings = num_embeddings

    def forward(self, x: torch.Tensor):
        x = F.one_hot(x, self.num_embeddings).permute(0, 3, 1, 2).to(self.background)

        d = shift_down(self.down_conv(x))
        dr = shift_right(self.downright_conv(x))
        x = d + dr

        bg = self.background.repeat(x.size(0), 1, 1, 1)[:, :, : x.size(2), :]
        for block in self.blocks:
            x = block(x, bg)

        x = self.out(x)
        return x


@torch.no_grad()
def sample(model: PixelSNAIL, size: tuple[int, int, int, int], device: torch.device) -> torch.Tensor:
    model.eval()

    _, H, W = size
    indices = torch.zeros(size, device=device, dtype=torch.long)

    for h in range(H):
        for w in range(W):
            logits = model(indices[:, : h + 1, :])
            probs = logits[:, :, h, w].softmax(1)
            index = torch.multinomial(probs, 1).squeeze()
            indices[:, h, w] = index

    model.train()
    return indices
