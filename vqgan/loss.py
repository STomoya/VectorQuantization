import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS
from torch.autograd import grad


def safe_div(numer, denom, eps=1e-8):
    return numer / denom.clamp(min=eps)


def grad_layer_wrt_loss(loss, weight):
    return calc_grad(outputs=loss, inputs=weight).detach()


def calc_grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Calculate gradients."""
    ones = torch.ones(outputs.size(), device=outputs.device)
    gradients = grad(outputs=outputs, inputs=inputs, grad_outputs=ones, retain_graph=True)[0]
    return gradients


class NonSaturatingGANLoss:
    def __init__(self, perceptual: bool = True, lpips_net: str = 'vgg', decoder_last_layer: nn.Module | None = None):
        self.perceptual = perceptual
        if self.perceptual:
            assert decoder_last_layer is not None, 'decoder_last_layer must be given.'

        if self.perceptual:
            self.lpips_fn = LPIPS(net=lpips_net)
            self.lpips_fn.eval()
            self.decoder_last_layer = decoder_last_layer
        else:
            self.lpips_fn = None
            self.decoder_last_layer = None

    def to(self, device: torch.device):
        if self.perceptual:
            self.lpips_fn.to(device)

    @staticmethod
    def fake_loss(logits: torch.Tensor) -> torch.Tensor:
        return F.softplus(logits).mean()

    @staticmethod
    def real_loss(logits: torch.Tensor) -> torch.Tensor:
        return F.softplus(-logits).mean()

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        real_loss = self.real_loss(real_logits)
        fake_loss = self.fake_loss(fake_logits)
        return real_loss + fake_loss

    def g_loss(
        self,
        fake_logits: torch.Tensor,
        real: torch.Tensor | None = None,
        fake: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.perceptual:
            torch._assert(real is not None and fake is not None, '"real" and "fake" must be given.')

        real_loss = self.real_loss(fake_logits)

        perceptual_loss = 0.0
        if self.perceptual:
            perceptual_loss = self.lpips_fn(
                F.interpolate(real, (224, 224), mode='bilinear'), F.interpolate(fake, (224, 224), mode='bilinear')
            ).mean()

            grad_norm_gan_loss = grad_layer_wrt_loss(real_loss, self.decoder_last_layer.weight).norm(p=2)
            grad_norm_vgg_loss = grad_layer_wrt_loss(perceptual_loss, self.decoder_last_layer.weight).norm(p=2)
            adaptive_weight = safe_div(grad_norm_vgg_loss, grad_norm_gan_loss)
            adaptive_weight.clamp_(max=1e4)
            real_loss = real_loss * adaptive_weight

        return real_loss, perceptual_loss
