from __future__ import annotations

import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed_dim = embed_dim
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        torch._assert(x.size(1) == self.embed_dim, f'Number of embeddings must be "{self.embed_dim}".')
        x = x.permute(0, 2, 3, 1).contiguous()
        x_flattened = x.reshape(-1, self.embed_dim)

        closest_indices = torch.argmin(torch.cdist(x_flattened, self.embedding.weight), dim=1)

        x_q = self.embedding(closest_indices).view(x.shape)

        if self.training:
            loss = torch.mean((x_q.detach() - x) ** 2) + self.beta * torch.mean((x_q - x.detach()) ** 2)
        else:
            loss = None

        x_q: torch.Tensor = x + (x_q - x).detach()

        x_q = x_q.permute(0, 3, 1, 2).contiguous()

        return x_q, loss

    def get_codebook_entry(self, indices: torch.LongTensor, shape: tuple[int, ...]) -> torch.Tensor:
        z_q: torch.Tensor = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


if __name__ == '__main__':
    quantize = VectorQuantizer(8192 * 2, 4)
    input = torch.randn(4, 4, 14, 14)
    output, loss = quantize(input)
    print(output.size(), loss)
