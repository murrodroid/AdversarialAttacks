import torch
from torch import nn, Tensor
from typing import Callable
from torchvision.ops.stochastic_depth import StochasticDepth
from src.finetuning.shifted_window_attention import ShiftedWindowAttention


class MLP(nn.Sequential):
    def __init__(self, in_features, hidden_features, activation_layer=nn.GELU, dropout=0.0):
        super().__init__(
            nn.Linear(in_features, hidden_features[0]),
            activation_layer(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features[0], hidden_features[1]),
            nn.Dropout(dropout),
        )


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: list[int],
        shift_size: list[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
        temperature: float = 1.5,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim=dim,
            window_size=window_size,
            shift_size=shift_size,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
            temperature=temperature,
        )

        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=[int(dim * mlp_ratio), dim],
            activation_layer=nn.GELU,
            dropout=dropout
        )

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x
