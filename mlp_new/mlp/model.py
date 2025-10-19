import math
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Pre-norm residual block with optional expansion and dropout."""

    def __init__(self, width: int, expansion: int = 2, dropout: float = 0.2):
        super().__init__()
        hidden = width * expansion
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, hidden)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, width)
        self.dropout2 = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        return residual + self.alpha * out


class MLPRegressor(nn.Module):
    """
    Residual MLP tailored for medium-sized tabular regression.

    Args:
        in_dim: number of input features.
        out_dim: number of output targets.
        hidden: width of hidden representation.
        depth: number of residual blocks.
        expansion: expansion ratio inside blocks.
        dropout: dropout probability applied inside residual blocks.
        head_dropout: dropout probability before the output projection.
    """

    def __init__(
        self,
        in_dim: int = 8,
        out_dim: int = 70,
        hidden: int = 256,
        depth: int = 6,
        expansion: int = 2,
        dropout: float = 0.2,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, expansion, dropout) for _ in range(depth)])
        neck_dim = max(hidden // 2, out_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, neck_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(neck_dim, out_dim),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


if __name__ == "__main__":
    dummy = torch.randn(16, 8)
    model = MLPRegressor()
    print(model(dummy).shape)
