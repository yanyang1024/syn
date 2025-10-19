import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return torch.mean(torch.abs(y_pred - y_true)).item()


def rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(
        self,
        patience: int = 30,
        min_delta: float = 0.0,
        mode: str = "min",
        save_path: str = "checkpoints/best_state.pt",
    ):
        assert mode in {"min", "max"}
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def __call__(self, score: float, model: torch.nn.Module, epoch: int) -> bool:
        if self._is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
        return self.counter >= self.patience


@dataclass
class TrainSummary:
    config: Dict[str, Any]
    history: list
    best_epoch: int
    best_val_mse: float
    best_metrics_raw: Dict[str, float]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
