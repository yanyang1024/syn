import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from dataset import make_dataloaders
from model import MLPRegressor
from utils import EarlyStopping, TrainSummary, count_parameters, mae, rmse


@dataclass
class TrainConfig:
    epochs: int = 400
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    split: float = 0.85
    seed: int = 42
    hidden: int = 256
    depth: int = 6
    expansion: int = 2
    dropout: float = 0.2
    head_dropout: float = 0.1
    patience: int = 60
    min_delta: float = 1e-4
    grad_clip: Optional[float] = 1.0
    amp: bool = True
    lr_factor: float = 0.5
    scheduler_patience: int = 12
    min_lr: float = 1e-6
    log_every: int = 10
    checkpoint_dir: str = "checkpoints"
    x_path: str = "x_input.csv"
    y_path: str = "y_output.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: Optional[float],
    amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp and device.type == "cuda"):
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp: bool,
) -> Dict[str, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        with autocast(enabled=amp and device.type == "cuda"):
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
        total_loss += loss.item() * len(xb)
        preds.append(pred.cpu())
        targets.append(yb.cpu())
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    return {"mse": total_loss / len(loader.dataset), "preds": preds, "targets": targets}


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train MLP regressor")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--split", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=12)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--x-path", type=str, default="x_input.csv")
    parser.add_argument("--y-path", type=str, default="y_output.csv")
    args = parser.parse_args()
    grad_clip = None if args.grad_clip <= 0 else args.grad_clip
    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        split=args.split,
        seed=args.seed,
        hidden=args.hidden,
        depth=args.depth,
        expansion=args.expansion,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        patience=args.patience,
        min_delta=args.min_delta,
        grad_clip=grad_clip,
        amp=not args.no_amp,
        lr_factor=args.lr_factor,
        scheduler_patience=args.scheduler_patience,
        min_lr=args.min_lr,
        log_every=args.log_every,
        checkpoint_dir=args.checkpoint_dir,
        x_path=args.x_path,
        y_path=args.y_path,
    )


def run(config: TrainConfig) -> TrainSummary:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, scaler_x, scaler_y = make_dataloaders(
        batch_size=config.batch_size, split=config.split, seed=config.seed, x_path=config.x_path, y_path=config.y_path
    )

    input_dim = train_loader.dataset.x.shape[1]
    output_dim = train_loader.dataset.y.shape[1]
    model_kwargs = dict(
        in_dim=input_dim,
        out_dim=output_dim,
        hidden=config.hidden,
        depth=config.depth,
        expansion=config.expansion,
        dropout=config.dropout,
        head_dropout=config.head_dropout,
    )
    model = MLPRegressor(**model_kwargs).to(device)
    total_params = count_parameters(model)
    print(f"Model parameters: {total_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_factor, patience=config.scheduler_patience, min_lr=config.min_lr
    )
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_state_path = ckpt_dir / "best_state.pt"
    early_stop = EarlyStopping(patience=config.patience, min_delta=config.min_delta, mode="min", save_path=str(best_state_path))
    scaler = GradScaler(enabled=config.amp and device.type == "cuda")

    history: List[Dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        train_mse = train_one_epoch(model, train_loader, optimizer, scaler, device, config.grad_clip, config.amp)
        val_results = evaluate(model, val_loader, device, config.amp)
        val_mse = val_results["mse"]
        val_preds_raw = torch.from_numpy(scaler_y.inverse_transform(val_results["preds"].numpy())).float()
        val_targets_raw = torch.from_numpy(scaler_y.inverse_transform(val_results["targets"].numpy())).float()
        val_mae = mae(val_preds_raw, val_targets_raw)
        val_rmse = rmse(val_preds_raw, val_targets_raw)

        scheduler.step(val_mse)

        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "val_mae_raw": val_mae,
                "val_rmse_raw": val_rmse,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if epoch == 1 or epoch % config.log_every == 0:
            print(
                f"Epoch {epoch:03d} | train MSE {train_mse:.5f} | val MSE {val_mse:.5f} "
                f"| val MAE(raw) {val_mae:.5f} | val RMSE(raw) {val_rmse:.5f} | lr {optimizer.param_groups[0]['lr']:.2e}"
            )

        if early_stop(val_mse, model, epoch):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    best_epoch = early_stop.best_epoch or history[-1]["epoch"]
    best_entry = next(item for item in history if item["epoch"] == best_epoch)

    # Load and persist final artifacts
    state_dict = torch.load(best_state_path, map_location="cpu")
    model.load_state_dict(state_dict)
    torch.save({"state_dict": state_dict, "model_kwargs": model_kwargs}, ckpt_dir / "best.pt")
    torch.save({"scaler_x": scaler_x, "scaler_y": scaler_y}, ckpt_dir / "scalers.pt")

    summary = TrainSummary(
        config={**config.__dict__, "model_kwargs": model_kwargs},
        history=history,
        best_epoch=best_epoch,
        best_val_mse=best_entry["val_mse"],
        best_metrics_raw={"mae": best_entry["val_mae_raw"], "rmse": best_entry["val_rmse_raw"]},
    )
    summary.save(str(ckpt_dir / "training_summary.json"))
    print(
        f"Best epoch {best_epoch} | val MSE {best_entry['val_mse']:.5f} | "
        f"val MAE(raw) {best_entry['val_mae_raw']:.5f} | val RMSE(raw) {best_entry['val_rmse_raw']:.5f}"
    )
    return summary


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
