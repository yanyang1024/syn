from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd
import torch

from model import MLPRegressor


def load_artifacts(checkpoint_dir: Union[str, Path] = "checkpoints", device: Union[str, torch.device] = "cpu"):
    checkpoint_dir = Path(checkpoint_dir)
    ckpt = torch.load(checkpoint_dir / "best.pt", map_location=device)
    scalers = torch.load(checkpoint_dir / "scalers.pt", map_location="cpu")
    model = MLPRegressor(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, scalers["scaler_x"], scalers["scaler_y"]


def _prepare_input(data: Union[str, Iterable[Iterable[float]], np.ndarray]) -> np.ndarray:
    if isinstance(data, str):
        arr = pd.read_csv(data, header=None).values.astype("float32")
    else:
        arr = np.asarray(data, dtype="float32")
    if arr.ndim != 2:
        raise ValueError("输入必须是二维数组，形状为 [样本数, 特征数]")
    return arr


@torch.no_grad()
def predict(
    data: Union[str, Iterable[Iterable[float]], np.ndarray],
    checkpoint_dir: Union[str, Path] = "checkpoints",
    device: Union[str, torch.device] = None,
    return_numpy: bool = True,
):
    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler_x, scaler_y = load_artifacts(checkpoint_dir, device="cpu")
    model.to(device)

    arr = _prepare_input(data)
    x_norm = torch.tensor(scaler_x.transform(arr), dtype=torch.float32, device=device)
    pred_norm = model(x_norm).cpu().numpy()
    pred = scaler_y.inverse_transform(pred_norm)
    if return_numpy:
        return pred
    return torch.from_numpy(pred)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument("--csv", type=str, default="new_input.csv", help="要预测的输入 CSV 文件")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--out", type=str, default="prediction.csv")
    args = parser.parse_args()

    outputs = predict(args.csv, checkpoint_dir=args.checkpoint_dir, return_numpy=True)
    np.savetxt(args.out, outputs, delimiter=",")
    print(f"Saved predictions to {args.out}, shape {outputs.shape}")
