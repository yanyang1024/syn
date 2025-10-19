import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class RegressionDataset(Dataset):
    """Holds tensors already normalized with pre-fit scalers."""

    def __init__(self, x_np: np.ndarray, y_np: np.ndarray, scaler_x: StandardScaler, scaler_y: StandardScaler):
        self.x = torch.tensor(scaler_x.transform(x_np), dtype=torch.float32)
        self.y = torch.tensor(scaler_y.transform(y_np), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def _load_arrays(x_path: str, y_path: str) -> tuple[np.ndarray, np.ndarray]:
    x = pd.read_csv(x_path, header=None).values.astype("float32")
    y = pd.read_csv(y_path, header=None).values.astype("float32")
    assert len(x) == len(y), "样本数不一致"
    return x, y


def make_dataloaders(
    batch_size: int = 64,
    split: float = 0.85,
    seed: int = 42,
    x_path: str = "x_input.csv",
    y_path: str = "y_output.csv",
):
    x, y = _load_arrays(x_path, y_path)
    n_samples = len(x)
    assert 0.0 < split < 1.0, "split 必须在 (0, 1) 内"

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    train_cut = int(split * n_samples)
    train_idx, val_idx = indices[:train_cut], indices[train_cut:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    scaler_x = StandardScaler().fit(x_train)
    scaler_y = StandardScaler().fit(y_train)

    train_ds = RegressionDataset(x_train, y_train, scaler_x, scaler_y)
    val_ds = RegressionDataset(x_val, y_val, scaler_x, scaler_y)

    drop_last = len(train_ds) > batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, scaler_x, scaler_y
