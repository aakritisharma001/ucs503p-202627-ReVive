"""
Training pipeline for ReVive.

Implements the training recipe from the paper (Section III-A):
  - Corruption: Gaussian noise (sigma=0.1) + random channel masking (p=0.05)
  - Loss: Huber loss on amplitude + 0.25 * Huber loss on first-difference
  - Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
  - Batch size 128, max 30 epochs, early stopping patience 3

The network is trained ONLY on normal (non-anomalous) windows -- this is
an unsupervised approach. It never sees anomaly labels during training;
labels are only used afterward, at evaluation time, to check how well
the anomaly SCORE separates known anomalies from normal data.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from repairnet import ReVive


def corrupt(x: torch.Tensor, sigma: float = 0.1, mask_p: float = 0.05) -> torch.Tensor:
    """
    Apply Gaussian noise + random channel masking to a batch of windows.
    x: (B, W, C)
    """
    noise = torch.randn_like(x) * sigma
    x_corrupted = x + noise

    # Channel masking: randomly zero out entire channels for some fraction
    # of windows in the batch, encouraging cross-channel repair learning.
    B, W, C = x.shape
    mask = (torch.rand(B, 1, C, device=x.device) > mask_p).float()
    x_corrupted = x_corrupted * mask

    return x_corrupted


def repair_loss(pred: torch.Tensor, target: torch.Tensor, lambda_diff: float = 0.25) -> torch.Tensor:
    """
    L = Huber(pred, target) + lambda_diff * Huber(diff(pred), diff(target))
    """
    huber = nn.HuberLoss()
    amp_loss = huber(pred, target)

    pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
    target_diff = target[:, 1:, :] - target[:, :-1, :]
    diff_loss = huber(pred_diff, target_diff)

    return amp_loss + lambda_diff * diff_loss


class WindowDataset(Dataset):
    """Wraps a (N, W, C) array of normal windows for training."""

    def __init__(self, windows: torch.Tensor):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


def train_repairnet(
    train_windows: torch.Tensor,
    val_windows: torch.Tensor,
    num_channels: int,
    hidden_dim: int = 128,
    sigma: float = 0.1,
    mask_p: float = 0.05,
    lambda_diff: float = 0.25,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 128,
    max_epochs: int = 30,
    patience: int = 3,
    device: str = "cpu",
):
    """
    Full training loop following the paper's recipe. Returns the trained
    model and a history dict of train/val losses per epoch.
    """
    model = ReVive(num_channels=num_channels, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(WindowDataset(train_windows), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(val_windows), batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            corrupted = corrupt(batch, sigma=sigma, mask_p=mask_p)

            optimizer.zero_grad()
            repaired = model(corrupted)
            loss = repair_loss(repaired, batch, lambda_diff=lambda_diff)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                corrupted = corrupt(batch, sigma=sigma, mask_p=mask_p)
                repaired = model(corrupted)
                loss = repair_loss(repaired, batch, lambda_diff=lambda_diff)
                val_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:2d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


if __name__ == "__main__":
    # Smoke test with synthetic data, to confirm the training loop runs
    # end-to-end before plugging in the real financial dataset.
    torch.manual_seed(0)
    W, C = 100, 3
    n_train, n_val = 512, 128

    # Synthetic "normal" data: smooth sinusoids + noise
    def make_windows(n):
        t = torch.linspace(0, 4 * 3.14159, W)
        base = torch.stack([torch.sin(t + i) for i in range(C)], dim=-1)
        data = base.unsqueeze(0) + 0.05 * torch.randn(n, W, C)
        return data

    train_windows = make_windows(n_train)
    val_windows = make_windows(n_val)

    model, history = train_repairnet(
        train_windows, val_windows, num_channels=C,
        max_epochs=5, patience=2,  # small for smoke test speed
    )
    print("\nSmoke test complete. Model trained without errors.")
    print(f"Final train loss: {history['train_loss'][-1]:.5f}")
    print(f"Final val loss:   {history['val_loss'][-1]:.5f}")
