"""
Inference pipeline for ReVive.

Implements the four-step inference procedure from the paper (Section III-C):
  1. Extract overlapping windows with stride 1
  2. Compute the structural discrepancy score s(x) for each window
  3. Robustly standardize using the training-set median/IQR
  4. Map window scores back onto the original time axis by averaging overlaps
"""

import numpy as np
import torch

from scoring import structural_score, robust_standardize


def extract_windows(series: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """series: (T, C) -> windows: (N, window_size, C)"""
    T, C = series.shape
    windows = []
    for start in range(0, T - window_size + 1, stride):
        windows.append(series[start:start + window_size])
    return np.stack(windows, axis=0)


@torch.no_grad()
def score_series(
    model,
    series: np.ndarray,
    window_size: int,
    ref_scores: np.ndarray,
    device: str = "cpu",
    w_diff: float = 0.5,
    w_trend: float = 0.5,
    w_corr: float = 0.25,
) -> np.ndarray:
    """
    Run the full inference pipeline on a single time series and return
    a per-timestep anomaly score (same length as the input series).

    ref_scores: raw (un-standardized) scores computed on the training
                set, used for robust median/IQR standardization.
    """
    model.eval()
    T, C = series.shape
    windows = extract_windows(series, window_size, stride=1)  # (N, W, C)

    windows_t = torch.tensor(windows, dtype=torch.float32, device=device)
    repaired_t = model(windows_t)  # no corruption applied at inference
    repaired = repaired_t.cpu().numpy()

    raw_scores = np.array([
        structural_score(windows[i], repaired[i], w_diff, w_trend, w_corr)
        for i in range(len(windows))
    ])

    standardized = robust_standardize(raw_scores, ref_scores)

    # Map window-level scores back onto the time axis by averaging overlaps.
    timeline_scores = np.zeros(T)
    counts = np.zeros(T)
    for i, score in enumerate(standardized):
        timeline_scores[i:i + window_size] += score
        counts[i:i + window_size] += 1
    counts[counts == 0] = 1  # avoid division by zero at edges
    timeline_scores /= counts

    return timeline_scores


def compute_reference_scores(model, train_series: np.ndarray, window_size: int, device: str = "cpu") -> np.ndarray:
    """
    Compute raw (un-standardized) structural scores on the training set,
    used as the reference distribution for robust standardization on
    new/test data.
    """
    model.eval()
    windows = extract_windows(train_series, window_size, stride=1)
    windows_t = torch.tensor(windows, dtype=torch.float32, device=device)
    with torch.no_grad():
        repaired_t = model(windows_t)
    repaired = repaired_t.cpu().numpy()

    raw_scores = np.array([
        structural_score(windows[i], repaired[i])
        for i in range(len(windows))
    ])
    return raw_scores
