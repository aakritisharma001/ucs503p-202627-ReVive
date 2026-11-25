"""
Data preparation for the Kaggle Credit Card Fraud dataset.

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Download creditcard.csv and place it in this data/ folder before running.

The dataset contains ~285,000 transactions over ~2 days, with 30 numeric
features (Time, V1-V28 PCA-transformed features, Amount) and a binary
'Class' label (1 = fraud, 0 = normal). Fraud cases are ~0.17% of the data.

Because ReVive (like the original JuRe paper) trains ONLY on
normal data, we split as follows:
  - Training set: normal transactions only, used to teach the model
    what "normal" looks like
  - Test set: a mix of normal + fraud transactions, used to evaluate
    how well the anomaly score separates the two

Since this is transactional (not naturally sequential) data, we treat
consecutive transactions ordered by 'Time' as a pseudo-time-series, which
is a reasonable and commonly used approach for adapting tabular fraud
data to sequence-based anomaly detectors.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_prepare(csv_path: str, window_size: int = 100, train_frac: float = 0.6):
    """
    Loads creditcard.csv, sorts by Time, normalizes features, and splits
    into normal-only training windows and mixed test windows with labels.

    Returns:
        train_windows: (N_train, window_size, C) normal-only windows
        test_windows:  (N_test, window_size, C) mixed windows
        test_labels:   (N_test,) 1 if window contains ANY fraud, else 0
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1.")
    df = pd.read_csv(csv_path).sort_values("Time", kind="stable", ignore_index=True)
    feature_cols = [c for c in df.columns if c not in ("Time", "Class")]
    features, labels = validate_inputs(df[feature_cols].to_numpy(), df["Class"].to_numpy())
    split = int(len(features) * train_frac)
    scaler = fit_normal_scaler(features[:split], labels[:split])

    train_windows = normal_windows(
        scaler.transform(features[:split]), labels[:split], window_size
    )
    test_windows, test_labels = _make_windows_with_labels(
        scaler.transform(features[split:]), labels[split:], window_size
    )
    if not len(train_windows) or not len(test_windows):
        raise ValueError("The split needs normal training windows and at least one test window.")
    return train_windows, test_windows, test_labels, scaler, feature_cols


def validate_inputs(features, labels):
    try:
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Feature columns must be numeric and labels must be 0 or 1.") from exc
    if features.shape[1] == 0 or not np.isfinite(features).all():
        raise ValueError("Provide at least one numeric feature and remove missing or infinite values.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1.")
    return features, labels.astype(int)


def fit_normal_scaler(features, labels):
    normal = features[labels == 0]
    if not len(normal):
        raise ValueError("The training split has no normal rows.")
    return StandardScaler().fit(normal)


def normal_windows(features, labels, window_size, stride=1):
    """Filter complete windows so adjacent rows keep their original order."""
    windows, flags = _make_windows_with_labels(features, labels, window_size, stride)
    return windows[flags == 0]


def _make_windows(features: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    windows = []
    for start in range(0, len(features) - window_size + 1, stride):
        windows.append(features[start:start + window_size])
    return np.stack(windows, axis=0) if windows else np.empty((0, window_size, features.shape[1]))


def _make_windows_with_labels(features: np.ndarray, labels: np.ndarray, window_size: int, stride: int = 1):
    windows, window_labels = [], []
    for start in range(0, len(features) - window_size + 1, stride):
        windows.append(features[start:start + window_size])
        # a window is "anomalous" if it contains at least one fraud row
        window_labels.append(1 if labels[start:start + window_size].sum() > 0 else 0)
    return (
        np.stack(windows, axis=0) if windows else np.empty((0, window_size, features.shape[1])),
        np.array(window_labels, dtype=int),
    )


if __name__ == "__main__":
    import os
    csv_path = os.path.join(os.path.dirname(__file__), "creditcard.csv")
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}")
        print("Download it from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("and place creditcard.csv in the data/ folder.")
    else:
        train_w, test_w, test_l, scaler, cols = load_and_prepare(csv_path)
        print(f"Train windows (normal only): {train_w.shape}")
        print(f"Test windows (mixed):        {test_w.shape}")
        print(f"Test anomaly rate:           {test_l.mean():.4f}")
        print(f"Feature columns used:        {len(cols)} -> {cols[:5]}...")
