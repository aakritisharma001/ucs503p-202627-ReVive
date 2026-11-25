"""
Structural discrepancy scoring for ReVive.

Implements the fixed, non-learned scoring function from the paper
(Section III-C, Eq. 2):

    s(x) = s_amp + w_diff * s_diff + w_trend * s_trend + w_corr * s_corr

where:
    s_amp   = mean absolute amplitude error between observed and repaired window
    s_diff  = mean absolute error between first-differences (temporal gradient)
    s_trend = mean absolute error between moving-average trends (kernel width W//10)
    s_corr  = RMS change in the upper-triangular cross-channel correlation matrix
              (0 for single-channel data)

Default weights (paper): w_diff=0.5, w_trend=0.5, w_corr=0.25
No parameter here is learned or fitted -- this score is applied identically
at inference regardless of dataset, exactly as in the original paper.
"""

import numpy as np


def _moving_average(x: np.ndarray, kernel_width: int) -> np.ndarray:
    """Simple moving average along the time axis. x: (W, C)"""
    if kernel_width < 1:
        kernel_width = 1
    kernel = np.ones(kernel_width) / kernel_width
    out = np.zeros_like(x)
    for c in range(x.shape[1]):
        out[:, c] = np.convolve(x[:, c], kernel, mode="same")
    return out


def _upper_tri_corr(x: np.ndarray) -> np.ndarray:
    """Upper-triangular cross-channel correlation matrix entries. x: (W, C)"""
    c = x.shape[1]
    if c <= 1:
        return np.array([])
    corr = np.corrcoef(x, rowvar=False)  # (C, C)
    iu = np.triu_indices(c, k=1)
    vals = corr[iu]
    return np.nan_to_num(vals, nan=0.0)  # constant channels -> undefined -> 0


def structural_score(
    observed: np.ndarray,
    repaired: np.ndarray,
    w_diff: float = 0.5,
    w_trend: float = 0.5,
    w_corr: float = 0.25,
) -> float:
    """
    Compute the fixed structural discrepancy score between an observed
    window and its network-repaired counterpart.

    observed, repaired: (W, C) numpy arrays for a single window.
    Returns a single scalar anomaly score (higher = more anomalous).
    """
    W, C = observed.shape

    # amplitude error
    s_amp = np.mean(np.abs(observed - repaired))

    # first-difference (temporal gradient) error
    obs_diff = np.diff(observed, axis=0)
    rep_diff = np.diff(repaired, axis=0)
    s_diff = np.mean(np.abs(obs_diff - rep_diff))

    # trend error (moving-average comparison, kernel width = W // 10)
    kernel_width = max(1, W // 10)
    obs_trend = _moving_average(observed, kernel_width)
    rep_trend = _moving_average(repaired, kernel_width)
    s_trend = np.mean(np.abs(obs_trend - rep_trend))

    # cross-channel correlation change
    if C > 1:
        obs_corr = _upper_tri_corr(observed)
        rep_corr = _upper_tri_corr(repaired)
        s_corr = np.sqrt(np.mean((obs_corr - rep_corr) ** 2)) if obs_corr.size else 0.0
    else:
        s_corr = 0.0

    return float(s_amp + w_diff * s_diff + w_trend * s_trend + w_corr * s_corr)


def robust_standardize(scores: np.ndarray, ref_scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Robustly standardize scores using the median/IQR of a reference
    (typically training-set) score distribution, per the paper's
    inference procedure (Section III-C, step 3).
    """
    median = np.median(ref_scores)
    q75, q25 = np.percentile(ref_scores, [75, 25])
    iqr = q75 - q25 + eps
    return (scores - median) / iqr


if __name__ == "__main__":
    # Sanity check: identical windows should score ~0; a clearly
    # different window should score much higher.
    rng = np.random.default_rng(0)
    w = rng.normal(size=(100, 3))
    score_identical = structural_score(w, w.copy())
    w_anomalous = w.copy()
    w_anomalous[50:60, 0] += 5.0  # inject an amplitude spike
    score_anomalous = structural_score(w, w_anomalous)
    print(f"Score (identical windows): {score_identical:.6f}  (expect ~0)")
    print(f"Score (spike injected):    {score_anomalous:.6f}  (expect >> 0)")
