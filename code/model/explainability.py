"""
Explainability module for ReVive.

The original JuRe paper outputs only a single aggregate discrepancy
score per window. This module is this project's novel addition: it
breaks that score down per-feature, so a reviewer can see WHICH
features drove a flagged anomaly, not just that something was flagged.

Approach: compute the absolute repair error (|observed - repaired|)
for each feature individually, averaged across the window, then rank
features by contribution. This reuses the same repaired output the
scoring function already computes -- no extra model inference needed.
"""

import numpy as np


def explain_window(observed: np.ndarray, repaired: np.ndarray, feature_names: list, top_k: int = 3) -> dict:
    """
    observed, repaired: (W, C) arrays for a single window.
    feature_names: list of length C with human-readable (or column) names.
    Returns a dict with per-feature contribution and the top-k drivers.
    """
    per_feature_error = np.mean(np.abs(observed - repaired), axis=0)  # (C,)
    total = per_feature_error.sum() + 1e-8
    contributions = per_feature_error / total  # normalize to sum to 1

    ranked_idx = np.argsort(-contributions)
    top_features = [
        {"feature": feature_names[i], "contribution": float(contributions[i])}
        for i in ranked_idx[:top_k]
    ]

    # Also find which timestep had the largest total repair error, for
    # "when in the window did this look most anomalous" context.
    per_timestep_error = np.mean(np.abs(observed - repaired), axis=1)  # (W,)
    peak_timestep = int(np.argmax(per_timestep_error))

    return {
        "top_features": top_features,
        "peak_timestep_in_window": peak_timestep,
        "all_feature_contributions": {
            feature_names[i]: float(contributions[i]) for i in range(len(feature_names))
        },
    }
