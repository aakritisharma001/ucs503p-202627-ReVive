"""
Classical baseline anomaly detectors for a fair state-of-the-art comparison.

IMPORTANT: these baselines are run on the SAME financial dataset as
ReVive. This is the honest way to build a "state of the art" table
for this project -- comparing methods against each other on our own data,
NOT comparing our numbers against the original JuRe paper's reported
numbers (which were measured on completely different sensor/industrial
datasets and are not a fair comparison point).

Both methods below are also cited directly in the JuRe paper's own
related-work and benchmark comparisons:
  - Isolation Forest (Liu, Ting, Zhou, 2008)
  - Local Outlier Factor / LOF (Breunig et al., 2000)
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score, roc_auc_score


def run_isolation_forest(train_features: np.ndarray, test_features: np.ndarray, random_state: int = 0) -> np.ndarray:
    """
    Train on normal-only data, score test data.
    Returns anomaly scores where HIGHER = more anomalous (matches
    ReVive's score convention).
    """
    clf = IsolationForest(random_state=random_state, contamination="auto")
    clf.fit(train_features)
    # decision_function: higher = more normal, so we negate it
    scores = -clf.decision_function(test_features)
    return scores


def run_lof(train_features: np.ndarray, test_features: np.ndarray, n_neighbors: int = 20) -> np.ndarray:
    """
    LOF with novelty=True allows scoring new/unseen data after fitting
    on the training (normal-only) set.
    """
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    clf.fit(train_features)
    scores = -clf.decision_function(test_features)
    return scores


def evaluate(scores: np.ndarray, labels: np.ndarray) -> dict:
    """
    labels: binary array, 1 = anomaly, 0 = normal.
    Returns AUC-PR and AUC-ROC, the same metrics the paper uses.
    """
    return {
        "AUC-PR": average_precision_score(labels, scores),
        "AUC-ROC": roc_auc_score(labels, scores),
    }


if __name__ == "__main__":
    # Smoke test with synthetic data containing a few obvious anomalies.
    rng = np.random.default_rng(0)
    n_train, n_test = 500, 200
    train_data = rng.normal(size=(n_train, 5))
    test_data = rng.normal(size=(n_test, 5))

    # Inject anomalies into a subset of the test set
    labels = np.zeros(n_test)
    anomaly_idx = rng.choice(n_test, size=20, replace=False)
    test_data[anomaly_idx] += rng.normal(scale=5, size=(20, 5))
    labels[anomaly_idx] = 1

    if_scores = run_isolation_forest(train_data, test_data)
    lof_scores = run_lof(train_data, test_data)

    print("Isolation Forest:", evaluate(if_scores, labels))
    print("LOF:             ", evaluate(lof_scores, labels))
