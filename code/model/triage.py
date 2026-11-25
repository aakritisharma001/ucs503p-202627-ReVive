"""
Triage logic: buckets a standardized anomaly score into a human-facing
severity level. Thresholds are simple and adjustable from the UI.
"""

def classify(score: float, review_threshold: float = 2.0, high_risk_threshold: float = 4.0) -> str:
    """
    score: robustly-standardized anomaly score (median/IQR scaled).
    Higher = more anomalous. Threshold defaults are in "IQR units above
    the median" -- adjustable via the dashboard sliders.
    """
    if score >= high_risk_threshold:
        return "High Risk"
    elif score >= review_threshold:
        return "Review Needed"
    else:
        return "Normal"


TRIAGE_COLORS = {
    "Normal": "#3FB984",
    "Review Needed": "#F2A93B",
    "High Risk": "#E15554",
}
