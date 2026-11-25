"""
ReVive — Explainable Financial Anomaly Detection
Streamlit dashboard: load/upload data, train, score, explain, triage.

Run with: streamlit run app.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
sys.path.append(os.path.join(os.path.dirname(__file__), "baselines"))
sys.path.append(os.path.join(os.path.dirname(__file__), "data"))

import time
import sqlite3
import numpy as np
import pandas as pd
import torch
import streamlit as st

from repairnet import ReVive
from train import train_repairnet
from scoring import structural_score, robust_standardize
from explainability import explain_window
from triage import classify, TRIAGE_COLORS
from baselines import run_isolation_forest, run_lof, evaluate as evaluate_baseline
from sklearn.metrics import average_precision_score, roc_auc_score
from pathlib import Path
from prepare_data import fit_normal_scaler, normal_windows, validate_inputs

PUBLIC_DEMO = os.environ.get("REVIVE_PUBLIC_DEMO") == "1"
DB_PATH = Path(os.environ.get("REVIVE_DB_PATH") or Path(__file__).with_name("alerts.db"))


# ---------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------
def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            window_index INTEGER,
            score REAL,
            triage_level TEXT,
            top_feature TEXT,
            top_feature_contribution REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_alerts(alerts: list):
    conn = sqlite3.connect(DB_PATH)
    conn.executemany(
        "INSERT INTO alerts (window_index, score, triage_level, top_feature, top_feature_contribution, timestamp) "
        "VALUES (?, ?, ?, ?, ?, datetime('now'))",
        alerts,
    )
    conn.commit()
    conn.close()


def load_alerts() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM alerts ORDER BY id DESC", conn)
    conn.close()
    return df


def clear_alerts():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM alerts")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------
# Synthetic data generator (fast demo path, no download needed)
# ---------------------------------------------------------------------
def generate_synthetic_data(n_rows=3000, n_features=6, anomaly_rate=0.02, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 40 * np.pi, n_rows)
    data = np.zeros((n_rows, n_features))
    for c in range(n_features):
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(0.8, 1.3)
        data[:, c] = np.sin(freq * t + phase) + 0.1 * rng.normal(size=n_rows)

    labels = np.zeros(n_rows, dtype=int)
    n_anomalies = int(n_rows * anomaly_rate)
    anomaly_starts = rng.choice(n_rows - 20, size=n_anomalies, replace=False)
    for start in anomaly_starts:
        length = rng.integers(3, 10)
        kind = rng.choice(["spike", "trend", "noise"])
        if kind == "spike":
            data[start:start + length, rng.integers(n_features)] += rng.uniform(3, 6)
        elif kind == "trend":
            data[start:start + length, :] += np.linspace(0, rng.uniform(2, 4), length)[:, None]
        else:
            data[start:start + length, :] += rng.normal(scale=2, size=(length, n_features))
        labels[start:start + length] = 1

    cols = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=cols)
    df["label"] = labels
    return df


def make_windows(features: np.ndarray, window_size: int, stride: int = 1):
    windows = []
    for start in range(0, len(features) - window_size + 1, stride):
        windows.append(features[start:start + window_size])
    return np.stack(windows, axis=0) if windows else np.empty((0, window_size, features.shape[1]))


def window_labels(labels: np.ndarray, window_size: int, stride: int = 1):
    out = []
    for start in range(0, len(labels) - window_size + 1, stride):
        out.append(1 if labels[start:start + window_size].sum() > 0 else 0)
    return np.array(out)


# Common column names datasets use for the anomaly/fraud label.
# Checked case-sensitively in this order; first match wins.
LABEL_COLUMN_CANDIDATES = ["label", "Label", "Class", "class", "target", "Target", "fraud", "Fraud", "is_fraud", "isFraud"]


def detect_label_column(df: pd.DataFrame):
    """Returns (has_labels: bool, column_name: str or None)."""
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return True, candidate
    return False, None


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="ReVive — Financial Anomaly Detection", layout="wide")
init_db()

st.title("ReVive")
st.caption("Explainable financial anomaly detection using a corrupt-and-repair denoising network")
if PUBLIC_DEMO:
    torch.set_num_threads(2)
    st.info("Public demo. Click Train & Score to analyze synthetic data.")
    with st.expander("Verified credit-card fraud benchmark"):
        st.markdown(Path(__file__).with_name("VERIFICATION.md").read_text(encoding="utf-8").split(
            "## Software Engineering verification", 1
        )[0])

with st.sidebar:
    st.header("1. Data")
    data_source = "Synthetic demo data"
    if not PUBLIC_DEMO:
        data_source = st.radio("Data source", ["Synthetic demo data", "Upload CSV"])

    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload transaction data (numeric columns + optional label column, e.g. 'label', 'Class', 'target', 'fraud')", type="csv")
        st.caption("For a fast live demo, keep the row limit below small (a few thousand rows). "
                   "The full Credit Card Fraud file (~285k rows) is real data for your reported "
                   "results, but is too slow to train live during a presentation.")
        row_limit = st.number_input(
            "Row limit (rows to actually use, taken from the start of the file)",
            min_value=500, max_value=300000, value=8000, step=500,
        )
    else:
        uploaded = None
        row_limit = None
        n_rows = 3000 if PUBLIC_DEMO else st.slider("Number of rows", 500, 5000, 3000, step=500)
        anomaly_rate = 0.02 if PUBLIC_DEMO else st.slider("Anomaly rate", 0.005, 0.05, 0.02, step=0.005)

    st.header("2. Model settings")
    if PUBLIC_DEMO:
        window_size, hidden_dim, max_epochs, window_stride = 50, 32, 3, 5
        st.caption("Window 50 · hidden width 32 · up to 3 epochs · stride 5")
    else:
        window_size = st.slider("Window size", 20, 100, 50, step=10)
        hidden_dim = st.select_slider("Hidden dimension", options=[32, 64, 128], value=64)
        max_epochs = st.slider("Max training epochs", 3, 30, 10)
        window_stride = st.slider(
            "Window stride (higher = fewer, faster windows)", 1, 20, 5,
            help="Windows are highly overlapping at stride 1, which is accurate but slow. "
                 "A higher stride skips ahead between windows, trading a little resolution "
                 "for a large speed-up -- recommended for live demos.",
        )

    st.header("3. Triage thresholds")
    threshold_mode = st.radio("Threshold mode", ["Validation calibrated", "Manual"])
    review_threshold = st.slider("Review Needed threshold", 0.5, 5.0, 2.0, step=0.1)
    high_risk_threshold = st.slider("High Risk threshold", 1.0, 8.0, 4.0, step=0.1)

    if threshold_mode == "Manual" and high_risk_threshold <= review_threshold:
        st.warning("Set the High Risk threshold above the Review Needed threshold.")
    run_button = st.button("Train & Score", type="primary", width="stretch")
    clear_button = not PUBLIC_DEMO and st.button("Clear stored alerts", width="stretch")

if clear_button:
    clear_alerts()
    st.sidebar.success("Alert history cleared.")

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
if data_source == "Upload CSV" and uploaded is not None:
    try:
        df = pd.read_csv(uploaded, nrows=row_limit + 1)
    except (ValueError, UnicodeError, pd.errors.ParserError):
        st.error("Upload a readable CSV with a header and numeric features.")
        st.stop()
    if len(df) > row_limit:
        df = df.iloc[:row_limit].reset_index(drop=True)
        st.info(f"Using the first {row_limit:,} rows of the uploaded file for this run (see sidebar to change).")
elif data_source == "Synthetic demo data":
    df = generate_synthetic_data(n_rows=n_rows, anomaly_rate=anomaly_rate)
else:
    df = None

if df is not None:
    st.subheader("Data preview")
    st.dataframe(df.head(10), width="stretch")

    has_labels, label_col = detect_label_column(df)
    all_feature_cols = [c for c in df.columns if c != label_col] if has_labels else list(df.columns)

    # Exclude columns that are monotonically increasing identifiers/time
    # counters rather than genuine behavioural features. 'Time' in the
    # Kaggle Credit Card Fraud dataset is seconds-since-first-transaction:
    # since it only ever increases, a chronological train/test split means
    # the model never sees the test set's Time range during training, so
    # every test window looks "anomalous" on Time alone -- this swamps the
    # real anomaly signal (confirmed: this caused 100% of windows to be
    # flagged, with 'Time' as the dominant explanation on every one, before
    # this exclusion was added). The original tested pipeline
    # (data/prepare_data.py) already excludes this column for the same
    # reason; this brings the dashboard in line with it.
    EXCLUDED_COLUMNS = {"time"}
    feature_cols = [c for c in all_feature_cols if c.strip().lower() not in EXCLUDED_COLUMNS]
    excluded_present = [c for c in all_feature_cols if c.strip().lower() in EXCLUDED_COLUMNS]
    if excluded_present:
        st.caption(f"Excluded from model input as a non-behavioural time/index counter: {', '.join(excluded_present)}")

    if run_button:
        if threshold_mode == "Manual" and high_risk_threshold <= review_threshold:
            st.stop()
        with st.spinner("Preprocessing..."):
            try:
                time_col = next((c for c in df.columns if c.strip().lower() == "time"), None)
                ordered = df
                if time_col is not None:
                    timestamps = pd.to_numeric(df[time_col], errors="raise")
                    if not np.isfinite(timestamps.to_numpy()).all():
                        raise ValueError("Time must contain finite values.")
                    ordered = df.iloc[np.argsort(timestamps.to_numpy(), kind="stable")]
                labels_raw = ordered[label_col].to_numpy() if has_labels else np.zeros(len(ordered))
                features, labels_raw = validate_inputs(ordered[feature_cols].to_numpy(), labels_raw)
                n_test_start = int(0.6 * len(features))
                n_val_start = int(0.9 * n_test_start)
                if min(n_val_start, n_test_start - n_val_start, len(features) - n_test_start) < window_size:
                    raise ValueError("Not enough rows for separate training, validation, and test windows.")
                scaler = fit_normal_scaler(features[:n_val_start], labels_raw[:n_val_start])
                features = scaler.transform(features)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

            train_w = normal_windows(features[:n_val_start], labels_raw[:n_val_start], window_size, window_stride)
            val_w = normal_windows(
                features[n_val_start:n_test_start], labels_raw[n_val_start:n_test_start],
                window_size, window_stride,
            )
            test_w = make_windows(features[n_test_start:], window_size, stride=window_stride)
            test_l = window_labels(labels_raw[n_test_start:], window_size, stride=window_stride)

        st.write(f"Train windows: **{train_w.shape[0]}** | Test windows: **{test_w.shape[0]}** | Test anomaly rate: **{test_l.mean():.3f}**")

        if len(train_w) < 10 or not len(val_w) or len(test_w) < 10:
            st.error("Not enough usable windows. Increase row count or decrease window size. Validation needs at least one fully normal window.")
        else:
            progress = st.progress(0.0, text="Training ReVive...")
            train_t = torch.tensor(train_w, dtype=torch.float32)
            val_t = torch.tensor(val_w, dtype=torch.float32)

            seed = 42
            torch.manual_seed(seed)
            model, history = train_repairnet(
                train_t, val_t, num_channels=train_w.shape[-1],
                hidden_dim=hidden_dim, max_epochs=max_epochs, patience=3,
            )
            progress.progress(0.5, text="Scoring test windows...")

            # Reference scores for standardization
            ref_windows = train_w[:2000] if len(train_w) > 2000 else train_w
            with torch.no_grad():
                ref_repaired = model(torch.tensor(ref_windows, dtype=torch.float32)).numpy()
            ref_scores = np.array([structural_score(ref_windows[i], ref_repaired[i]) for i in range(len(ref_windows))])

            def score_windows(windows):
                repaired_batches = []
                with torch.no_grad():
                    for start in range(0, len(windows), 256):
                        batch = torch.tensor(windows[start:start + 256], dtype=torch.float32)
                        repaired_batches.append(model(batch).numpy())
                reconstruction = np.concatenate(repaired_batches, axis=0)
                raw = np.array([structural_score(x, y) for x, y in zip(windows, reconstruction)])
                return robust_standardize(raw, ref_scores), reconstruction

            validation_scores, _ = score_windows(val_w)
            revive_scores, repaired = score_windows(test_w)
            calibrated_review, calibrated_high = np.quantile(validation_scores, [0.95, 0.99])

            progress.progress(0.75, text="Running baselines...")
            train_flat = train_w.reshape(len(train_w), -1)
            test_flat = test_w.reshape(len(test_w), -1)
            if_scores = run_isolation_forest(train_flat, test_flat)
            lof_train = train_flat[:min(3000, len(train_flat))]
            lof_scores = run_lof(lof_train, test_flat)

            progress.progress(1.0, text="Done.")
            time.sleep(0.3)
            progress.empty()

            # ---------------- Results table ----------------
            st.subheader("Held-out baseline comparison")
            if has_labels and len(np.unique(test_l)) == 2:
                results = {
                    "ReVive (ours)": {
                        "AUC-PR": average_precision_score(test_l, revive_scores),
                        "AUC-ROC": roc_auc_score(test_l, revive_scores),
                    },
                    "Isolation Forest": evaluate_baseline(if_scores, test_l),
                    "LOF": evaluate_baseline(lof_scores, test_l),
                }
                results_df = pd.DataFrame(results).T.round(4)
                st.dataframe(results_df, width="stretch")
            elif has_labels:
                st.info("The test windows contain only one class. Both classes are needed for the benchmark.")
            else:
                st.info("No label column found. Showing anomaly scores only.")

            st.caption(
                f"Seed: {seed} · Chronological split: {n_val_start:,} train / "
                f"{n_test_start - n_val_start:,} validation / {len(features) - n_test_start:,} test rows · "
                f"Window: {window_size} · Stride: {window_stride} · Hidden width: {hidden_dim} · "
                f"Epochs run: {len(history['val_loss'])}/{max_epochs}"
            )
            st.subheader("Alert threshold comparison")
            st.caption(
                f"Calibrated thresholds use the 95th and 99th percentiles of "
                f"{len(val_w):,} normal validation windows. Both policies below score the same test windows."
            )
            if len(val_w) < 100:
                st.caption("The calibration sample is small and overlapping. The upper percentile can be unstable.")
            default_thresholds = (2.0, 4.0)
            manual_thresholds = (review_threshold, high_risk_threshold)
            calibrated_thresholds = (calibrated_review, calibrated_high)
            policies = [
                ("Original defaults", default_thresholds),
                ("Validation calibrated", calibrated_thresholds),
            ]
            if manual_thresholds != default_thresholds and threshold_mode == "Manual":
                policies.append(("Manual", manual_thresholds))
            rows = []
            for name, (review, high) in policies:
                high_flags = revive_scores >= high
                flags = (revive_scores >= review) | high_flags
                row = {
                    "Policy": name, "Review threshold": review, "High threshold": high,
                    "Flagged": int(flags.sum()), "High risk": int(high_flags.sum()),
                    "Alert rate": flags.mean(),
                }
                if has_labels:
                    positives = int(test_l[flags].sum())
                    row["Precision"] = positives / flags.sum() if flags.any() else None
                    row["Recall"] = positives / test_l.sum() if test_l.any() else None
                rows.append(row)
            st.dataframe(pd.DataFrame(rows).set_index("Policy").round(4), width="stretch")
            if threshold_mode == "Validation calibrated":
                review_threshold, high_risk_threshold = calibrated_thresholds
            st.caption(f"Active policy: {threshold_mode}. Precision and recall refer to all flagged windows.")

            # ---------------- Timeline chart ----------------
            st.subheader("Anomaly score timeline")
            chart_df = pd.DataFrame({
                "window_index": np.arange(len(revive_scores)),
                "ReVive score": revive_scores,
            })
            st.line_chart(chart_df.set_index("window_index"))

            # ---------------- Explainability + triage + storage ----------------
            st.subheader("Flagged windows")
            flagged_rows = []
            db_rows = []
            for i in range(len(test_w)):
                level = classify(revive_scores[i], review_threshold, high_risk_threshold)
                if level == "Normal":
                    continue
                explanation = explain_window(test_w[i], repaired[i], feature_cols)
                top_feat = explanation["top_features"][0]
                flagged_rows.append({
                    "Window": i,
                    "Score": round(float(revive_scores[i]), 3),
                    "Triage": level,
                    "Top feature": top_feat["feature"],
                    "Contribution": round(top_feat["contribution"], 3),
                    "Actual label": "Fraud" if (has_labels and test_l[i] == 1) else ("Normal" if has_labels else "—"),
                })
                db_rows.append((i, float(revive_scores[i]), level, top_feat["feature"], float(top_feat["contribution"])))

            if flagged_rows:
                save_alerts(db_rows)
                flagged_df = pd.DataFrame(flagged_rows).sort_values("Score", ascending=False)

                total_flagged = len(flagged_df)
                MAX_DISPLAY_ROWS = 500
                if total_flagged > MAX_DISPLAY_ROWS:
                    st.warning(
                        f"{total_flagged} windows were flagged — showing the top {MAX_DISPLAY_ROWS} "
                        f"by score. All {total_flagged} are still saved to the alert history below. "
                        f"The table displays the highest-scoring alerts first."
                    )
                display_df = flagged_df.head(MAX_DISPLAY_ROWS)

                def color_triage(val):
                    color = TRIAGE_COLORS.get(val, "#FFFFFF")
                    return f"background-color: {color}; color: white"

                styler = display_df.style
                if hasattr(styler, "map"):
                    styler = styler.map(color_triage, subset=["Triage"])
                else:
                    styler = styler.applymap(color_triage, subset=["Triage"])

                st.dataframe(
                    styler,
                    width="stretch", height=350,
                )

                # Drill-down panel
                st.subheader("Explanation drill-down")
                @st.fragment
                def render_explanation():
                    selected_window = st.selectbox("Select a flagged window to inspect", display_df["Window"].tolist())
                    if selected_window is not None:
                        idx = int(selected_window)
                        explanation = explain_window(test_w[idx], repaired[idx], feature_cols)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Top contributing features:**")
                            for feat in explanation["top_features"]:
                                st.write(f"- {feat['feature']}: {feat['contribution']:.1%} of repair error")
                        with col2:
                            st.write(f"**Peak anomaly timestep in window:** {explanation['peak_timestep_in_window']}")
                            obs_df = pd.DataFrame(test_w[idx], columns=feature_cols)
                            st.line_chart(obs_df)
                render_explanation()
            else:
                st.info("No windows crossed the Review/High Risk thresholds with current settings.")

# ---------------- Alert history from DB ----------------
st.divider()
st.subheader("Stored alert history (SQLite)")
alerts_df = load_alerts()
if len(alerts_df) > 0:
    st.dataframe(alerts_df, width="stretch")
else:
    st.caption("No alerts stored yet. Run detection above to populate this table.")
