# ReVive — Streamlit Demo

A local, clickable demo of the full ReVive pipeline: data → train →
score → explain → triage → store → visualize. Built to show a working
prototype to the professor, running entirely on localhost.

## What this demonstrates

- **ReVive model** (the corrupt-and-repair reimplementation) trains
  live on either synthetic demo data or an uploaded CSV.
- **Held-out baseline comparison**: ReVive's AUC-PR/AUC-ROC vs.
  Isolation Forest and LOF baselines, computed on the same data.
- **Explainability**: per-feature attribution for every flagged
  window (this project's novel addition beyond the original paper).
- **Triage**: Normal / Review Needed / High Risk tiers with validation-calibrated
  thresholds by default and manual thresholds available in the sidebar.
- **Storage**: every flagged window is written to a local SQLite
  database (`alerts.db`), and the full alert history is shown at
  the bottom of the page.
- **Dashboard**: timeline chart of anomaly scores, a sortable flagged-
  windows table, and a drill-down panel per selected anomaly.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

This opens automatically at **http://localhost:8501**.

## Demo walkthrough (for showing the professor)

1. Leave "Synthetic demo data" selected in the sidebar (no download
   needed — it generates a realistic-looking time series with
   injected spikes, trend shifts, and noise anomalies).
2. Click **Train & Score**.
3. Watch the progress bar: preprocessing → training (you'll see it's
   a real, live training run, not a canned result) → scoring →
   baselines.
4. Review the **held-out baseline comparison table** — ReVive vs.
   Isolation Forest vs. LOF, computed live on this run's data.
5. Scroll to the **timeline chart** — anomaly score per window.
6. Check the **flagged windows table** — color-coded by triage level.
7. Pick a window in the **drill-down** selector — see which features
   drove that specific flag, and the feature values across the
   window.
8. Scroll to the bottom — the **stored alert history** table shows
   everything just written to the local SQLite database, proving
   the storage layer works, not just the in-memory display.

## Using real data instead

Switch "Data source" to **Upload CSV** in the sidebar. Any CSV with
numeric feature columns works; include an optional `label` column
(1 = anomaly, 0 = normal) to also see accuracy metrics. The Kaggle
Credit Card Fraud dataset (`creditcard.csv`) works directly, though
note the full dataset (~285k rows) will take longer to train live in
the UI than the synthetic demo data — for a live demo, consider
uploading a contiguous chronological slice of a few thousand rows instead.

## Evaluation setup

The Streamlit app sorts the selected rows by Time when present and splits the
rows before extracting any windows. The app uses the earliest 54% for training,
the next 6% for early stopping, and the final 40% for evaluation. The scaler
fits normal training rows only. Training and validation accept contiguous
windows with no known anomalies. Without labels, these partitions are assumed
normal and the test benchmark is omitted. The window-level benchmark requires
both classes in the test windows and reports average precision and ROC-AUC.

The app fixes the PyTorch training seed at 42 and displays it with the split,
window length, stride, hidden width and epochs. Repeatability is checked within
the same Python and dependency environment.

**Alert policy.** ReVive uses the training-score median and IQR to standardize
scores. The calibrated Review Needed and High Risk cutoffs are the 95th and
99th percentiles of scores from normal validation windows. This is the same
validation partition used for early stopping. The test partition supplies the
alert counts, flag rate, precision, and recall for both the original 2.0/4.0
policy and the calibrated policy. Precision and recall describe all flagged
windows. If there are no flags or no positives, the corresponding metric is
undefined. With an unlabeled CSV, calibration assumes the validation data is
normal and the app omits precision and recall. Overlap between windows and
distribution shifts can make the observed test alert rate differ from the
validation percentile.

The separate `data/prepare_data.py` helper keeps its five return values and
uses a chronological 60/40 train/test split by default. See [verification and benchmark results](VERIFICATION.md) for the recorded run.

## What this does NOT yet include (see project documentation, Section 9)

- The REST API layer (this app calls the pipeline directly, in-process,
  rather than through a separate backend service)
- The corrected multi-baseline ensemble (Isolation Forest + LOF +
  COPOD combined) — currently IF and LOF are shown as separate,
  individual baselines only
- Production deployment considerations (streaming ingestion, rolling
  windows, retraining) — this is a course-prototype demo, not a
  deployed system

## Hosting

See [deployment instructions](DEPLOYMENT.md) for a public Streamlit link or a private Docker deployment with persistent SQLite.
