# ReVive benchmark and verification

The Streamlit benchmark used the first 8,000 rows of the ULB credit-card CSV, sorted by Time. Seed 42 was used with window size 50, stride 5, hidden width 64, and a maximum of 10 epochs. Training stopped after six epochs. The chronological split was 4,320 training rows, 480 validation rows, and 3,200 test rows. There were 835 normal training windows, 87 normal validation windows, and 631 test windows. Of those test windows, 135 contained at least one labeled fraud transaction.

| Detector | Average precision | ROC-AUC |
|---|---:|---:|
| ReVive | 0.9681 | 0.9894 |
| Isolation Forest | 0.3249 | 0.6646 |
| LOF | 0.4914 | 0.8161 |

Average precision is labeled AUC-PR in the dashboard.

| Alert policy | Review / High Risk thresholds | Flagged | High Risk | Alert rate | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Original defaults | 2.0000 / 4.0000 | 626 | 564 | 99.21% | 21.57% | 100% |
| Validation calibrated | 4.7970 / 5.0895 | 338 | 290 | 53.57% | 39.94% | 100% |

Both rows use the same seeded model and the same test scores. Calibration reduced alerts by 288 while retaining all 135 positive test windows on this subset. The 53.57% alert rate still represents substantial reviewer workload. The calibration sample was small, with windows overlapping by 45 rows, and the validation distribution may differ from the test distribution. These figures describe this subset and thresholds.

## Software Engineering verification

| Test | Input or action | Expected behavior | Observed |
|---|---|---|---|
| CSV upload | Upload 8,000 labeled transactions and run Train & Score | Complete scoring and display the benchmark | Passed. All three detectors reported results. |
| Invalid input | Upload a CSV with a text value in a feature column | Show a validation error | Passed. The app displayed the numeric feature error. |
| Triage | Score the same holdout under both threshold policies | Show the flagged counts, precision and recall | Passed. The default and calibrated policies flagged 626 and 338 windows respectively. |
| Explanation | Render the explanation for a flagged window | Show contributing features and a peak timestep | Passed. The explanation panel rendered for a flagged window. |
| Persistence | Open a fresh app session using the same SQLite file | Show the stored alerts | Passed. SQLite and the reopened alert history both contained 338 alerts for the run. |
| Repeatability | Repeat the CSV run with identical settings | Match the displayed benchmark and alert policy tables | Passed. Both tables matched at their displayed precision. |

Verified using CPU, Python 3.12.14, PyTorch 2.15.0+cpu, Streamlit 1.67.0 and scikit-learn 1.8.0. The header plus the first 8,000 data rows of the tested CSV had SHA-256 `740bb83e9dc52d971af6b9d8cba4df1bf87aaf4c94567b72f8b82f819216f2d5`.
