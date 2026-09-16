Week 3
Focus: Building the training pipeline and getting a first (weak) result on real data

Summary
This week the training pipeline came together end-to-end and produced a first real result on the fraud dataset — but the result was weak (AUC-PR 0.36), which kicked off a debugging effort that carried into the following weeks.

What I did
●	Implemented the corruption procedure used during training (noise injection plus random channel masking) and the combined training loss (reconstruction error plus a temporal-difference term).
●	Built the full training loop and trained the model on the real fraud dataset for the first time.
●	Ran the trained model against the held-out test set and computed AUC-PR as the primary evaluation metric, given how imbalanced the fraud data is.
●	Got a first end-to-end result: AUC-PR 0.36 — a working pipeline, but noticeably weaker than expected given how cleanly the architecture had verified against the paper.
●	Started a structured debugging pass: re-checked the training/test split, the windowing logic, and whether test-time inference was genuinely running under the same conditions the paper describes (i.e., no corruption applied at test time).

Technical details / observations
●	A first-pass 0.36 AUC-PR is a legitimate, working result — not a crash or an obviously broken pipeline — which made this a genuine debugging problem rather than an obvious bug to spot immediately.
●	Flagged three suspects going into the debugging week: possible leftover corruption being applied at inference time by mistake, possible window-level label misalignment (a window being labeled 'anomalous' even when the fraud point sits at the very edge of the window), and possible train/test window overlap diluting the evaluation.

Issues / blockers
●	The weak first result meant the pipeline could not yet be trusted for final reporting — needed a dedicated debugging pass before treating any number as final.

Learnings
●	A pipeline that runs cleanly and produces a plausible-looking number can still be quietly wrong — a weak-but-plausible result needs the same scrutiny as a crash.

Time spent (approx.): ~9 hours (training pipeline, first full training run, initial evaluation, early debugging investigation).
Plan for next week: Debug the pipeline systematically to find and fix whatever is suppressing the real detection performance.
