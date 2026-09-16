Project Journal — Aakriti Sharma
Project: ReVive — Financial Anomaly Detection (UCS503 Semester Project)
Role: Explainability & Baseline Methodology

Week 2
Focus: Prototyping the classical Isolation Forest + LOF + COPOD ensemble baseline

Summary
This week built out the classical ensemble that would serve as our comparison baseline against the detection engine — three well-established, no-training-required anomaly detection methods combined into a single ensemble score.

What I did
●	Implemented Isolation Forest, Local Outlier Factor (LOF), and COPOD as three independent anomaly detectors, each run on the same windowed fraud data format used by the detection engine track.
●	Built a first version of the ensemble that combined all three methods' anomaly scores into a single combined score, to serve as a stronger, fairer baseline than any single classical method alone.
●	Ran the ensemble on a subset of the fraud dataset to confirm all three methods executed correctly and produced sensible individual outputs before combining them.
●	Did an initial combined-score run and got a first ensemble result — usable as a starting comparison point, though flagged for review since the combination logic hadn't been closely scrutinized yet.

Technical details / observations
●	Chose these three specific classical methods because they represent meaningfully different detection philosophies (isolation-based, density-based, and copula/statistical-based) — a more informative comparison than three variants of a similar idea.

Issues / blockers
●	The initial score combination was a simple average of the three methods' raw outputs — flagged as something to scrutinize further, since the three methods don't naturally produce scores on the same scale.

Learnings
●	Combining scores from methods with fundamentally different underlying score distributions needs more care than a naive average — this became the focus of the following week's work.

Time spent (approx.): ~8 hours (implementing three classical detectors, initial ensemble combination, first test run).
Plan for next week: Scrutinize the ensemble's score-combination logic more carefully — this was flagged as a risk area at the end of this week.
