Project Journal — Aakriti Sharma
Project: ReVive — Financial Anomaly Detection (UCS503 Semester Project)
Role: Explainability & Baseline Methodology

Week 4
Focus: Running the full baseline comparison and starting the limitations analysis

Summary
With the ensemble methodology corrected, this week focused on producing a clean, fair comparison against the detection engine's results, and starting the harder, more analytical work of identifying where and why the model might fail.

What I did
●	Ran the corrected classical ensemble (Isolation Forest + LOF + COPOD, properly normalized and combined) on the same real fraud dataset and test windows used by the detection engine track, for a direct, fair comparison.
●	Compared the ensemble's results against the detection engine's verified performance and documented both sets of numbers clearly, making sure the comparison was on identical data splits and evaluation metrics.
●	Began the model limitations/failure-mode discussion — working through scenarios where the corrupt-and-repair approach could plausibly struggle: very gradual/slow-drift anomalies that don't produce a sharp reconstruction error, cold-start behavior on a brand-new pattern of normal behavior the model hasn't seen, and sensitivity to the corruption hyperparameters chosen during training.
●	Cross-checked these hypothesized limitations against the classical ensemble's own known weaknesses (e.g., Isolation Forest's difficulty with local, density-based anomalies; LOF's computational cost at scale) to frame a balanced discussion rather than only critiquing our own model.

Technical details / observations
●	Framing the limitations discussion to cover both our detection engine and the baseline ensemble (rather than only the classical methods) was a deliberate choice — a one-sided limitations section would read as less credible in a viva.

Issues / blockers
●	None blocking — this week's work was primarily comparative analysis and structured reasoning rather than new implementation, so no technical blockers came up.

Learnings
●	A fair 'state of the art' comparison needs identical data splits and metrics across every method being compared — otherwise even a correct individual result can be an unfair comparison.

Time spent (approx.): ~7 hours (running the full comparison, documenting results, drafting the limitations discussion).
Plan for next week: Finalize the limitations/failure-mode write-up and hand off both the baseline results and the limitations section for inclusion in the documentation.
