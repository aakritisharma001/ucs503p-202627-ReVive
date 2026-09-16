Week 5
Focus: Finalizing verified results and handing off the detection engine

Summary
With the pipeline debugged and stable, this week was about locking in the final, verified numbers, documenting exactly what was fixed and why, and handing off a clean version of the detection engine to the rest of the team.

What I did
●	Ran the final, fully verified pipeline end-to-end on the real fraud dataset and recorded the official result: AUC-PR 0.98.
●	Wrote up a clear before/after account of the debugging process (corruption-leak fix and label-alignment fix) for the documentation, including the intermediate 0.36 result as an honest record of the debugging journey rather than hiding it.
●	Cleaned up the codebase (removed dead code from earlier debugging attempts, added inline comments explaining why corruption is disabled at inference and how window labels are assigned) so the engine is presentable and explainable in a viva.
●	Handed off the finalized detection engine and its outputs to the explainability/ensemble track for direct comparison against the classical baseline ensemble.
●	Supported the dashboard track by confirming the exact input/output format the engine expects and returns, so the Streamlit app could be wired up correctly.

Technical details / observations
●	Deliberately kept the 0.36 → 0.98 debugging story in the documentation rather than only reporting the final number, since it demonstrates genuine engineering rigor and is a strong, honest story for a viva question about the model's development process.

Issues / blockers
●	None outstanding — the engine is stable, verified across multiple runs, and its interface is documented for the other two tracks to build against.

Learnings
●	Documenting a debugging journey (not just the final result) turns what could look like an unexplained improvement into a credible, defensible engineering story.

Time spent (approx.): ~6 hours (final verified run, documentation write-up, code cleanup, handoff coordination).
Plan for next week: Support integration with the ensemble comparison and dashboard, and prepare to explain the debugging process and final architecture clearly for the viva.
