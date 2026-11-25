Week 4
Focus: Debugging the pipeline — finding and fixing the issues behind the weak result

Summary
This was the core debugging week. Found two real, separate issues that were both quietly suppressing the model's real performance, fixed them one at a time, and watched the score climb substantially as each fix landed.

What I did
●	Confirmed the first suspected bug: the inference script was still applying the training-time corruption function to test windows before scoring them, instead of feeding the model clean, unmodified data — meaning the model was being evaluated on artificially noisy input the entire time.
●	Fixed this by disabling corruption entirely in the inference path, keeping it strictly confined to the training loop as originally intended.
●	Re-ran evaluation after this fix — AUC-PR improved substantially, confirming this was a major contributor to the earlier weak result.
●	Investigated the second suspected issue: how window-level labels were being assigned. The original logic labeled a window as 'anomalous' based on any fraud point appearing anywhere inside a 100-step window, which diluted the true anomaly signal across mostly-normal windows.
●	Fixed the labeling logic to align window-level labels more precisely with where the anomalous point actually falls, and correspondingly tightened the scoring/threshold alignment against the true event location.
●	Re-ran the full pipeline after both fixes and obtained a verified AUC-PR of 0.98, a large and repeatable improvement over the original 0.36.
●	Re-ran the corrected pipeline multiple times with different random seeds to confirm the 0.98 result was stable and not a one-off fluke.

Technical details / observations
●	The corruption-at-inference bug was the larger contributor of the two — it meant every test window had noise injected into it before scoring, which directly degraded the model's ability to cleanly distinguish normal from anomalous behavior.
●	The label-alignment fix mattered less in raw magnitude but improved the precision of what 'correct detection' meant, tightening the precision-recall curve used for AUC-PR.
●	Verified stability across seeds specifically because a jump this large warranted extra scrutiny before treating it as final — repeated runs stayed consistently high, which gave confidence it wasn't a fluke.

Issues / blockers
●	Both bugs were silent — neither caused a crash or an error message, which is why they required deliberate, structured investigation rather than being caught automatically.

Learnings
●	Always double check that any data-augmentation/corruption logic is strictly scoped to training and cannot leak into the inference path — this class of bug produces a plausible-but-wrong number rather than an obvious failure.
●	Window-level label definitions matter a lot for evaluation correctness in windowed time-series anomaly detection — a loose labeling rule can silently blur the evaluation signal.

Time spent (approx.): ~11 hours (bug investigation, two fixes, repeated verification runs across seeds).
Plan for next week: Finalize and document the verified results, and hand off the detection engine for integration with the ensemble/explainability track and the dashboard.
