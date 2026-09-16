Week 2
Focus: Reimplementing the architecture from scratch and verifying it against the paper

Summary
With the spec fully extracted, this week was pure implementation — building the detection engine's architecture in code and proving it matches the paper exactly before trusting it with any training.

What I did
●	Implemented the model architecture from scratch: input projection, a single residual block (depthwise conv + pointwise conv + activation), and a zero-initialized output projection with a global residual/skip connection.
●	Wrote a parameter-counting check and compared it directly against the paper's reported figure to confirm the reimplementation was structurally correct, not just approximately similar.
●	Implemented the inference-time scoring function (the four-term formula combining amplitude, temporal-difference, trend, and correlation error) exactly as specified, including the robust standardization step.
●	Ran manual sanity tests: identical windows should score near zero, an artificially injected spike should score clearly higher.

Technical details / observations
●	Parameter count matched the paper's reported figure exactly, which was the strongest available signal that the architecture was reproduced correctly with no reference code to check against.
●	The zero-init output layer means the model is mathematically an identity function before any training — confirmed this behavior explicitly rather than assuming it from the architecture description alone.

Issues / blockers
●	None blocking — the architecture matched the spec on the first fully-written attempt.

Learnings
●	Matching a paper's exact reported parameter count is a much stronger correctness check than 'the code runs without crashing.'

Time spent (approx.): ~7 hours (architecture implementation, parameter verification, scoring function, sanity tests).
Plan for next week: Build the full training pipeline (corruption procedure, loss, optimizer) and get a first end-to-end result on real fraud data.
