Week 1
Focus: Finding the JuRe paper and discovering no official code was released

Summary
This week was about locking down our base method and confirming what we could actually build on. Found the JuRe paper as a strong candidate for our anomaly detection engine, but discovered its official code repository was empty except for a README — meaning the engine would have to be reimplemented from the paper's spec, not adapted from existing code.

What I did
●	Searched Papers with Code / the paper's own GitHub link for the JuRe (Just Repair) time-series anomaly detection paper and reviewed it as a candidate detection engine.
●	Cloned the official repository (github.com/iis-esslingen/JuRe) and inspected its contents — found only a single README.md, with the authors stating the paper is under review and code release is 'coming soon.'
●	Re-read the paper closely to extract every implementation detail needed to reimplement it independently: the residual-block architecture, the corrupt-and-repair training recipe, the loss function, and the four-term inference scoring formula.
●	Documented the full spec (layer-by-layer architecture, corruption parameters, optimizer settings, scoring weights) as a reference for the build in the coming weeks.

Technical details / observations
●	The architecture is intentionally minimal — one residual block, ~17,665 parameters at channel width 1 — which made 'reimplement from the paper' a realistic option rather than an open-ended research task.
●	Confirmed with the team that using a fully-specified paper (even without released code) still counts as an 'available engine' for the course's constraints, since every design decision needed to rebuild it is already made by the original authors — we're implementing, not inventing.

Issues / blockers
●	No reference code meant zero ability to debug-by-comparison against a working implementation — any bugs in the coming weeks would have to be caught purely against the paper's formulas and our own tests.

Learnings
●	A GitHub link on a paper's listing doesn't guarantee usable code — checking this early avoided losing a week later on a dead end.

Time spent (approx.): ~5 hours (paper research, repo inspection, spec extraction).
Plan for next week: Begin implementing the model architecture from the extracted spec and verify it against the paper's reported parameter count.
