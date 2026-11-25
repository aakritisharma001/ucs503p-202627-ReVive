Project Journal — Aakriti Sharma
Project: ReVive — Financial Anomaly Detection (UCS503 Semester Project)
Role: Explainability & Baseline Methodology

Week 1
Focus: Working through the course's 'engine availability' constraint and what it means for our approach

Summary
This week was about resolving a real tension: the course explicitly disallows open-ended research and requires an already-available engine, but our chosen base method (the JuRe paper) had no released code. Worked through what would and wouldn't be defensible under the course's rules before the team committed further time to the reimplementation.

What I did
●	Reviewed the course's semester-project page and its stated philosophy in detail — specifically the 'no research delays' rule and the requirement for a 'mature engine' already available for the core solution.
●	Discussed with the team whether reimplementing a fully-specified paper (architecture, training recipe, and scoring formula all given, just no code) counts as 'engine availability' under this rule, versus counting as disallowed research.
●	Concluded that since every design decision was already made by the original authors and nothing being done was novel algorithm research, the reimplementation is defensible as 'integrating an available (if not pre-coded) engine' — the position to take if this comes up with the professor.
●	Started scoping a fallback/complementary angle: a classical, no-training-required ensemble (Isolation Forest, LOF, COPOD) that could stand on its own as an unambiguously 'available' engine if needed, and would also double as a rigorous baseline either way.

Technical details / observations
●	This framing question mattered enough to resolve early, since it affects how the whole project — not just one section of the report — should be described to the professor.

Issues / blockers
●	No official ruling from the professor on this specific framing yet — proceeding on the team's best-judgment interpretation, with a documented rationale ready in case it's questioned.

Learnings
●	Course-level constraints (like 'no research delays') need to be actively interpreted and applied to a specific technical choice, not just read once and assumed to be fine — worth resolving explicitly rather than leaving it ambiguous.

Time spent (approx.): ~5 hours (reading course materials, team discussion, framing writeup, initial baseline scoping).
Plan for next week: Prototype the classical ensemble baseline (Isolation Forest, LOF, COPOD) to have as both a course-safe fallback and a rigorous comparison point.
