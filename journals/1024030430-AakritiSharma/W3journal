Project Journal — Aakriti Sharma
Project: ReVive — Financial Anomaly Detection (UCS503 Semester Project)
Role: Explainability & Baseline Methodology

Week 3
Focus: Catching and fixing a real methodology flaw: naive score averaging

Summary
This week's close look at the ensemble's combination logic paid off — found a genuine methodology flaw (naively averaging three differently-scaled anomaly scores) that was skewing the ensemble's results toward whichever method happened to produce the largest raw numbers, and fixed it properly.

What I did
●	Examined the raw score distributions of Isolation Forest, LOF, and COPOD individually and found they operate on meaningfully different scales and ranges, meaning a simple average implicitly let whichever method had the largest numeric range dominate the combined score.
●	Confirmed this was a real problem, not a theoretical concern, by checking cases where the naive-averaged ensemble disagreed with what the individually strongest method would have flagged — the averaging was measurably diluting genuine anomaly signal from the better-calibrated methods.
●	Fixed this by standardizing each method's scores (converting to a common scale, e.g. rank- or z-score-based normalization) before combining them, so each method contributes proportionally rather than being dominated by raw scale differences.
●	Re-ran the ensemble with the corrected combination logic and confirmed the combined score now behaved sensibly — agreeing more closely with the strongest individual method's flags rather than being pulled off by scale artifacts.
●	Documented the flaw and the fix clearly, since it's a good, concrete example of methodological rigor to bring up in the documentation and viva.

Technical details / observations
●	This is a classic and easy-to-miss pitfall in any ensemble that combines heterogeneous anomaly scores — worth catching specifically because it would have quietly under-mined the fairness of the baseline comparison against the detection engine if left in.

Issues / blockers
●	None outstanding after the fix — but this did cost meaningful time to properly diagnose, since the ensemble wasn't crashing or obviously broken, only subtly miscalibrated.

Learnings
●	Combining scores from different anomaly detection methods always needs an explicit normalization step first — never average raw, differently-scaled scores and assume the result is meaningful.

Time spent (approx.): ~9 hours (diagnosing the scaling issue, implementing normalization, re-running and verifying the corrected ensemble).
Plan for next week: Run the corrected ensemble against the detection engine's real results for a full comparison, and begin the model limitations/failure-mode analysis.
