Project Journal — Mayur goyal
Project: ReVive — Financial Anomaly Detection (UCS503 Semester Project)

Week 4
Focus: Debugging the Streamlit app — label detection, Styler, and speed fixes

Summary
This week focused entirely on turning the rough first version of the demo app into something reliable and presentation-ready, fixing three real, separate bugs found during last week's test run.

What I did
●	Fixed the label-detection bug: the app was assuming a fixed column name/position for the fraud label, which broke on differently-formatted uploads; replaced this with more robust detection that checks column names and value patterns (e.g., a binary 0/1 column) rather than assuming a fixed schema.
●	Fixed the pandas Styler rendering issue: the highlighted-row styling wasn't consistently displaying in Streamlit's table component across different pandas/Streamlit version combinations; resolved by switching to a more reliable styling approach compatible with the deployed environment.
●	Fixed the app's slowness on larger uploads: profiled the app and found it was reloading and re-initializing the detection model on every interaction instead of caching it; added proper caching so the model loads once and is reused, and batched the scoring calls instead of scoring row-by-row.
●	Re-tested the app end-to-end after each fix individually, then did a final full run-through with a realistic-sized upload to confirm all three issues were genuinely resolved together, not just individually.

Technical details / observations
●	The speed fix (model caching + batched scoring) made the single biggest practical difference to how usable the demo felt — the app went from a noticeable multi-second delay per interaction to feeling responsive.
●	The label-detection fix also made the app meaningfully more robust for the live demo, since it no longer depends on the uploaded file matching one exact expected format.

Issues / blockers
●	None outstanding after this week's fixes — all three identified bugs were resolved and re-verified.

Learnings
●	Reloading a model on every UI interaction is an easy performance mistake in Streamlit apps specifically — caching the expensive setup step is usually the single highest-value fix for responsiveness.

Time spent (approx.): ~9 hours (diagnosing and fixing three separate bugs, re-testing after each fix, final full run-through).
Plan for next week: Assemble the full documentation, proposal, and presentation slides now that the diagrams and demo app are both complete.
