Project Journal — Mayur goyal
Project: ReVive — Financial Anomaly Detection (UCS503 Semester Project)

Week 3
Focus: Building the initial Streamlit demo app

Summary
This week started the dashboard/demo track proper — a working Streamlit app that lets a user upload transaction data and see anomaly flags, wired up to the detection engine and baseline outputs being produced by the other two tracks.

What I did
●	Set up the Streamlit app skeleton: a file upload widget, a results table, and a basic summary view showing how many transactions were flagged.
●	Wired the app up to call the detection engine track's scoring function and display per-transaction anomaly scores in the results table.
●	Added a basic results table styled to highlight flagged (high-score) transactions for readability.
●	Ran the app end-to-end for the first time against a sample of the fraud dataset and confirmed the upload → score → display flow worked as a rough first pass, though several rough edges were noted for follow-up.

Technical details / observations
●	Kept the first version deliberately minimal (upload, score, display) rather than trying to build every planned feature (explainability breakdown, export, thresholds) at once, to get a working end-to-end demo as early as possible.

Issues / blockers
●	Noted several rough edges during the first test run to fix the following week: the app wasn't reliably detecting which column in an uploaded file was the fraud/label column, the results table's styling (using pandas Styler) wasn't rendering correctly in all cases, and the app felt noticeably slow when scoring larger uploads.

Learnings
●	Getting a rough end-to-end version working first, then fixing specific bugs against it, was more productive than trying to polish each piece in isolation before connecting them.

Time spent (approx.): ~8 hours (Streamlit app skeleton, wiring to the detection engine, first end-to-end test run).
Plan for next week: Debug and fix the issues found this week — label detection, Styler rendering, and app speed — to get the demo presentation-ready.
