Project Journal — Mayur goyal
Project: ReVive — Financial Anomaly Detection (UCS503 Semester Project)

Week 2
Focus: Building all five UML diagrams

Summary
This week produced the full required diagram set — Use Case, Activity, Class, DFD, and Sequence diagrams — covering the system from five complementary perspectives, cross-checked against how the detection engine and baseline tracks were shaping the actual architecture.

What I did
●	Built the Use Case Diagram covering both actors and all identified use cases.
●	Built the Activity Diagram for the core detection flow: upload → preprocess → score → threshold decision → alert/log.
●	Built the Class Diagram identifying the system's core components (data ingestion, preprocessing, the detection engine, the scoring/ensemble logic, explainability module, triage logic, alert storage, API layer) and their relationships.
●	Built the Data Flow Diagram (Level 0 and Level 1) showing how data moves between ingestion, detection, explanation, storage, and the dashboard.
●	Built the Sequence Diagram for the scenario 'user uploads a file and views a flagged anomaly,' including a branch for the normal vs. flagged-anomaly outcomes.
●	Rendered all five diagrams using proper diagramming tools (Graphviz/PlantUML) for clean, standard UML notation rather than freehand drawing.

Technical details / observations
●	Cross-checked each diagram against the detection engine track's evolving pipeline and the baseline/ensemble track's comparison logic to keep the diagrams accurate to the real system being built, not an idealized version of it.

Issues / blockers
●	None blocking this week.

Learnings
●	Building all five diagrams together, cross-referenced against both engineering tracks in the same week, kept them mutually consistent with each other rather than drifting out of sync if built one at a time over a longer period.

Time spent (approx.): ~9 hours (all five diagrams, cross-checking against both engineering tracks, rendering with Graphviz/PlantUML).
Plan for next week: Start building the Streamlit demo app so the project has a working, presentable interface ahead of the prototype evaluation.
