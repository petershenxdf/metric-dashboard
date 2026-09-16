# Product workflow

The only product workflow is active-learning-dashboard under /workflows/active-learning-dashboard/.
The index imports CSV/JSON/MAT, creates sessions, offers the Wine fixture and resumes saved work.

The session page contains model projection, linked point evidence and explicit labeling controls,
a six-column review matrix, explanation-only tree rules and round history.
Default sorting is Model Instability descending; default visibility is unlabeled records.
Select a category header to sort and a score tile to inspect that reason. ALL/ANY conditions,
NA filters, recommendability gates and batch selection work on the round's frozen scores.

Old sessions display preserved model results and an explicit six-score upgrade button.
No DeepSeek control or interpretation endpoint remains.
Module labs remain independent debugging surfaces, not separate product workflows.
