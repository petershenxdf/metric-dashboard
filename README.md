# Metric Dashboard

A local Flask dashboard for six-score, multi-round human review of structured tabular data.
CSV, JSON and MAT imports share preprocessing, SSDBCODI, MDS, explanation-only decision trees and SQLite history.

## Current workflow

Import → SSDBCODI → six independent review scores → linked matrix / scatterplot → explicit human labels → next saved round.

The six categories are Cluster Assignment Conflict, Label Coverage Gap, Weak Group Reachability,
Local Sparsity, Known-Outlier Similarity and Model Instability.
The old eight-category plans, history-weighted meta-ranking, evidence checklists, DeepSeek client,
prompt and interpretation endpoint have been removed. No API key or external service is required.

Scores use model-space distances, not 2D coordinates. Midrank percentiles are frozen for each round.
They are neither probabilities nor a cross-category utility score. Missing scores display patterned NA.
See [the scoring design](docs/modules/active_learning/design.md) and [API contracts](docs/state_and_api_contracts.md).

## Run

Use Python 3.9–3.11 with the current dependency ranges:

~~~sh
python -m pip install -r requirements.txt
python run.py
~~~

Open http://127.0.0.1:5001/workflows/active-learning-dashboard/.
HOST and PORT can override the server defaults. Retained engineering labs live under /modules/.

### Windows / PowerShell

Use a project-local environment to avoid mixing Conda numerical-library DLLs:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
~~~

On the session index, choose **Load Wine Dataset**; no upload is needed. The bundled
wine.mat contains 129 records with 13 features. Initially, Coverage, Reachability and
Known-Outlier Similarity may be NA because they require explicit human references.
This does not indicate a failed dataset load.

If an existing Conda environment terminates during NumPy matrix operations, use the
project-local environment above, or launch with the environment properly activated.
Changing dataset scores cannot fix a native numerical-library loading failure.

For an existing Conda environment, use `conda run` so its numerical-library DLLs
are resolved from that environment (rather than invoking its python.exe alone):

~~~powershell
conda run -n dev --no-capture-output python run.py
~~~

The only application-specific environment setting is optional:

~~~text
METRIC_DASHBOARD_ACTIVE_LEARNING_DB_PATH=runtime_data/active_learning/active_learning.sqlite3
~~~

Never commit runtime_data, SQLite files or .env. Ground-truth columns are evaluation-only and
cannot overlap feature, metadata or point-ID roles. Wine is only a demo/regression fixture.

## Review

- Choose a category to sort descending; reverse order with its header or direction button.
- Filter by review status, model group, point ID, numeric score conditions or NA.
- Match ALL / Match ANY combines score conditions; other filters always restrict the result.
- Recommendable-only applies the category's actual gates, not merely a high percentile.
- Click a tile, point-ID button or plot point to inspect evidence and comparison records.
- Check a duplicate-free recommended batch, or label manually selected records.
- Submit semantic type, Normal, Outlier or Unsure. Unsure defers for one round, never becoming a reference.
- Review and revert round history. Old rounds offer an explicit upgrade button; loading them does not rerun the model.

Default MinPts is 3, recommendation percentile threshold 75, batch size 4, optional label budget,
and an exact-analysis limit of 2,000 records. MinPts must be smaller than the record count.
Instability runs the full pipeline at every valid distinct setting in MinPts−1, MinPts, MinPts+1.
This adds compute cost. Empty queues are not filled with unrelated points.

## Validation

~~~sh
python -m unittest discover -s tests
python -m compileall app tests
git diff --check
~~~

Optional real-browser checks (test dependencies only):

~~~sh
python -m pip install playwright
python -m playwright install --with-deps chromium
python -m unittest discover -s tests/browser -v
~~~

These check exact color boundaries, ties and NA ordering, ALL/ANY filters, keyboard and plot/table
linking, selected-point visibility, label refresh, request failures and responsive matrix scrolling.

## Code map

- app/modules/active_learning/review.py: six raw scores, fixed percentile pools, recommendation gates.
- app/modules/active_learning/service.py: label feedback, complete reruns, frozen round snapshots.
- app/modules/ssdbcodi/algorithm.py: model calculations and recorded full-path expansion.
- app/modules/rule_panel/: explanation-only tree rules.
- app/workflows/active_learning_dashboard.py: HTTP workflow.
- app/static/review_matrix.js and review_matrix.css: linked matrix and point review.

The implementation follows Dashboard_Active_Learning_LaTeX.zip supplied for this change.
Existing category-guide PDFs and literature_grounded_category_scores_report.tex predate this
implementation and are retained as historical documents, not current specifications.
Equal-budget policy benchmarking and a human usability study remain future evaluation work;
unit and browser tests do not establish improved labeling efficiency.
