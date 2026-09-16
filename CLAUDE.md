# Repository Guidance

## Product

One product workflow: a generic, persistent active-learning dashboard for tabular data.

DatasetAdapter → preprocessing → MDS / SSDBCODI → six-score review matrix
→ explicit human LabelEvents → next saved round.

Wine is a fixture only. Never encode wine-specific assumptions in reusable services.

## Boundaries

- data_workspace owns common dataset and matrix contracts.
- projection owns display coordinates; distances for scores are never taken from the plot.
- ssdbcodi owns clustering, outlier results and recorded expansion paths.
- algorithm_adapters exposes the shared AnalysisResult boundary.
- selection and labeling own their module-lab state.
- scatterplot renders model truth.
- rule_panel generates explanation-only decision-tree rules, not review priorities.
- active_learning owns imports, sessions, rounds, six scores, eligibility and label history.
- workflows orchestrate modules; modules must not import workflows.

There is no LLM/DeepSeek integration, meta-priority score or legacy eight-category plan.
Do not restore these paths. See docs/modules/active_learning/design.md for the current specification.

## Invariants

Keep scoring deterministic and network-free. Keep all six scores separate.
Freeze eligible percentile pools within a round; filters must not recompute them.
Use midrank ties, stable point-ID ordering, explicit NA reasons and duplicate-free batches.
Keep raw data separate from model transforms. Never expose evaluation truth through analysis,
score evidence, metadata shown in the plot, or model references.
Human semantic classes and confirmed normal/outlier status are separate dimensions.
Unsure does not become a confirmed reference.
Keep cluster colors separate from review tile colors and selection outlines.
Store full round/label transitions atomically; never mutate a round from GET.

## Commands

~~~sh
python run.py
python -m unittest discover -s tests
python -m unittest discover -s tests/browser -v
python -m compileall app tests
git diff --check
~~~

Browser tests require optional Playwright and Chromium; normal runtime does not.
The product URL is http://127.0.0.1:5001/workflows/active-learning-dashboard/.

JSON APIs use {"ok": true, "data": {}, "error": null, "diagnostics": {}}.
Add tests for scoring, round transitions and changed API contracts. Update current docs.
Do not commit .env, runtime_data, generated caches, SQLite files or credentials.
Only METRIC_DASHBOARD_ACTIVE_LEARNING_DB_PATH is needed in .env.example.
