# Integration testing

## Standard suite

~~~sh
python -m unittest discover -s tests
python -m compileall app tests
git diff --check
~~~

The suite covers import/preprocessing and role isolation, retained module labs, SSDBCODI,
expansion-tree bottlenecks (including zero-distance edges), six score formulas, exact midrank ties,
coverage support quantiles, self/reference exclusions, known-outlier local gates, member-set
instability under renamed/split/merged/outlier outputs, incomplete runs, eligibility and duplicate-free
batches. Service tests cover complete fixed-input reruns, atomic failures, corrections, stale
submissions, budgets, empty queues, restart, history/revert and explicit legacy upgrade.

## Optional real browser suite

~~~sh
python -m pip install playwright
python -m playwright install --with-deps chromium
python -m unittest discover -s tests/browser -v
~~~

A local temporary Flask server and temporary SQLite database are created per test.
Checks include ALL/ANY/NA semantics, stable unrounded sorting and color bands, unchanged scores
after filters, keyboard/plot/table selection, hidden-selection notice, label refresh with retained
point ID, request errors and responsive horizontal scrolling. No production dataset is touched.

Feedback regressions also cover propagation of Normal support without assigning a semantic type,
replacement of rejected bootstrap anchors, fewer-than-K candidates, no automatic candidates,
all-confirmed-outlier results, and correction/revert of those results. Browser checks cover explicit
single/batch mode, cancellation/confirmation, plot batch rings, NA/Reviewed evidence, zero-instability
wording, malformed saved views and full filter restoration across submission and reload.

To use an already-installed Chrome instead of Playwright's downloaded Chromium, set
METRIC_TEST_BROWSER_PATH to its full executable path before running the browser suite.
On Windows use the activated Conda environment described in README; direct invocation without
activation may fail to load the numerical library DLLs. Test databases and screenshots are temporary.

No test result establishes improved labeling efficiency. Equal-label-budget comparisons with random
review and human usability studies from the design document remain separate evaluation tasks.
