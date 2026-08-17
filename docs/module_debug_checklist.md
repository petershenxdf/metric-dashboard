# Module Debug Checklist

Use module pages to verify a component in isolation. The final product remains the active-learning dashboard.

## For Every Retained Module

- Blueprint is registered through app/module_registry.py.
- /modules/<slug>/ loads.
- /modules/<slug>/health returns the shared envelope.
- /modules/<slug>/api/state returns an ownership-focused summary.
- Pure service tests pass.
- Route tests pass.
- Fixtures are deterministic and clearly marked as debug data.
- The module does not import app/workflows.
- Links point to the active-learning product, not deleted workflows.
- User-facing terminology matches the final dashboard.

## Boundary Checks

- data_workspace: validates Dataset and FeatureMatrix contracts.
- projection: owns coordinates, not labels or clusters.
- algorithm_adapters: exposes AnalysisResult and delegates to SSDBCODI.
- selection: owns selected IDs only.
- labeling: owns annotations only.
- scatterplot: renders supplied state and delegates actions.
- ssdbcodi: owns clustering/outlier computation and scores.
- rule_panel: owns explanation rules and deterministic plan construction.

## Final Workflow Check

After a module change, create an active-learning session and verify that import, analysis, rules, recommendation, plot linking, label commit, and the next round still work.

## Completion

~~~bash
python -m unittest discover -s tests
python -m compileall app tests
git diff --check
~~~
