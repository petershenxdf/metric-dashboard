# Repository Guidance

## Product

This repository contains one product workflow: a generic, persistent active-learning dashboard for structured tabular data.

The current loop is:

~~~text
DatasetAdapter -> preprocessing -> MDS -> SSDBCODI
-> decision-tree explanation rules
-> deterministic RecommendationPlanV2
-> optional DeepSeek V4 Pro translation
-> LabelEvents -> next immutable round
~~~

Wine is only a test and demo fixture. Never encode wine-specific assumptions in reusable services.

## Commands

~~~bash
python run.py
python -m unittest discover -s tests
python -m compileall app tests
git diff --check
~~~

The default product URL is http://127.0.0.1:5001/workflows/active-learning-dashboard/.

## Boundaries

- data_workspace owns common dataset and feature-matrix contracts.
- projection owns display coordinates.
- ssdbcodi owns clustering, outlier results, and analysis scores.
- algorithm_adapters exposes the stable analysis boundary.
- selection owns transient point selection.
- labeling owns module-lab annotations.
- scatterplot renders state and does not own analysis truth.
- rule_panel generates explanation-only rules and deterministic recommendations.
- active_learning owns dataset versions, sessions, rounds, LabelEvents, history-aware plans, and LLM translation.
- workflows orchestrate modules but modules must never import from app/workflows.

DeepSeek never selects points and never changes clustering, outlier status, rules, or labels. Its output must preserve plan_id, focus_category, target_rule_ids, and ordered recommended_point_ids.

## Flask Structure

app/__init__.py creates the app and registers lazy module/workflow blueprints from app/module_registry.py.

Retained module labs are available under /modules/<slug>/. The only registered workflow is active-learning-dashboard.

JSON APIs use:

~~~json
{"ok": true, "data": {}, "error": null, "diagnostics": {}}
~~~

## Engineering Rules

- Keep deterministic analysis and recommendation logic in pure services.
- Keep API/network code outside ranking logic.
- Preserve stable IDs and deterministic tie-breaking.
- Store raw values separately from transformed model features.
- Never expose ground-truth columns to analysis, recommendations, plots, or TranslationPacket.
- Add tests whenever a schema, round transition, ranking rule, or provider contract changes.
- Keep module debug pages useful, but do not create additional product workflows for module combinations.
- Update current docs instead of adding historical Step documents.

## Environment

Only the active DeepSeek and persistence settings belong in .env.example:

~~~text
METRIC_DASHBOARD_DEEPSEEK_BASE_URL
METRIC_DASHBOARD_DEEPSEEK_API_KEY
METRIC_DASHBOARD_LLM_TIMEOUT_SECONDS
METRIC_DASHBOARD_ACTIVE_LEARNING_DB_PATH
~~~

Do not commit .env, runtime_data, SQLite files, generated caches, or local API keys.
