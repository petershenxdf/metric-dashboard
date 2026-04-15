# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the app
python run.py
# App available at http://127.0.0.1:5000

# Run all tests
python -m unittest discover -s tests

# Run tests for a single module
python -m unittest discover -s tests/modules/data_workspace
python -m unittest discover -s tests/modules/projection

# Check for syntax errors
python -m compileall app tests
```

## Architecture

This is a local Flask dashboard for human-in-the-loop metric learning. The stack is pure Python + Flask + vanilla JS — no frontend framework, no production database.

### App Factory

`app/__init__.py` exports `create_app(enabled_modules=None)`, which calls:
1. `register_core_routes(app)` — `/`, `/health`, `/modules/`, `/workflows/`
2. `register_modules(app, enabled_modules)` — loads blueprints from the module registry
3. `register_workflows(app, enabled_modules)` — registers workflow blueprints from `app/workflows/`

### Module Registry

Each module is declared as a `ModuleInfo` entry (slug, package_name, title, purpose, status, blueprint_factory). The dashboard shell reads this registry; modules never import the dashboard shell.

### Module Contract

Every module under `app/modules/<module_name>/` must have:

| File | Purpose |
|------|---------|
| `service.py` | Pure logic, independently unit-tested |
| `schemas.py` | Module-local or re-exported shared schemas |
| `fixtures.py` | Sample data for tests and debug pages |
| `routes.py` | Flask blueprint with debug page and APIs |
| `templates/<module_name>/index.html` | Browser-visible debug page |

### Route Conventions

- Python packages: `snake_case` (e.g., `data_workspace`)
- Flask URL slugs: `kebab-case` (e.g., `data-workspace`)
- Module pages: `/modules/<module-slug>/`
- Module APIs: `/modules/<module-slug>/api/<action>`
- Workflow pages: `/workflows/<workflow-slug>/`

### API Response Envelope

All debug APIs must return this shape:

```json
{ "ok": true, "data": {}, "error": null, "diagnostics": {} }
```

Errors use `"ok": false` with `"error": { "code": "...", "message": "..." }`.

### State Ownership

State is owned by exactly one module — other modules read through contracts, never mutate:

| State | Owner |
|-------|-------|
| dataset / feature matrix | `data_workspace` |
| projection coordinates | `projection` |
| cluster assignments | `algorithm_adapters` |
| outlier scores | `algorithm_adapters` |
| selected point IDs | `selection` |
| manual annotations | `labeling` |
| chat history | `chatbox` |
| structured instructions | `intent_instruction` |
| metric constraints | `metric_learning_adapter` |
| refinement history | `refinement_orchestrator` |

### Current Pipeline (Steps 1–6, all implemented)

```
data_workspace → projection → algorithm_adapters → selection → labeling → scatterplot
```

- `algorithm_adapters`: LOF outlier detection runs first, then deterministic KMeans on non-outlier points via `SequentialLofThenKMeansProvider`. Future providers replace this while preserving the same dashboard-facing schemas.
- `selection`: supports `select`/`deselect`/`replace`/`toggle`/`clear`, named selection groups (not semantic labels), sources include `point_click`, `rectangle`, `lasso`, `api`, `workflow_fixture`, `selection_group`.
- `labeling`: converts selected points into `assign_cluster`, `assign_new_class`, `mark_outlier`, `mark_not_outlier` annotations → structured feedback instructions.
- `scatterplot`: builds a render payload from upstream state; does not own selection or label truth.

The main manual test page for the full Step 1–6 path is `/workflows/scatter-labeling/`.

### Workflows

Workflow files live in `app/workflows/`. A workflow page connects multiple modules on one visual debug page. It does not own module internals — it orchestrates through module schemas and service calls.

Key workflows:
- `/workflows/data-projection/` — Steps 1–2
- `/workflows/default-analysis/` — Steps 1–3 (uses `default_analysis_outlier_debug` fixture with visible outliers)
- `/workflows/analysis-selection/` — Steps 1–4 with click + rectangle selection
- `/workflows/analysis-labeling/` — Steps 1–5 (main test page pre-scatterplot)
- `/workflows/scatter-labeling/` — Steps 1–6 (current main manual test page)

### Module Boundaries (never cross these)

1. Scatterplot does not parse chat text and does not own selection or label truth.
2. Chatbox does not call clustering, outlier detection, projection, or metric learning.
3. Selection state is owned by the selection module only.
4. Labeling owns manual annotations; scatterplot sends label actions *to* labeling.
5. Existing clustering/outlier algorithms are only accessed through `algorithm_adapters`.
6. Modules do not import unrelated module internals — use schemas, services, APIs, or workflow pages.

### Definition of Done for a Module

A module is complete only when all of these are true:
1. Documented input/output contracts exist.
2. Pure service logic passes unit tests.
3. Flask debug page exists at `/modules/<module-slug>/`.
4. At least one JSON state API endpoint exists.
5. Flask route smoke tests pass.
6. The debug page can be opened and visually inspected in the browser.
7. Design document at `docs/modules/<module_name>/design.md` is updated.

### Docs to Read Before Starting a New Module

```
docs/overview.md
docs/flask_app.md
docs/process.md
docs/module_debug_checklist.md
docs/integration_testing.md
docs/state_and_api_contracts.md
docs/modules/<module_name>/design.md
```
