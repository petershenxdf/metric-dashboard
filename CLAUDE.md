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

This is a local Flask dashboard for human-in-the-loop metric learning. The stack is pure Python + Flask + vanilla JS - no frontend framework, no production database.

### App Factory

`app/__init__.py` exports `create_app(enabled_modules=None)`, which calls:
1. `register_core_routes(app)` - `/`, `/health`, `/modules/`, `/workflows/`
2. `register_modules(app, enabled_modules)` - loads blueprints from the module registry
3. `register_workflows(app, enabled_modules)` - registers workflow blueprints from `app/workflows/`

### Module Registry

Each module is declared as a `ModuleInfo` entry (slug, package_name, title, purpose, status, blueprint_factory) in `app/module_registry.py`. Blueprint factories are lazy-loaded via `importlib.import_module` — modules are only imported when their blueprint is actually registered. The dashboard shell reads this registry; modules never import the dashboard shell.

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

State is owned by exactly one module - other modules read through contracts, never mutate:

| State | Owner |
|-------|-------|
| dataset / feature matrix | `data_workspace` |
| projection coordinates | `projection` |
| cluster assignments | `algorithm_adapters` (legacy LOF+KMeans) and `ssdbcodi` (new integrated provider) |
| outlier scores | `algorithm_adapters` (LOF score) and `ssdbcodi` (`tScore`) |
| ssdbcodi per-point intermediate scores (`rScore`, `lScore`, `simScore`, `tScore`) | `ssdbcodi` |
| selected point IDs | `selection` |
| manual annotations | `labeling` |
| chat history | `chatbox` |
| structured instructions | `intent_instruction` |
| metric constraints | `metric_learning_adapter` |
| refinement history | `refinement_orchestrator` |

### Current Pipeline (Steps 1-6, all implemented)

```
data_workspace -> projection -> algorithm_adapters -> selection -> labeling -> scatterplot
```

- `algorithm_adapters`: LOF outlier detection runs first, then deterministic KMeans on non-outlier points via `SequentialLofThenKMeansProvider`. Future providers replace this while preserving the same dashboard-facing schemas.
- `selection`: supports `select`/`deselect`/`replace`/`toggle`/`clear`, named selection groups (not semantic labels), sources include `point_click`, `rectangle`, `lasso`, `api`, `workflow_fixture`, `selection_group`.
- `labeling`: converts selected points into `assign_cluster`, `assign_new_class`, `mark_outlier`, `mark_not_outlier` annotations -> structured feedback instructions.
- `scatterplot`: builds a render payload from upstream state; does not own selection or label truth.

The main manual test page for the full Step 1-6 path is `/workflows/scatter-labeling/`.

### SSDBCODI Module (parallel clustering/outlier provider)

`app/modules/ssdbcodi/` implements *Semi-Supervised Density-Based Clustering with Outlier Detection Integrated* ([arXiv:2208.05561](https://arxiv.org/abs/2208.05561)) as a separate, registered module. It is the algorithm the project plans to adopt in place of the existing `SequentialLofThenKMeansProvider`.

- Bootstrap: density-safe KMeans (default `k=3`, user-configurable) seeds SSDBCODI by promoting each dense cluster's centroid-nearest point to a labeled normal seed. These bootstrap anchors remain active as a stable baseline; manual labels override only the explicitly labeled point.
- Algorithm formulas follow the paper contract: symmetric `rDist = max(cDist(p), cDist(q), dist(p,q))`, `lScore` from nearest-neighbor `rDist`, `simScore` from nearest labeled outlier distance, and `tScore = alpha(1-rScore) + beta(1-lScore) + gamma*simScore`.
- The debug page uses the existing `selection` and `labeling` module stores: click and rectangle selection are additive, selected points use black center dots, saved selection groups work, and labels are limited to `cluster_1...cluster_n` plus `outlier`.
- The debug page includes multiple deterministic fixtures (`demo`, `moons`, `circles`) selected by `dataset_id`; selection, labels, and SSDBCODI store state are scoped per dataset.
- GET `/modules/ssdbcodi/` previews the current result without writing run history. `POST /modules/ssdbcodi/api/label` saves pending feedback only; `POST /modules/ssdbcodi/api/run` recomputes and stores results in `SsdbcodiStore`.
- Per-point scores `rScore`, `lScore`, `simScore`, `tScore` are persisted in `SsdbcodiStore` for downstream metric-learning consumption.
- Output schemas (`ClusterResult`, `OutlierResult`) live in `app/shared/schemas.py` (re-exported by `algorithm_adapters/schemas.py`), so downstream modules consume SSDBCODI results without changes.
- The debug page is at `/modules/ssdbcodi/`. See `docs/modules/ssdbcodi/design.md` for the full contract.

### Workflows

Workflow files live in `app/workflows/`. A workflow page connects multiple modules on one visual debug page. It does not own module internals - it orchestrates through module schemas and service calls.

Key workflows:
- `/workflows/data-projection/` - Steps 1-2
- `/workflows/default-analysis/` - Steps 1-3 (uses `default_analysis_outlier_debug` fixture with visible outliers)
- `/workflows/analysis-selection/` - Steps 1-4 with click + rectangle selection
- `/workflows/analysis-labeling/` - Steps 1-5 (main test page pre-scatterplot)
- `/workflows/scatter-labeling/` - Steps 1-6 (current main manual test page)

### Shared Layer (`app/shared/`)

Code that multiple modules or workflows need lives in `app/shared/`:

| File | Purpose |
|------|---------|
| `schemas.py` | `Dataset`, `FeatureMatrix`, `AnalysisResult`, `ClusterResult`, `OutlierResult`, etc. |
| `flask_helpers.py` | `api_success`, `api_error` response envelope helpers |
| `fixtures.py` | Cross-module fixture datasets (wide-gap, default analysis) used by workflows and scatterplot |
| `request_helpers.py` | Shared Flask request parsing (`n_clusters_from_request`, `dataset_id_from_request`, `apply_selection_action_or_error`) |
| `effective_analysis.py` | Pure logic to overlay manual labels on raw algorithm output |

**Layering rule:** `modules → shared` is OK. `modules → workflows` is a violation. When both modules and workflows need the same code, it belongs in `app/shared/`. Workflow files (`app/workflows/fixtures.py`, `app/workflows/effective_analysis.py`) are thin re-export shims pointing to `app/shared/` for backward compatibility.

### Module Boundaries (never cross these)

1. Scatterplot does not parse chat text and does not own selection or label truth.
2. Chatbox does not call clustering, outlier detection, projection, or metric learning.
3. Selection state is owned by the selection module only.
4. Labeling owns manual annotations; scatterplot sends label actions *to* labeling.
5. Existing clustering/outlier algorithms are only accessed through `algorithm_adapters`.
6. Modules do not import unrelated module internals - use schemas, services, APIs, or workflow pages.
7. Modules never import from `app/workflows/` - shared code goes in `app/shared/`.

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

这个项目的核心设计思想是模块化设计：每个模块相对独立，有自己的测试、可视化单元，又可以和其他模块整合，开发时不至于互相影响。
