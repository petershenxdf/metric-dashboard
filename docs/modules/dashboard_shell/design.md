# Dashboard Shell Module Design

## Purpose

The dashboard shell owns the local Flask app structure.

It is responsible for making every module visible in the browser and for composing modules into workflow demos and the final integrated dashboard.

The shell should stay thin. It should not contain data normalization, MDS, clustering, outlier detection, selection logic, labeling logic, chat parsing, metric learning, or refinement internals.

## Responsibilities

1. Create the Flask app with `create_app()`.
2. Register core routes.
3. Register module debug routes.
4. Register workflow demo routes.
5. Provide shared base templates and CSS.
6. Show module status and links.
7. Compose modules into the final dashboard only after modules work independently.

## Not Responsible For

1. Computing projections.
2. Running algorithms.
3. Owning selection state.
4. Owning manual label state.
5. Parsing chat messages.
6. Creating metric-learning constraints.

## Target Files

```text
app/
  __init__.py
  module_registry.py
  routes.py
  templates/
    base.html
    home.html
    modules_index.html
    workflows_index.html
  static/
    app.css
    app.js
```

## Public API

```python
def create_app(enabled_modules=None):
    ...
```

`enabled_modules` should make it possible to run:

1. all modules.
2. one module.
3. a small subset of modules for interaction testing.

## Required Flask Routes

```text
/                 integrated dashboard or landing page
/health           app health
/modules/         module lab index
/workflows/       workflow demo index
```

## Module Registry Contract

Each module should expose:

```python
def create_blueprint():
    ...
```

The shell registers modules through a central registry:

```python
ModuleInfo(
    slug="data-workspace",
    package_name="data_workspace",
    title="Data Workspace",
    purpose="Dataset loading, point IDs, metadata, and feature matrix.",
    status="working",
    blueprint_factory=data_workspace.create_blueprint,
)
```

Modules should not import the dashboard shell.

## Flask Debug Page

The shell debug page is `/modules/`.

It should show:

1. module name.
2. module purpose.
3. status: missing, stubbed, working, integrated.
4. link to module debug page.
5. link to module health/API endpoint.

## Workflow Pages

The shell should also expose workflow pages:

```text
/workflows/data-projection/
/workflows/default-analysis/
/workflows/selection-context/
/workflows/analysis-selection/
/workflows/selection-labeling/
/workflows/analysis-labeling/
/workflows/scatter-selection/
/workflows/scatter-labeling/
/workflows/chat-selection/
/workflows/chat-intent/
/workflows/instruction-constraints/
/workflows/refinement-loop/
```

These pages combine a few modules at a time, not the entire system.

Current working workflow pages:

1. `/workflows/data-projection/`
2. `/workflows/default-analysis/`
3. `/workflows/selection-context/`
4. `/workflows/analysis-selection/`
5. `/workflows/selection-labeling/`
6. `/workflows/analysis-labeling/`

## Testing

Unit or service tests:

1. `create_app()` returns a Flask app.
2. enabled module filtering works.
3. module registry rejects unknown modules clearly.

Flask route tests:

1. `GET /health` returns OK.
2. `GET /modules/` returns 200.
3. registered module pages return 200.
4. disabled modules are not mounted.

Manual browser check:

1. run `python run.py`.
2. open `http://127.0.0.1:5000/modules/`.
3. confirm module cards and workflow links are visible.

## Completion Criteria

This module is complete when the local Flask app can host module pages and workflow demos.
