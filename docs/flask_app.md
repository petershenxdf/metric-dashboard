# Local Flask App Design

## 1. Purpose

This project should run as one simple local Flask app.

The Flask app is not only the final product shell. It is also the main visual testing environment for each module.

## 2. Local Run Command

Primary command:

```powershell
python run.py
```

Expected local URL:

```text
http://127.0.0.1:5000
```

No deployment, Docker, cloud setup, auth system, or production database is required at this stage.

## 3. App Factory

The app should use a small app factory:

```python
def create_app(enabled_modules=None):
    app = Flask(__name__)
    register_core_routes(app)
    register_modules(app, enabled_modules)
    register_workflows(app, enabled_modules)
    return app
```

The app factory should make it easy to enable a single module, a group of modules, or the full dashboard.

## 4. Module Registry

Use a central module registry:

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

The dashboard shell should register modules through this registry. Modules should not import the dashboard shell.

## 5. Required Routes

App routes:

```text
/                         integrated dashboard home
/health                   app health
/modules/                 module lab index
/workflows/               workflow demo index
```

Module routes:

```text
/modules/<module>/        visible module debug page
/modules/<module>/health  module health
/modules/<module>/api/... module state/action APIs
```

Workflow routes:

```text
/workflows/data-projection/
/workflows/default-analysis/
/workflows/selection-context/
/workflows/selection-labeling/
/workflows/analysis-selection/
/workflows/analysis-labeling/
/workflows/scatter-selection/
/workflows/scatter-labeling/
/workflows/provider-feedback/
/workflows/chat-selection/
/workflows/chat-intent/
/workflows/instruction-constraints/       Path A constraint preview
/workflows/instruction-ssdbcodi/           Path B plan preview
/workflows/metric-refinement-loop/         Path A end-to-end
/workflows/direct-refinement-loop/         Path B end-to-end
/workflows/strategy-comparison/            Path A vs Path B side-by-side
```

`/workflows/` groups these routes by debug purpose:

1. core pipeline smoke tests,
2. state boundary probes,
3. visual integration tests,
4. provider diagnostics,
5. future workflows.

See `docs/workflows.md` for the detailed workflow map and route stability rule.

## 6. Module Debug Page Standard

Every module debug page should include:

1. Module name and status.
2. Input fixture controls or example inputs.
3. Main visible output.
4. JSON/state preview.
5. Links to related APIs.
6. Clear notes about what is real and what is mocked.

For example:

1. `data_workspace` shows a dataset table and feature matrix preview.
2. `projection` shows an SVG MDS plot and coordinate table.
3. `selection` shows clickable points and selected/unselected JSON.
4. `labeling` shows selected points converted into cluster/outlier annotations.
5. `analysis-labeling` shows Steps 1-5 together on one visual debug page.
6. `scatterplot` shows the Step 1-6 render payload with click/rectangle selection, saved groups, adjustable clusters, and labeling workflows.
7. `provider-feedback` shows the Step 6.5 provider contract between `algorithm_adapters` and SSDBCODI score diagnostics.
8. `chatbox` shows a chat UI with mock or real selection and label context.

## 7. Testing Layers

Each module should have three testing layers:

1. Unit tests
   - Pure Python tests for schemas and services.

2. Flask tests
   - Flask test client verifies routes and APIs.

3. Browser/manual checks
   - Developer opens the module page and inspects the visual state.

Unit tests are necessary, but they are not enough.

## 8. Local State

For early development, use simple local state:

1. in-memory state for current dataset, selection, and manual annotations.
2. fixtures for repeatable module demos.
3. no production database.

If persistence becomes useful later, prefer a small local JSON or SQLite layer, but do not add it early unless needed.

## 9. Frontend Simplicity

Use:

1. Flask templates.
2. simple CSS.
3. vanilla JavaScript for interactions.
4. SVG or canvas for visualization.

Avoid a heavy frontend framework unless the project clearly needs it later.

## 10. Completion Rule

A module is not complete until:

1. its service tests pass.
2. its Flask route tests pass.
3. its module debug page can be opened locally.
4. its debug page makes the module behavior visible.
