# Metric Dashboard

## What This Project Is

Metric Dashboard is a local Flask dashboard for human-in-the-loop metric learning.

The app is meant to run locally, stay simple, and make every module visible for debugging. Deployment, production auth, cloud infrastructure, and complex frontend frameworks are not part of the current scope.

Run target:

```powershell
python run.py
```

Expected local URL:

```text
http://127.0.0.1:5000
```

## Product Goal

The dashboard helps a user inspect high-dimensional data, select points, and give feedback that can guide metric learning.

The first product version has two main user-facing areas:

1. Scatterplot
   - Projects user data into 2D with MDS.
   - Shows default clustering results.
   - Shows default outlier detection results.
   - Colors points by cluster.
   - Marks outliers.
   - Lets the user select points.
   - Lets selected points become explicit cluster or outlier labels through the labeling module.

2. Chatbox
   - Receives feedback about selected or unselected points.
   - Converts relevant feedback into structured instructions.
   - Asks clarification when feedback is incomplete.
   - Does not directly run clustering, outlier detection, or metric learning.

Existing clustering and outlier detection algorithms are treated as fixed external logic. They should be wrapped by adapters, not redesigned or silently replaced.

## System Loop

```text
user data
  -> data workspace
  -> MDS projection
  -> clustering adapter
  -> outlier adapter
  -> scatterplot
  -> point selection
  -> direct labeling / annotation
  -> or chatbox feedback
  -> intent instruction for chat-derived feedback
  -> unified structured feedback
  -> metric-learning adapter
  -> refinement orchestrator
  -> updated projection/clusters/outliers
  -> updated scatterplot
```

## Main Design Principle

This is a Flask-first, module-first project.

There should be one simple local Flask app, and every module should have its own visible debug page inside that app.

Unit tests are required, but they are not enough. Because this is a dashboard, each module must also be inspectable in the browser.

Each module should have:

1. pure service logic.
2. unit tests.
3. Flask route tests.
4. a Flask debug page.
5. at least one JSON/state API endpoint.
6. fixtures or mock data for local testing.
7. a clear boundary from other modules.

## Required Local Routes

The Flask app should provide:

```text
/                                      integrated dashboard
/health                                app health
/modules/                              module lab index
/modules/<module_name>/                module debug page
/modules/<module_name>/health          module health
/modules/<module_name>/api/...         module state/action APIs
/workflows/                            workflow demo index
/workflows/<workflow_name>/            multi-module interaction demo
```

The module lab is important. It lets the developer open one module at a time and check whether it works before connecting it to the full dashboard.

## Planned Modules

| Module | Role | Debug Page |
| --- | --- | --- |
| `dashboard_shell` | Flask app, module registry, shared layout, workflow links | `/`, `/modules/`, `/workflows/` |
| `data_workspace` | Dataset loading, point IDs, feature matrix | `/modules/data-workspace/` |
| `projection` | MDS projection into 2D | `/modules/projection/` |
| `algorithm_adapters` | Adapter boundary for clustering and outlier providers, currently backed by SSDBCODI | `/modules/algorithm-adapters/` |
| `selection` | Selected and unselected point state | `/modules/selection/` |
| `labeling` | Manual point annotations, cluster labels, and outlier labels | `/modules/labeling/` |
| `scatterplot` | Point rendering, clusters, outliers, visual selection | `/modules/scatterplot/` |
| `ssdbcodi` | Active semi-supervised clustering/outlier provider plus score diagnostics | `/modules/ssdbcodi/` |
| `chatbox` | Chat UI, user feedback, clarification display | `/modules/chatbox/` |
| `intent_instruction` | Message classification and structured instruction output | `/modules/intent-instruction/` |
| `metric_learning_adapter` | Structured instruction to metric-learning constraints | `/modules/metric-learning-adapter/` |
| `refinement_orchestrator` | Coordinates full refinement loop | `/modules/refinement-orchestrator/` |

## Current Implementation Status

The current working modules are:

1. `dashboard_shell`
   - app factory, registry-driven module/workflow registration, module lab, workflow lab, and placeholders.

2. `data_workspace`
   - fixture dataset, stable point IDs, feature matrix API, and debug page.

3. `projection`
   - MDS projection, projection API, SVG debug plot, and `/workflows/data-projection/`.

4. `algorithm_adapters`
   - defaults to the SSDBCODI integrated clustering/outlier provider.
   - keeps the old LOF-then-KMeans provider available as an explicit legacy provider.
   - returns the same `ClusterResult`, `OutlierResult`, and `AnalysisResult` schemas to downstream modules.
   - `n_clusters` can be adjusted from the module page, workflow page, or API query string.
   - `/workflows/default-analysis/` shows data, projection, outliers, and clusters together.

5. `selection`
   - owns selected and unselected point state.
   - supports `select`, `deselect`, `replace`, `toggle`, and `clear`.
   - supports saved selection groups, which are reusable named point sets.
   - preserves `source`, `mode`, and metadata fields for future UI gestures such as lasso and rectangle selection.
   - `/workflows/selection-context/` shows Data Workspace point IDs converted into reusable selection context.
   - `/workflows/analysis-selection/` connects Steps 1-4 on one shared visual layer with dataset switching, click selection, and rectangle selection.

6. `labeling`
   - owns manual point annotations derived from selected points.
   - supports assigning selected points to a cluster or new class.
   - supports marking selected points as outliers or not outliers.
   - converts annotations into structured feedback instructions.
   - `/workflows/selection-labeling/` shows selection context beside annotation output.
   - `/workflows/analysis-labeling/` connects Steps 1-5 on one shared visual layer: data, projection, SSDBCODI outliers, SSDBCODI clusters, selection, and labeling.
   - in the Step 1-5 workflow, labels are limited to `cluster_1...cluster_n` and `outlier`; those labels are passed into SSDBCODI and then reflected in the effective cluster/outlier state and frontend colors/markers.
   - `/workflows/analysis-labeling/` remains the main manual browser test page for the completed Step 1-5 path.

7. `scatterplot`
   - builds a point render payload from projection, cluster, outlier, selection, and label state.
   - shows projected points with cluster colors, outlier markers, selected point indicators, and manual label context.
   - preserves click selection, rectangle selection, saved selection groups, and adjustable cluster count in the Step 1-6 workflows.
   - exposes `/modules/scatterplot/api/render-payload` for downstream UI rendering.
   - `/workflows/scatter-selection/` tests scatterplot click/rectangle selection and saved groups flowing through the selection module.
   - `/workflows/scatter-labeling/` is the current main Step 1-6 manual browser test page: data, projection, algorithms, scatterplot, selection, saved groups, and labeling together.

The default algorithm-adapter fixture is `default_analysis_outlier_debug`, not Iris. It intentionally contains three compact clusters plus three distant outlier candidates so Step 3 is visually inspectable.

8. `ssdbcodi`
   - implements the active semi-supervised clustering/outlier provider and keeps an independent debug module at `/modules/ssdbcodi/`.
   - uses density-safe KMeans center seeds as stable bootstrap anchors, then merges manual labels on top so one relabel does not drop unrelated anchors.
   - includes selectable debug datasets (`demo`, `moons`, `circles`) to test separated, curved, and ring-shaped structures.
   - persists `rScore`, `lScore`, `simScore`, and `tScore` for downstream metric-learning use.
   - reuses the existing selection and labeling contracts: additive click/rectangle selection, black center dots for selected points, saved selection groups, and label controls limited to `cluster_1...cluster_n` plus `outlier`.
   - keeps label entry and execution separate: Apply Label saves pending labeling feedback; Run and Store recomputes and persists SSDBCODI.
   - returns dashboard-compatible `ClusterResult` and `OutlierResult` schemas and now backs the default `algorithm_adapters` provider boundary.
   - `/workflows/provider-feedback/` verifies the promoted provider boundary beside standalone SSDBCODI score diagnostics.

## Workflow Debug Map

The workflow index is grouped by debugging purpose, not just build order:

1. Core pipeline smoke tests:
   - `/workflows/data-projection/`
   - `/workflows/default-analysis/`
2. State boundary probes:
   - `/workflows/selection-context/`
   - `/workflows/selection-labeling/`
3. Visual integration tests:
   - `/workflows/analysis-selection/`
   - `/workflows/analysis-labeling/`
   - `/workflows/scatter-selection/`
   - `/workflows/scatter-labeling/`
4. Provider diagnostics:
   - `/workflows/provider-feedback/`
5. Future workflows:
   - `/workflows/chat-selection/`
   - `/workflows/chat-intent/`
   - `/workflows/instruction-constraints/`
   - `/workflows/refinement-loop/`

See `docs/workflows.md` for the current workflow contract and grouping rules.

## Module Boundary Rules

1. Scatterplot does not parse chat text.
2. Chatbox does not run clustering, outlier detection, projection, or metric learning.
3. Selection state is owned by the selection module, not hidden inside scatterplot.
4. Labeling owns manual cluster/outlier annotations derived from selected points.
5. Scatterplot can expose label actions, but it must send them to labeling instead of owning label truth.
6. Intent instruction receives chat text plus context and outputs structured instructions.
7. Metric-learning adapter receives structured instructions, not raw chat text.
8. Existing algorithms are accessed only through algorithm adapters.
9. Dashboard shell composes modules but does not own module internals.
10. Integration should happen through schemas, services, APIs, or workflow pages.

## Structured Instructions

Actionable user feedback should become stable structured instructions.

Initial instruction types:

```text
assign_cluster
assign_new_class
same_class
different_class
split_into_n_classes
merge_groups
is_outlier
not_outlier
needs_clarification
non_actionable
```

Example:

```json
{
  "instruction_type": "same_class",
  "status": "actionable",
  "source": "chat_intent",
  "target": {
    "source": "selected_points",
    "point_ids": ["p1", "p7", "p9"]
  },
  "explicitness": "explicit",
  "requires_followup": false,
  "followup_question": null
}
```

If the user input is vague, incomplete, irrelevant, or too general, the system must not invent a hard constraint. It should ask for clarification or mark the message as non-actionable.

## Development Order

Follow the process in `docs/process.md`.

Current planned order:

1. `dashboard_shell`
   - create the Flask app shell and module lab first.

2. `data_workspace`
   - make dataset and feature matrix visible in Flask.

3. `projection`
   - make MDS output visible as an SVG/table in Flask.

4. `algorithm_adapters`
   - make SSDBCODI adapter outputs visible through the existing clustering/outlier schemas.
   - keep the legacy LOF-then-KMeans provider available for comparison.

5. `selection`
   - make selected/unselected state interactive in Flask.
   - keep action/source/mode fields extensible for future selection gestures.
   - let users save and restore named selections without turning them into semantic labels.

6. `labeling`
   - convert selected point IDs into manual cluster/outlier annotations.

7. `scatterplot`
   - render projected points, clusters, outliers, selection, and manual label state.

8. `chatbox`
   - build chat UI with mock or real selection context.

9. `intent_instruction`
   - compile messages into structured instructions.

10. `metric_learning_adapter`
   - convert structured instructions into constraints.

11. `refinement_orchestrator`
   - coordinate one full update loop.

12. integrated dashboard
   - combine already-tested modules.

## Testing Expectations

For each module, do all three:

1. Unit tests:

```powershell
python -m unittest discover -s tests
```

2. Flask route tests:

Use Flask test client to verify debug pages and APIs.

3. Manual browser check:

```powershell
python run.py
```

Then open:

```text
http://127.0.0.1:5000/modules/<module_name>/
```

Confirm the module's visible output, state preview, and interactions work.

## Documentation Map

Read these before coding:

```text
README.md
docs/overview.md
docs/flask_app.md
docs/process.md
docs/module_debug_checklist.md
docs/integration_testing.md
docs/state_and_api_contracts.md
docs/modules/<module_name>/design.md
```

## Rule for Future AI Agents

Do not jump straight into the full dashboard.

For the current module:

1. read the relevant docs.
2. implement only that module.
3. add or update its Flask debug page.
4. add unit tests and Flask route tests.
5. make sure the module can be inspected in the browser.
6. update the module design document if behavior changes.
