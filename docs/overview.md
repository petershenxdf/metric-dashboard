# Project Overview and Top-Level Design

## 1. Goal

Build a local Flask dashboard for human-in-the-loop metric learning.

The app should let a user:

1. Load or use a sample dataset.
2. Project data into 2D with MDS.
3. View default clustering and outlier detection results.
4. Select points in a scatterplot.
5. Give feedback through a chatbox.
6. Convert feedback into structured instructions.
7. Use those instructions to guide metric learning.
8. Rerun projection, clustering, and outlier detection.
9. See the updated visualization.

Deployment is out of scope for now. The app only needs to run locally.

## 2. Flask-First Architecture

The project should be one simple Flask app with a module lab.

The module lab is a set of local pages that expose each module independently. This is required because unit tests alone are not enough for a dashboard project.

Every module should have:

1. A service layer for pure logic.
2. A Flask route layer for visible local testing.
3. Fixtures or mock data for standalone demos.
4. API endpoints for inspecting internal state.
5. Clear contracts for integration with other modules.

## 3. Route Model

```text
/                                      integrated dashboard
/health                                health check
/modules/                              module lab index
/modules/<module_name>/                module debug page
/modules/<module_name>/api/...         module debug/state APIs
/workflows/<workflow_name>/            multi-module interaction demo
```

The module debug page should answer the question: "Is this module working correctly enough to trust it before integration?"

The workflow page should answer the question: "Do these modules interact correctly when combined?"

## 4. Core User Flow

```text
data workspace
  -> projection
  -> algorithm adapters
  -> scatterplot
  -> selection
  -> chatbox
  -> intent instruction
  -> metric-learning adapter
  -> refinement orchestrator
  -> updated projection and algorithm outputs
```

Detailed flow:

1. Data workspace creates a dataset with stable point IDs.
2. Projection computes 2D coordinates with MDS.
3. Algorithm adapters call existing clustering and outlier detection.
4. Scatterplot renders points with cluster colors and outlier markers.
5. User selects points.
6. Selection module stores selected/unselected state.
7. Chatbox receives user text and current selection context.
8. Intent instruction module classifies and compiles structured instructions.
9. Metric-learning adapter converts instructions into constraints.
10. Refinement orchestrator runs the update sequence.
11. The integrated dashboard refreshes the visible state.

## 5. Product Constraints

1. Existing clustering and outlier detection algorithms must not be redesigned.
2. Chatbox must not directly perform clustering or outlier detection.
3. Scatterplot must not parse language.
4. Selection state must be accessible outside the scatterplot.
5. Vague user feedback must not become hard supervision without clarification.
6. All cross-module contracts should use shared schemas.
7. Every module must be independently visible in Flask.

## 6. Target File Structure

```text
metric-dashboard/
  run.py
  requirements.txt
  README.md

  app/
    __init__.py
    module_registry.py
    routes.py

    shared/
      schemas.py
      flask_helpers.py
      fixtures.py

    templates/
      base.html
      home.html
      modules_index.html
      workflows_index.html

    static/
      app.css
      app.js

    modules/
      dashboard_shell/
        routes.py
        service.py
        templates/dashboard_shell/
        static/dashboard_shell/

      data_workspace/
        schemas.py
        service.py
        fixtures.py
        routes.py
        templates/data_workspace/
        static/data_workspace/

      projection/
        schemas.py
        mds.py
        service.py
        fixtures.py
        routes.py
        templates/projection/
        static/projection/

      algorithm_adapters/
        schemas.py
        clustering.py
        outliers.py
        service.py
        fixtures.py
        routes.py
        templates/algorithm_adapters/

      selection/
        schemas.py
        store.py
        service.py
        fixtures.py
        routes.py
        templates/selection/
        static/selection/

      scatterplot/
        schemas.py
        service.py
        fixtures.py
        routes.py
        templates/scatterplot/
        static/scatterplot/

      chatbox/
        schemas.py
        service.py
        fixtures.py
        routes.py
        templates/chatbox/
        static/chatbox/

      intent_instruction/
        schemas.py
        classifier.py
        compiler.py
        fixtures.py
        routes.py
        templates/intent_instruction/

      metric_learning_adapter/
        schemas.py
        adapter.py
        fixtures.py
        routes.py
        templates/metric_learning_adapter/

      refinement_orchestrator/
        schemas.py
        service.py
        fixtures.py
        routes.py
        templates/refinement_orchestrator/

    workflows/
      data_projection.py
      scatter_selection.py
      chat_intent.py
      refinement_loop.py

  tests/
    modules/
      data_workspace/
      projection/
      algorithm_adapters/
      selection/
      scatterplot/
      chatbox/
      intent_instruction/
      metric_learning_adapter/
      refinement_orchestrator/
    flask/
      test_module_pages.py
      test_workflow_pages.py

  docs/
    overview.md
    flask_app.md
    process.md
    module_debug_checklist.md
    integration_testing.md
    state_and_api_contracts.md
    modules/
      <module_name>/
        design.md
```

This is the target structure. The project can migrate toward it gradually.

## 7. Module Contract

Each module should expose these boundaries where applicable:

1. `service.py`
   - pure logic.
   - independently unit tested.

2. `schemas.py`
   - module-local schemas or re-exports of shared schemas.

3. `fixtures.py`
   - sample data for tests and Flask module demo.

4. `routes.py`
   - Flask blueprint for module debug page and APIs.

5. `templates/<module_name>/`
   - HTML for visible local testing.

6. `static/<module_name>/`
   - small module-specific JavaScript or CSS when needed.

## 8. Module List

| Module | Main Job | Flask Debug Page |
| --- | --- | --- |
| Dashboard Shell | App factory, module registry, integrated pages | `/`, `/modules/`, `/workflows/` |
| Data Workspace | Dataset identity and feature matrix | `/modules/data-workspace/` |
| Projection | MDS 2D coordinates | `/modules/projection/` |
| Algorithm Adapters | Existing clustering/outlier wrappers | `/modules/algorithm-adapters/` |
| Selection | Selected/unselected point state | `/modules/selection/` |
| Scatterplot | Visual point rendering and selection UI | `/modules/scatterplot/` |
| Chatbox | Dialogue UI and clarification flow | `/modules/chatbox/` |
| Intent Instruction | Message classification and structured instructions | `/modules/intent-instruction/` |
| Metric-Learning Adapter | Instruction to constraint conversion | `/modules/metric-learning-adapter/` |
| Refinement Orchestrator | End-to-end update coordination | `/modules/refinement-orchestrator/` |

## 9. Structured Instruction Families

Start with these families:

1. `same_class`
2. `different_class`
3. `split_into_n_classes`
4. `merge_groups`
5. `is_outlier`
6. `not_outlier`
7. `needs_clarification`
8. `non_actionable`

Example:

```json
{
  "instruction_type": "same_class",
  "status": "actionable",
  "target": {
    "source": "selected_points",
    "point_ids": ["p1", "p7", "p9"]
  },
  "explicitness": "explicit",
  "requires_followup": false,
  "followup_question": null
}
```

## 10. Integration Strategy

Do not wait until the end to use Flask.

Each step should add:

1. Pure service behavior.
2. Unit tests.
3. Flask module page.
4. Flask API endpoint.
5. A small integration or workflow page when the module has a neighbor to interact with.

The final dashboard should be built by composing already visible modules.
