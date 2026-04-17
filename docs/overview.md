# Project Overview and Top-Level Design

## 1. Goal

Build a local Flask dashboard for human-in-the-loop metric learning.

The app should let a user:

1. Load or use a sample dataset.
2. Project data into 2D with MDS.
3. View default clustering and outlier detection results.
4. Select points in a scatterplot.
5. Assign selected points to a cluster/class or mark them as outliers.
6. Give feedback through a chatbox.
7. Convert direct labels and chat feedback into structured instructions.
8. Use those instructions to guide metric learning.
9. Rerun projection, clustering, and outlier detection.
10. See the updated visualization.

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
  -> labeling / annotation
  -> chatbox
  -> intent instruction for chat-derived feedback
  -> unified structured feedback
  -> metric-learning adapter
  -> refinement orchestrator
  -> updated projection and algorithm outputs
```

Detailed flow:

1. Data workspace creates a dataset with stable point IDs.
2. Projection computes 2D coordinates with MDS.
3. Algorithm adapters call existing clustering and outlier detection through replaceable providers.
   The current provider runs Local Outlier Factor first, excludes detected outliers,
   and then runs deterministic KMeans on the remaining points.
4. Scatterplot renders points with cluster colors and outlier markers.
5. User selects points through clicks, lasso, rectangle, API calls, or future selection gestures.
6. Selection module stores selected/unselected state, can save reusable named selection groups, and exposes reusable selection context.
7. Labeling module converts direct label actions into manual annotations or structured feedback instructions.
8. Chatbox receives user text and current selection/labeling context.
9. Intent instruction module classifies chat text and compiles structured instructions.
10. Metric-learning adapter merges labeling annotations plus structured instructions into a `ConstraintSet`, runs a replaceable metric learner (default ITML), and returns a Mahalanobis matrix `M`. Its Cholesky factor `L` is applied as a linear pre-transform to the feature matrix.
11. Refinement orchestrator runs the update sequence on the transformed matrix, records history, and supports rollback.
12. The integrated dashboard refreshes the visible state.

## 5. Product Constraints

1. Existing clustering and outlier detection algorithms must not be redesigned.
2. Chatbox must not directly perform clustering or outlier detection.
3. Scatterplot must not parse language.
4. Selection state must be accessible outside the scatterplot.
5. Labeling state must be owned by the labeling module, not by scatterplot or chatbox.
6. Vague user feedback must not become hard supervision without clarification.
7. All cross-module contracts should use shared schemas.
8. Every module must be independently visible in Flask.

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
      request_helpers.py
      effective_analysis.py

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
        static/selection/ optional for larger debug-page interactions

      labeling/
        schemas.py
        service.py
        fixtures.py
        routes.py
        templates/labeling/
        static/labeling/ optional for larger debug-page interactions

      scatterplot/
        schemas.py
        service.py
        fixtures.py
        routes.py
        templates/scatterplot/
        static/scatterplot/

      ssdbcodi/
        algorithm.py
        schemas.py
        service.py
        store.py
        fixtures.py
        routes.py
        templates/ssdbcodi/

      chatbox/
        schemas.py
        service.py
        fixtures.py
        routes.py
        templates/chatbox/
        static/chatbox/

      intent_instruction/
        schemas.py
        router.py
        extractor.py
        providers/
          base.py
          mock.py
          local_qwen.py
          cloud_claude.py
        fixtures.py
        routes.py
        templates/intent_instruction/

      metric_learning_adapter/
        schemas.py
        constraint_builder.py
        providers/
          base.py
          identity.py
          itml.py
        adapter.py
        fixtures.py
        routes.py
        templates/metric_learning_adapter/

      refinement_orchestrator/
        schemas.py
        service.py
        history.py
        fixtures.py
        routes.py
        templates/refinement_orchestrator/

    workflows/
      fixtures.py               re-exports from app.shared.fixtures
      effective_analysis.py      re-exports from app.shared.effective_analysis
      data_projection.py
      default_analysis.py
      selection_context.py
      selection_labeling.py
      analysis_selection.py
      analysis_labeling.py
      scatter_selection.py
      scatter_labeling.py
      chat_intent.py
      refinement_loop.py

  tests/
    modules/
      data_workspace/
      projection/
      algorithm_adapters/
      selection/
      labeling/
      scatterplot/
      ssdbcodi/
      chatbox/
      intent_instruction/
      metric_learning_adapter/
      refinement_orchestrator/
    flask_app/
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
| Algorithm Adapters | LOF outlier detection, KMeans clustering, and future algorithm providers | `/modules/algorithm-adapters/` |
| Selection | Selected/unselected point state | `/modules/selection/` |
| Labeling | Manual point annotations, cluster labels, and outlier labels | `/modules/labeling/` |
| Scatterplot | Visual point rendering and selection UI | `/modules/scatterplot/` |
| SSDBCODI | Semi-supervised density-based clustering with integrated outlier detection (replaces the sequential LOF + KMeans provider) | `/modules/ssdbcodi/` |
| Chatbox | Dialogue UI, suggestion chips, clarification flow | `/modules/chatbox/` |
| Intent Instruction | Router + extractor with replaceable LLM provider; emits instruction deltas | `/modules/intent-instruction/` |
| Metric-Learning Adapter | Constraint builder + replaceable metric learner (default ITML), returns learned `M` | `/modules/metric-learning-adapter/` |
| Refinement Orchestrator | End-to-end update coordination, history, rollback | `/modules/refinement-orchestrator/` |

## 9. Structured Instruction Families

Chat-derived feedback flows through intent instruction and produces an evolving `StructuredInstruction` state. Each turn the extractor emits a delta that is applied to this state.

### Phase 1 Intents (ITML-Aligned)

These intents map cleanly to pair-based metric learning constraints and are the initial implementation scope:

1. `feature_weight` - increase, decrease, or ignore a feature (implemented through pre-scaling, not pair constraints).
2. `group_similar` - two groups should be closer together.
3. `group_dissimilar` - two groups should be farther apart.
4. `merge_clusters` - two or more clusters should be treated as one.
5. `anchor_point` - one reference point attracts a target group.
6. `ignore_cluster` - a cluster is excluded from metric updates this round.

Router-level meta-categories that do not produce pair constraints:

7. `needs_clarification`
8. `non_actionable`
9. `meta_query`

### Labeling-Derived Instructions

Manual labels from the labeling module remain as-is. In the constraint builder, `assign_cluster` annotations become intra-label must-link pairs, and `mark_outlier` / `mark_not_outlier` annotations stay within the labeling module's effective state rather than being translated into metric pair constraints in Phase 1.

### Phase 2 (Deferred)

The following intents are intentionally excluded from Phase 1 because metric change alone cannot drive them:

1. `split_cluster` - requires changing the clustering algorithm's `k` or running sub-clustering.
2. `reclassify_outlier` - LOF uses a fixed contamination threshold; a metric change may not move a point across the boundary.

Both will be revisited after the clustering and outlier providers are swapped for algorithms that can accept these signals directly. They are not blockers for Phase 1.

Example delta:

```json
{
  "operations": [
    {
      "op": "add",
      "constraint": {
        "id": "c2",
        "intent": "group_similar",
        "group_a": {"source": "selection_group", "ref": "group_001"},
        "group_b": {"source": "cluster", "ref": "cluster_2"}
      }
    }
  ]
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

## 11. Current Working Slice

The currently implemented working slice is:

```text
data_workspace
  -> projection
  -> algorithm_adapters
  -> selection
  -> labeling
  -> scatterplot
```

Browser checks:

```text
/modules/data-workspace/
/modules/projection/
/modules/algorithm-adapters/
/modules/selection/
/workflows/data-projection/
/workflows/default-analysis/
/workflows/selection-context/
/workflows/analysis-selection/
/workflows/selection-labeling/
/workflows/analysis-labeling/
/modules/scatterplot/
/workflows/scatter-selection/
/workflows/scatter-labeling/
```

`/workflows/default-analysis/` uses the `default_analysis_outlier_debug` fixture
so outliers are visible during local debugging. It should not be interpreted as
the final user dataset flow.

`/workflows/selection-context/` uses a selection debug fixture to show how stable
point IDs become selected/unselected context for downstream labeling, chatbox,
and intent modules.

The selection module also supports saved selection groups. These are named point
sets for quickly restoring a previous selection; they are intentionally separate
from semantic labels, which remain the labeling module's responsibility.

`/workflows/analysis-selection/` connects the Step 1-4 path on one shared
fixture: Data Workspace creates point IDs and features, Projection computes MDS
coordinates, Algorithm Adapters mark clusters/outliers, and Selection overlays
active and saved selections on the same SVG plot. It includes a dataset dropdown,
click selection, and rectangle selection. New clicks or rectangle selections are
added to the active selection so the user does not need to choose a selection mode.

`/workflows/selection-labeling/` shows the Step 5 boundary: selection context
is consumed by labeling, and manual annotations are converted into structured
feedback instructions without involving chatbox or metric learning.

`/workflows/analysis-labeling/` is the full Step 1-5 visual test page. It uses
the same projection, LOF outlier detection, KMeans clustering, additive
click/rectangle selection, saved selection groups, and labeling controls on one
shared point-ID fixture. Manual labels are limited to `cluster_1...cluster_n`
and `outlier`; they update the effective cluster/outlier state used by the
frontend while raw algorithm outputs remain available in the state API.

`/modules/scatterplot/` is the Step 6 module page. It turns already-computed
projection, analysis, selection, and labeling state into a render payload and
visible SVG plot without owning selection or label truth. `/workflows/scatter-selection/`
and `/workflows/scatter-labeling/` verify those boundaries with selection and
labeling connected. The Step 1-6 workflow preserves prior interaction behavior:
click selection, rectangle selection, saved selection groups, adjustable
`n_clusters`, and manual cluster/outlier labeling.

`/modules/ssdbcodi/` is the dedicated module page for the SSDBCODI algorithm
([arXiv:2208.05561](https://arxiv.org/abs/2208.05561)). It is a parallel
clustering/outlier provider that replaces the existing sequential LOF + KMeans
approach with a single semi-supervised density-based pass. Bootstrap behavior:
the module computes density-safe KMeans center seeds (default `k = 3`,
user-configurable) so obvious far outliers are not promoted to normal seeds.
Those bootstrap seeds remain stable anchors, and manual labels override only
the explicitly labeled points. The debug page uses the same selection behavior
as Step 1-6: click and rectangle selection add to the active selection,
selected points use black center dots, and saved selection groups are restored
through the selection module. Label controls are limited to
`cluster_1...cluster_n` plus `outlier`; label actions save pending feedback,
while Run and Store recomputes and persists SSDBCODI. Per-point intermediate
scores (`rScore`, `lScore`, `simScore`, `tScore`) are persisted in
`SsdbcodiStore` for downstream metric-learning use. The page also includes
`demo`, `moons`, and `circles` fixtures for browser testing different shapes.
See `docs/modules/ssdbcodi/design.md` for the full contract.
