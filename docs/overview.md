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

The feedback-to-update loop forks into two parallel update strategies after the shared upstream stages. Users can pick either strategy per refinement; comparison is available through `/workflows/strategy-comparison/`.

```text
data workspace
  -> projection
  -> algorithm adapters
  -> scatterplot
  -> selection
  -> labeling / annotation
  -> chatbox (with strategy selector)
  -> intent instruction for chat-derived feedback
  -> unified structured feedback
      |
      +-- Path A (metric learning) --------------------------------------------+
      |     -> metric_learning_adapter (ConstraintSet + LearnedMetric M, L)    |
      |     -> metric_refinement_orchestrator                                  |
      |         -> apply L as linear pre-transform: X' = X · L                 |
      |         -> re-projection on X'                                         |
      |         -> algorithm_adapters re-run (SSDBCODI) on X'                  |
      |                                                                        |
      +-- Path B (direct SSDBCODI) --------------------------------------------+
            -> direct_feedback_adapter (DirectFeedbackPlan: seeds, scales, k)  |
            -> direct_refinement_orchestrator                                  |
                -> apply feature_scale S: X' = X · S (diagonal only)           |
                -> algorithm_adapters re-run (SSDBCODI) with merged seeds      |
                   and param overrides                                         |
                -> re-projection on X' only if feature geometry changed        |
      -> updated projection and algorithm outputs (from whichever path ran)
```

Detailed flow:

1. Data workspace creates a dataset with stable point IDs.
2. Projection computes 2D coordinates with MDS.
3. Algorithm adapters call clustering and outlier detection through replaceable providers.
   The current default provider is SSDBCODI, which emits cluster assignments and
   outlier flags through the same dashboard schemas. The old LOF-then-KMeans
   provider remains available explicitly for comparison.
4. Scatterplot renders points with cluster colors and outlier markers.
5. User selects points through clicks, lasso, rectangle, API calls, or future selection gestures.
6. Selection module stores selected/unselected state, can save reusable named selection groups, and exposes reusable selection context.
7. Labeling module converts direct label actions into manual annotations or structured feedback instructions.
8. Chatbox receives user text and current selection/labeling context, and exposes a refinement strategy selector (`metric_learning` | `direct_ssdbcodi`).
9. Intent instruction module classifies chat text and compiles structured instructions. It emits all eight Phase 1 intents; downstream adapters enforce path-specific acceptance.
10. **Path A**: `metric_learning_adapter` merges labeling annotations plus structured instructions into a `ConstraintSet`, runs a replaceable metric learner (default ITML), and returns a Mahalanobis matrix `M`. Its Cholesky factor `L` is applied as a linear pre-transform to the feature matrix. `split_cluster` and `reclassify_outlier` are rejected here with `intent_deferred` and redirected to Path B.
11. **Path B**: `direct_feedback_adapter` compiles the same structured feedback into a `DirectFeedbackPlan` (seed updates, `feature_scale`, `param_overrides`, excluded clusters, merged clusters) that is fed directly to SSDBCODI. `split_cluster` becomes `n_clusters += 1` plus interior seeds; `reclassify_outlier` becomes a labeling outlier override. No Mahalanobis metric is learned on Path B.
12. `metric_refinement_orchestrator` (Path A) and `direct_refinement_orchestrator` (Path B) each run their update sequence, record their own history, and support rollback independently. Keeping the orchestrators separate keeps each strategy's step list, error codes, and history easy to debug in isolation.
13. The integrated dashboard refreshes the visible state from whichever path ran.

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

      metric_learning_adapter/          Path A: pair constraints -> learned metric
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

      direct_feedback_adapter/          Path B: feedback -> SSDBCODI-native plan
        schemas.py
        plan_builder.py
        fixtures.py
        routes.py
        templates/direct_feedback_adapter/

      metric_refinement_orchestrator/   Path A orchestrator
        schemas.py
        service.py
        history.py
        fixtures.py
        routes.py
        templates/metric_refinement_orchestrator/

      direct_refinement_orchestrator/   Path B orchestrator
        schemas.py
        service.py
        history.py
        fixtures.py
        routes.py
        templates/direct_refinement_orchestrator/

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
      provider_feedback.py
      chat_selection.py
      chat_intent.py
      instruction_constraints.py       Path A constraint preview
      instruction_ssdbcodi.py          Path B plan preview
      metric_refinement_loop.py        Path A end-to-end
      direct_refinement_loop.py        Path B end-to-end
      strategy_comparison.py           Path A vs Path B side-by-side

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
      direct_feedback_adapter/
      metric_refinement_orchestrator/
      direct_refinement_orchestrator/
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
| Algorithm Adapters | Active SSDBCODI provider boundary plus legacy LOF/KMeans comparison provider | `/modules/algorithm-adapters/` |
| Selection | Selected/unselected point state | `/modules/selection/` |
| Labeling | Manual point annotations, cluster labels, and outlier labels | `/modules/labeling/` |
| Scatterplot | Visual point rendering and selection UI | `/modules/scatterplot/` |
| SSDBCODI | Active semi-supervised density-based clustering with integrated outlier detection and score diagnostics | `/modules/ssdbcodi/` |
| Chatbox | Dialogue UI, suggestion chips, clarification flow, strategy selector | `/modules/chatbox/` |
| Intent Instruction | Router + extractor with replaceable LLM provider; emits instruction deltas for both paths | `/modules/intent-instruction/` |
| Metric-Learning Adapter | **Path A** constraint builder + replaceable metric learner (default ITML), returns learned `M` | `/modules/metric-learning-adapter/` |
| Direct Feedback Adapter | **Path B** plan builder that compiles feedback into SSDBCODI-native seeds, feature scales, and param overrides | `/modules/direct-feedback-adapter/` |
| Metric Refinement Orchestrator | **Path A** end-to-end update coordination, history, rollback | `/modules/metric-refinement-orchestrator/` |
| Direct Refinement Orchestrator | **Path B** end-to-end update coordination, history, rollback | `/modules/direct-refinement-orchestrator/` |

## 9. Structured Instruction Families

Chat-derived feedback flows through intent instruction and produces an evolving `StructuredInstruction` state. Each turn the extractor emits a delta that is applied to this state.

### Phase 1 Intents

The extractor emits eight intents in Phase 1. Six are path-agnostic; two are Path B-only.

Shared intents (valid on both paths):

1. `feature_weight` - increase, decrease, or ignore a feature.
2. `group_similar` - two groups should be closer together.
3. `group_dissimilar` - two groups should be farther apart.
4. `merge_clusters` - two or more clusters should be treated as one.
5. `anchor_point` - one reference point attracts a target group.
6. `ignore_cluster` - a cluster is excluded from this update round.

Path B-only intents (deferred on Path A):

7. `split_cluster` - requires changing SSDBCODI's `n_clusters` and/or adding interior seeds. Accepted by `direct_feedback_adapter`; rejected by `metric_learning_adapter` with `intent_deferred`.
8. `reclassify_outlier` - requires flipping a labeled outlier override in the labeling store. Accepted by `direct_feedback_adapter`; rejected by `metric_learning_adapter` with `intent_deferred`.

Router-level meta-categories that do not produce pair constraints or plan entries:

9. `needs_clarification`
10. `non_actionable`
11. `meta_query`

### Labeling-Derived Instructions

Manual labels from the labeling module remain as-is. On **Path A**, `assign_cluster` annotations become intra-label must-link pairs, and `mark_outlier` / `mark_not_outlier` annotations stay within the labeling module's effective state rather than being translated into metric pair constraints in Phase 1. On **Path B**, every manual label becomes a direct `seed_updates` entry in the `DirectFeedbackPlan`, including outlier overrides.

### Why the Two Paths

Path A learns a distance metric from pair constraints (ITML) and retransforms the feature space. Path B uses SSDBCODI's own semi-supervised inputs (seeds, labeled outliers, `n_clusters`, contamination, feature scales) to update the algorithm directly. Both consume the same structured feedback. `/workflows/strategy-comparison/` runs both paths on the same feedback snapshot so the two update strategies can be evaluated side-by-side.

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
/workflows/selection-labeling/
/workflows/analysis-selection/
/workflows/analysis-labeling/
/modules/scatterplot/
/workflows/scatter-selection/
/workflows/scatter-labeling/
/workflows/provider-feedback/
```

The grouped workflow contract lives in `docs/workflows.md`. Use that document
when adding, reordering, or renaming workflow pages.

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
the same projection, SSDBCODI clustering/outlier detection, additive
click/rectangle selection, saved selection groups, and labeling controls on one
shared point-ID fixture. Manual labels are limited to `cluster_1...cluster_n`
and `outlier`; they are passed into SSDBCODI and update the effective
cluster/outlier state used by the frontend while baseline outputs remain
available in the state API.

`/modules/scatterplot/` is the Step 6 module page. It turns already-computed
projection, analysis, selection, and labeling state into a render payload and
visible SVG plot without owning selection or label truth. `/workflows/scatter-selection/`
and `/workflows/scatter-labeling/` verify those boundaries with selection and
labeling connected. The Step 1-6 workflow preserves prior interaction behavior:
click selection, rectangle selection, saved selection groups, adjustable
`n_clusters`, and manual cluster/outlier labeling.

`/workflows/provider-feedback/` is the Step 6.5 provider diagnostics page. It
checks that `algorithm_adapters.run_default_analysis()` resolves to SSDBCODI
while the standalone SSDBCODI result still exposes seed records and per-point
scores for future metric-learning work.

`/modules/ssdbcodi/` is the dedicated module page for the SSDBCODI algorithm
([arXiv:2208.05561](https://arxiv.org/abs/2208.05561)). It is the active
clustering/outlier provider behind `algorithm_adapters` and keeps a dedicated
debug page for inspecting its scores. Bootstrap behavior:
the module computes density-safe KMeans center seeds (default `k = 3`,
user-configurable) so obvious far outliers are not promoted to normal seeds.
Those bootstrap seeds stay active as reusable seed inputs across runs;
however, under the current weighted-distance assignment rule a bootstrap
seed point's final `cluster_id` is recomputed each run and can shift to
another class. Only manual cluster annotations are output locks for their
own points. The debug page uses the same selection behavior
as Step 1-6: click and rectangle selection add to the active selection,
selected points use black center dots, and saved selection groups are restored
through the selection module. Label controls are limited to
`cluster_1...cluster_n` plus `outlier`; label actions save pending feedback,
while Run and Store recomputes and persists SSDBCODI. Per-point intermediate
scores (`rScore`, `lScore`, `simScore`, `tScore`) are persisted in
`SsdbcodiStore` for downstream metric-learning use. The page also includes
`demo`, `moons`, and `circles` fixtures for browser testing different shapes.
See `docs/modules/ssdbcodi/design.md` for the full contract.
