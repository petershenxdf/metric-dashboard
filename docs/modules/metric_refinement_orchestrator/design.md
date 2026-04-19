# Metric Refinement Orchestrator Module Design (Path A)

## Purpose

The metric refinement orchestrator coordinates the **Path A** end-to-end update
flow: structured feedback becomes a learned Mahalanobis metric, the metric is
applied as a linear pre-transform to the feature matrix, and projection plus
algorithm adapters run again on the transformed features.

This is one of two sibling orchestrators. Path B (`direct_refinement_orchestrator`)
feeds the same structured feedback directly into SSDBCODI without learning a
metric. They are kept as independent modules because mixing their strategy
branches, error codes, and run histories inside one orchestrator makes each
strategy harder to debug in isolation.

This orchestrator does not own the implementation of metric learning, projection,
clustering, outlier detection, labeling, chat, or scatterplot. It calls other
modules in the correct order and records a history of runs so they can be
inspected and rolled back.

## Path A Scope

Path A updates data by **learning a distance metric**:

```text
trigger
  -> metric_learning_adapter.build_constraints
  -> metric_learning_adapter.fit        (returns LearnedMetric M, L)
  -> apply L to feature matrix          (X' = X · L)
  -> projection.run                     (on X')
  -> algorithm_adapters.run             (SSDBCODI, on X')
  -> record run in history
  -> return updated dashboard state
```

Path A keeps `split_cluster` and `reclassify_outlier` **deferred**. These intents
cannot be expressed through a distance metric alone: ITML cannot change
KMeans's `k`, and a metric change may not move a point across SSDBCODI's
contamination threshold. If such an intent appears, this orchestrator returns
`intent_deferred` with a hint that the user should route it through Path B
(`direct_refinement_orchestrator`) instead.

## Responsibilities

1. Receive a refinement trigger from labeling, intent instruction, or a manual
   "refine" button with `strategy: "metric_learning"`.
2. Ask `metric_learning_adapter` to build a `ConstraintSet` from current
   labeling annotations plus the current `StructuredInstruction`.
3. Call `metric_learning_adapter.fit` to produce a `LearnedMetric`.
4. Apply the metric's `L` as a linear pre-transform to the feature matrix.
5. Trigger an updated projection on the transformed matrix.
6. Rerun clustering and outlier detection through algorithm adapters on the
   transformed matrix.
7. Record each run in refinement history with inputs, outputs, and step
   statuses.
8. Support rollback to a prior run.
9. Provide a Flask timeline page for debugging the update flow.

## Not Responsible For

1. Parsing natural language.
2. Rendering scatterplot points.
3. Implementing metric learning internals (owned by `metric_learning_adapter`).
4. Implementing clustering or outlier detection (owned by
   `algorithm_adapters` / `ssdbcodi`).
5. Owning selection state.
6. Owning manual label state.
7. Owning structured instruction state.
8. Executing Path B (direct SSDBCODI) refinement - that belongs to
   `direct_refinement_orchestrator`.

## Target Files

```text
app/modules/metric_refinement_orchestrator/
  __init__.py
  schemas.py
  service.py
  history.py
  fixtures.py
  routes.py
  templates/metric_refinement_orchestrator/index.html

tests/modules/metric_refinement_orchestrator/
  test_service.py
  test_history.py
  test_routes.py
```

## Run Result Contract

```json
{
  "refinement_run_id": "metric_refine_003",
  "strategy": "metric_learning",
  "status": "completed",
  "trigger": {
    "source": "chat_intent",
    "instruction_version": 4
  },
  "steps": [
    {"name": "constraint_build", "status": "completed", "pair_count": 12, "conflicts": 0},
    {"name": "metric_fit",       "status": "completed", "provider": "itml"},
    {"name": "projection",       "status": "completed"},
    {"name": "clustering",       "status": "completed"},
    {"name": "outlier",          "status": "completed"}
  ],
  "outputs": {
    "learned_metric_id": "metric_003",
    "projection_id":    "projection_003",
    "cluster_run_id":   "cluster_003",
    "outlier_run_id":   "outlier_003"
  },
  "previous_run_id": "metric_refine_002"
}
```

## History and Rollback

Every completed run is appended to this orchestrator's own in-memory history
list. Each entry stores:

1. the `ConstraintSet` used,
2. the `LearnedMetric` produced,
3. references to projection, cluster, and outlier run IDs,
4. the active `StructuredInstruction` version at trigger time.

Rollback replaces the current active metric with a prior run's metric and
re-emits its projection and analysis outputs without rerunning them.

History is **scoped to Path A only**. The Path B orchestrator keeps its own
independent history. The `/workflows/strategy-comparison/` page reads from both
histories to show side-by-side comparisons.

## Flask Routes

```text
/modules/metric-refinement-orchestrator/                 orchestrator debug page
/modules/metric-refinement-orchestrator/health           module health
/modules/metric-refinement-orchestrator/api/run          run refinement now
/modules/metric-refinement-orchestrator/api/history      list prior runs
/modules/metric-refinement-orchestrator/api/rollback     rollback to a prior run id
/modules/metric-refinement-orchestrator/api/reset        clear history
/workflows/metric-refinement-loop/                       full Path A workflow demo
```

## Flask Debug Page Requirements

The page should show:

1. current active `StructuredInstruction` summary.
2. current labeling annotation summary.
3. run button (triggers Path A).
4. timeline of each orchestration step for the latest run.
5. history list with run IDs and step statuses.
6. rollback buttons per history entry.
7. intermediate payload previews (constraint set, learned metric metadata).
8. error states when a step fails, including the `intent_deferred` case with
   a pointer to Path B.

## Testing

Unit tests:

1. actionable instruction triggers expected step order.
2. incomplete instruction is not triggered by the orchestrator (intent module
   should have blocked it earlier).
3. `split_cluster` or `reclassify_outlier` intent triggers an `intent_deferred`
   error before metric fit, with a `suggested_strategy: "direct_ssdbcodi"` hint.
4. metric-fit failure returns diagnostics and does not mutate projection.
5. projection failure returns diagnostics and preserves prior run as active.
6. algorithm adapter failures return diagnostics.
7. history is appended only on success.
8. rollback restores the targeted run as active without rerunning computation.

Flask route tests:

1. debug page returns 200.
2. run API returns timeline JSON.
3. history API returns a list scoped to Path A only.
4. rollback API accepts a known run ID and rejects unknown IDs.
5. reset API clears Path A history only (does not touch Path B).

Manual browser check:

1. open `/modules/metric-refinement-orchestrator/`.
2. run sample refinement.
3. inspect the timeline.
4. verify failed and successful steps are visually clear.
5. submit a `split_cluster` instruction and confirm the `intent_deferred`
   error names Path B as the correct route.
6. click rollback on a prior run and confirm active state updates.

## Completion Criteria

This module is complete when:

1. The Path A refinement loop can be tested with mock providers end to end.
2. Every Path A run is inspectable on its own timeline page.
3. Rollback works for at least one prior Path A run.
4. Deferred intents produce a clear, named error that points the user to
   Path B instead of silently no-opping.
