# Refinement Orchestrator Module Design

## Purpose

The refinement orchestrator coordinates the end-to-end update flow after valid user feedback.

It does not own the implementation of metric learning, projection, clustering, outlier detection, labeling, chat, or scatterplot. It calls other modules in the correct order and records a history of runs so they can be inspected and rolled back.

## Responsibilities

1. Receive a refinement trigger from labeling, intent instruction, or a manual "refine" button.
2. Ask the metric-learning adapter to build a `ConstraintSet` from current labeling annotations plus the current `StructuredInstruction`.
3. Call the metric-learning adapter's fit to produce a `LearnedMetric`.
4. Apply the metric's `L` as a linear pre-transform to the feature matrix.
5. Trigger an updated projection on the transformed matrix.
6. Rerun clustering and outlier detection through algorithm adapters on the transformed matrix.
7. Record each run in refinement history with inputs, outputs, and step statuses.
8. Support rollback to a prior run.
9. Provide a Flask timeline page for debugging the update flow.

## Not Responsible For

1. Parsing natural language.
2. Rendering scatterplot points.
3. Implementing metric learning internals.
4. Implementing clustering or outlier detection.
5. Owning selection state.
6. Owning manual label state.
7. Owning structured instruction state.

## Target Files

```text
app/modules/refinement_orchestrator/
  __init__.py
  schemas.py
  service.py
  history.py
  fixtures.py
  routes.py
  templates/refinement_orchestrator/index.html

tests/modules/refinement_orchestrator/
  test_service.py
  test_history.py
  test_routes.py
```

## Main Flow

```text
trigger (labeling change or instruction delta or manual refine)
  -> metric_learning_adapter.build_constraints
  -> metric_learning_adapter.fit
  -> apply L to feature matrix
  -> projection.run
  -> algorithm_adapters.run (LOF then KMeans, on transformed matrix)
  -> record run in history
  -> return updated dashboard state
```

Phase 1 never passes `split_cluster` or `reclassify_outlier` intents through this flow. If such an intent appears (for example from a future provider), the orchestrator returns an error that names the intent as deferred until the clustering or outlier algorithm is upgraded.

## Run Result Contract

```json
{
  "refinement_run_id": "refine_003",
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
  "previous_run_id": "refine_002"
}
```

## History and Rollback

Every completed run is appended to an in-memory history list. Each entry stores:

1. the `ConstraintSet` used,
2. the `LearnedMetric` produced,
3. references to projection, cluster, and outlier run IDs,
4. the active `StructuredInstruction` version at trigger time.

Rollback replaces the current active metric with a prior run's metric and re-emits its projection and analysis outputs without rerunning them.

## Flask Routes

```text
/modules/refinement-orchestrator/                 orchestrator debug page
/modules/refinement-orchestrator/health           module health
/modules/refinement-orchestrator/api/run          run refinement now
/modules/refinement-orchestrator/api/history      list prior runs
/modules/refinement-orchestrator/api/rollback     rollback to a prior run id
/modules/refinement-orchestrator/api/reset        clear history
/workflows/refinement-loop/                       full refinement workflow demo
```

## Flask Debug Page Requirements

The page should show:

1. current active `StructuredInstruction` summary.
2. current labeling annotation summary.
3. run button.
4. timeline of each orchestration step for the latest run.
5. history list with run IDs and step statuses.
6. rollback buttons per history entry.
7. intermediate payload previews (constraint set, learned metric metadata).
8. error states when a step fails, including the "intent deferred" case.

## Testing

Unit tests:

1. actionable instruction triggers expected step order.
2. incomplete instruction is not triggered by the orchestrator (intent module should have blocked it earlier).
3. deferred intent triggers an `intent_deferred` error before metric fit.
4. metric-fit failure returns diagnostics and does not mutate projection.
5. projection failure returns diagnostics and preserves prior run as active.
6. algorithm adapter failures return diagnostics.
7. history is appended only on success.
8. rollback restores the targeted run as active without rerunning computation.

Flask route tests:

1. debug page returns 200.
2. run API returns timeline JSON.
3. history API returns a list.
4. rollback API accepts a known run ID and rejects unknown IDs.
5. reset API clears history.

Manual browser check:

1. open `/modules/refinement-orchestrator/`.
2. run sample refinement.
3. inspect the timeline.
4. verify failed and successful steps are visually clear.
5. click rollback on a prior run and confirm active state updates.

## Completion Criteria

This module is complete when:

1. The refinement loop can be tested with mock providers end to end.
2. Every run is inspectable on a timeline page.
3. Rollback works for at least one prior run.
4. Deferred intents produce a clear, named error without silently no-opping.
