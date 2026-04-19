# Direct Refinement Orchestrator Module Design (Path B)

## Purpose

The direct refinement orchestrator coordinates the **Path B** end-to-end update
flow: structured feedback is compiled into SSDBCODI-native inputs (seeds,
feature scales, parameter overrides, excluded clusters, merged clusters) and
fed directly into the SSDBCODI provider. No Mahalanobis metric is learned on
this path.

This is the sibling of `metric_refinement_orchestrator` (Path A). The two
orchestrators are kept independent because:

1. They have different step lists, different error codes, and different
   intermediate payloads.
2. Mixing both strategies into one orchestrator makes each strategy harder to
   debug in isolation and makes history ambiguous.
3. `split_cluster` and `reclassify_outlier` are **native** on Path B but
   **deferred** on Path A, so the two orchestrators reject or accept different
   intent sets.

The `/workflows/strategy-comparison/` page reads both histories side by side so
developers can compare the two update strategies on the same feedback.

## Path B Scope

Path B updates data by **feeding feedback directly to SSDBCODI**:

```text
trigger (strategy: "direct_ssdbcodi")
  -> direct_feedback_adapter.build_plan
        produces a DirectFeedbackPlan:
          - seed_updates        (cluster seeds and outlier seeds)
          - feature_scale       (pre-scale of feature matrix)
          - param_overrides     (n_clusters, alpha, beta, contamination, min_pts)
          - excluded_clusters
          - merged_cluster_groups
  -> apply feature_scale to feature matrix     (X' = X · S)
  -> algorithm_adapters.run_default_analysis   (SSDBCODI provider, on X')
        with merged seeds and param overrides
  -> projection.run                             (on X' if feature_scale changed)
  -> record run in history
  -> return updated dashboard state
```

Unlike Path A, Path B does not learn a pair-based Mahalanobis metric. When
`feature_scale` entries exist the feature matrix is pre-scaled with a diagonal
matrix `S`, which is a simpler transform than a full `L`. Projection reruns
only when `feature_scale` or `seed_updates` changed the feature geometry;
when only seeds or `n_clusters` change, projection can be reused.

## Why `split_cluster` and `reclassify_outlier` Are Native Here

The two intents that Path A defers map directly onto SSDBCODI's native inputs:

1. `split_cluster` - increment `n_clusters` and/or add new cluster seeds inside
   the target group. SSDBCODI's density-safe KMeans bootstrap picks up the
   higher `k` and the new seeds, which produces the requested split.
2. `reclassify_outlier` - push points into labeling's `mark_outlier` or
   `mark_not_outlier` annotations; SSDBCODI reads those overrides on re-run.
   This is the same override mechanism already implemented in Step 6.5.

The metric-only Path A cannot force either outcome: ITML changes distances but
not `k`, and a metric change may not move a point across the contamination
threshold. Path B therefore treats these intents as first-class inputs, not
deferred errors.

## Responsibilities

1. Receive a refinement trigger from labeling, intent instruction, or a manual
   "direct refine" button with `strategy: "direct_ssdbcodi"`.
2. Ask `direct_feedback_adapter` to build a `DirectFeedbackPlan` from current
   labeling annotations plus the current `StructuredInstruction`.
3. Apply `feature_scale` (diagonal pre-scale) to the feature matrix when
   present.
4. Merge `seed_updates` with existing SSDBCODI bootstrap seeds and manual
   labels (labeling module remains the source of truth for manual labels).
5. Call `algorithm_adapters.run_default_analysis` with param overrides and the
   merged seeds so SSDBCODI re-runs with the new configuration.
6. Trigger an updated projection on the transformed matrix when geometry
   changed.
7. Record each completed run in a Path-B-specific history list.
8. Support rollback to a prior run.
9. Provide a Flask timeline page for debugging the direct-update flow.

## Not Responsible For

1. Parsing natural language.
2. Rendering scatterplot points.
3. Owning selection state.
4. Owning manual label state (still `labeling`).
5. Owning structured instruction state (still `intent_instruction`).
6. Implementing SSDBCODI internals (still `ssdbcodi`).
7. Learning a Mahalanobis metric (that is Path A).
8. Executing Path A refinement - that belongs to
   `metric_refinement_orchestrator`.

## Target Files

```text
app/modules/direct_refinement_orchestrator/
  __init__.py
  schemas.py
  service.py
  history.py
  fixtures.py
  routes.py
  templates/direct_refinement_orchestrator/index.html

tests/modules/direct_refinement_orchestrator/
  test_service.py
  test_history.py
  test_routes.py
```

## Run Result Contract

```json
{
  "refinement_run_id": "direct_refine_003",
  "strategy": "direct_ssdbcodi",
  "status": "completed",
  "trigger": {
    "source": "chat_intent",
    "instruction_version": 4
  },
  "steps": [
    {"name": "plan_build",   "status": "completed", "seed_updates": 4, "feature_scale_changed": true, "param_overrides": {"n_clusters": 4}},
    {"name": "feature_scale","status": "completed"},
    {"name": "ssdbcodi_run", "status": "completed", "provider": "ssdbcodi"},
    {"name": "projection",   "status": "completed"},
    {"name": "effective_analysis", "status": "completed"}
  ],
  "outputs": {
    "direct_feedback_plan_id": "plan_003",
    "ssdbcodi_run_id":         "ssdbcodi_003",
    "projection_id":           "projection_003",
    "cluster_run_id":          "cluster_003",
    "outlier_run_id":          "outlier_003"
  },
  "previous_run_id": "direct_refine_002"
}
```

## History and Rollback

Each completed run is appended to this orchestrator's own in-memory history
list. Each entry stores:

1. the `DirectFeedbackPlan` used,
2. the resulting SSDBCODI run ID, projection ID, cluster run ID, and outlier
   run ID,
3. the active `StructuredInstruction` version at trigger time,
4. the labeling annotation snapshot at trigger time.

Rollback restores the plan and stored outputs of a prior run as active without
re-running the algorithm. History is **scoped to Path B only**. Path A keeps
its own independent history in `metric_refinement_orchestrator`.

## Flask Routes

```text
/modules/direct-refinement-orchestrator/                 orchestrator debug page
/modules/direct-refinement-orchestrator/health           module health
/modules/direct-refinement-orchestrator/api/run          run direct refinement now
/modules/direct-refinement-orchestrator/api/history      list prior Path B runs
/modules/direct-refinement-orchestrator/api/rollback     rollback to a prior run id
/modules/direct-refinement-orchestrator/api/reset        clear Path B history
/workflows/direct-refinement-loop/                       full Path B workflow demo
/workflows/strategy-comparison/                          Path A vs Path B side-by-side
```

## Flask Debug Page Requirements

The page should show:

1. current active `StructuredInstruction` summary.
2. current labeling annotation summary (especially outlier overrides).
3. run button (triggers Path B).
4. timeline of each orchestration step for the latest run.
5. `DirectFeedbackPlan` preview (seed updates, feature_scale, param overrides).
6. history list with run IDs and step statuses.
7. rollback buttons per history entry.
8. explicit note that `split_cluster` and `reclassify_outlier` are accepted on
   this path.

## Testing

Unit tests:

1. actionable instruction triggers expected step order.
2. `split_cluster` intent produces a plan with `n_clusters += 1` and new seeds;
   the run completes without an `intent_deferred` error.
3. `reclassify_outlier` intent is converted into labeling outlier overrides and
   the SSDBCODI run reflects them.
4. `feature_weight` intent populates `feature_scale` and the feature matrix is
   pre-scaled before SSDBCODI.
5. Seeds from `anchor_point` intent are merged with bootstrap seeds.
6. `ignore_cluster` intent removes that cluster's seeds from the merged seed
   set for this run.
7. `merge_clusters` intent relabels seeds from the absorbed cluster(s).
8. SSDBCODI run failure returns diagnostics and preserves prior active run.
9. Projection is reused when only seed updates change and feature geometry is
   unchanged.
10. History is appended only on success.
11. Rollback restores the targeted Path B run as active without recomputation.

Flask route tests:

1. debug page returns 200.
2. run API returns timeline JSON with the Path B step list.
3. history API returns a list scoped to Path B only.
4. rollback API accepts a known run ID and rejects unknown IDs.
5. reset API clears Path B history only (does not touch Path A).

Manual browser check:

1. open `/modules/direct-refinement-orchestrator/`.
2. run a sample direct refinement with a `split_cluster` instruction and
   confirm a new cluster appears on the downstream analysis output.
3. run a sample direct refinement with a `reclassify_outlier` instruction and
   confirm the point's outlier state flips on the downstream analysis output.
4. inspect the timeline and confirm the `DirectFeedbackPlan` is visible.
5. click rollback on a prior run and confirm active state updates.
6. open `/workflows/strategy-comparison/` and confirm Path A and Path B runs
   on the same feedback render side by side.

## Completion Criteria

This module is complete when:

1. The Path B refinement loop can be tested with real SSDBCODI end to end.
2. `split_cluster` and `reclassify_outlier` intents are accepted and produce
   visible changes in the downstream analysis.
3. Every Path B run is inspectable on its own timeline page.
4. Rollback works for at least one prior Path B run.
5. `/workflows/strategy-comparison/` shows matched runs from Path A and
   Path B on the same feedback for side-by-side evaluation.
