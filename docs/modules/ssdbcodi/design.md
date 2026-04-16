# SSDBCODI Module Design

## Purpose

The SSDBCODI module implements *Semi-Supervised Density-Based Clustering with
Outlier Detection Integrated* ([arXiv:2208.05561](https://arxiv.org/abs/2208.05561))
as a first-class module in the dashboard. SSDBCODI is the algorithm the project
intends to adopt in place of the existing `algorithm_adapters` provider
(`SequentialLofThenKMeansProvider`) for the human-in-the-loop refinement loop.

It produces:

1. cluster assignments compatible with `algorithm_adapters.ClusterResult`,
2. outlier flags compatible with `algorithm_adapters.OutlierResult`,
3. four per-point intermediate scores (`rScore`, `lScore`, `simScore`,
   `tScore`) that are persisted in a module-owned store for downstream
   metric-learning steps.

## Responsibilities

1. Compute pairwise Euclidean distances, core distances (`cDist`), and
   reachability distances (`rDist`).
2. Run a modified Prim-style SSDBSCAN expansion from labeled seed points to
   propagate cluster labels while tracking the maximum edge weight (`Emax`)
   along each point's path to its seed.
3. Compute per-point scores from the paper-aligned formulas:
   - `rDist(p, q) = max(cDist(p), cDist(q), dist(p, q))`.
   - `rScore = exp(-Emax)` - connectivity strength to the nearest labeled
     normal seed.
   - `LD(q) = mean(rDist(nearest_i(q), q))` over the `MinPts` nearest
     neighbors by `rDist`; `lScore = exp(-LD(q))`.
   - `simScore = exp(-dist(q, nearest_labeled_outlier))`. If no labeled
     outlier exists yet, the module sets `simScore = 0` so the similarity term
     does not invent outlier evidence.
   - `tScore = alpha(1 - rScore) + beta(1 - lScore) + (1 - alpha - beta) * simScore`.
4. Mark the top `contamination` fraction by `tScore` as outliers and honor any
   manual outlier overrides from the labeling module.
5. Persist the latest result and a bounded run history in `SsdbcodiStore`.
6. Expose a Flask debug page that visualizes the current preview and supports
   selection-driven labeling plus explicit stored re-runs.

## Not Responsible For

1. Owning manual annotations - those remain in the `labeling` module store.
2. Owning selection state - those remain in the `selection` module store.
3. Performing metric learning (the `metric_learning_adapter` module covers
   that step).
4. Replacing the existing `algorithm_adapters` blueprint; SSDBCODI provides a
   parallel module that downstream code can opt into.

## Bootstrap Flow

SSDBCODI needs normal seed labels before it can expand. The service therefore
keeps a bootstrap seed layer available whenever `bootstrap=True`:

1. Compute a density-safe candidate set using core distances so obvious
   isolated points are not promoted to normal seeds.
2. Run deterministic KMeans (`algorithm_adapters.clustering.kmeans`) on those
   dense candidates with `n_clusters` (default 3, user-configurable through
   controls / API).
3. For each cluster, take the candidate point closest to the cluster centroid
   and use it as a seed labeled `cluster_<i>`.
4. Run SSDBCODI on all points with the bootstrap seeds. With no manual labels,
   this output is the *initial* clustering plus outliers.

Manual `assign_cluster` annotations are merged on top of those bootstrap seeds.
That means the initial bootstrap anchors remain stable unless the user
explicitly relabels that same point or marks that point as an outlier. This
prevents one new label from accidentally dropping distant bootstrap anchors.
Manual `mark_outlier` annotations force points into the outlier set and also
become labeled outliers for `simScore`; `mark_not_outlier` removes points from
the outlier set.

The debug page intentionally separates feedback entry from algorithm execution:
`POST /api/label` saves pending manual feedback through the labeling module,
while `POST /api/run` is the only action that re-runs SSDBCODI and stores a new
result. To keep the UI consistent with the labeling workflow, the page still
previews pending labels immediately: selected points are recolored on the
client after a successful label action, but neighboring points are not
recomputed until Run and Store.

The debug page does not write run history on `GET`; it only stores results
after `POST /api/run`. This keeps browser refreshes and pending label edits
from polluting `SsdbcodiStore.history`.

## Debug Datasets

The module exposes deterministic local fixtures through the `dataset_id`
parameter:

1. `ssdbcodi_demo_fixture` - three compact separated clusters with far
   outliers. This is the default sanity-check dataset.
2. `ssdbcodi_moons_fixture` - two curved moon-shaped groups with outliers.
   This checks density expansion on non-convex structure.
3. `ssdbcodi_circles_fixture` - concentric inner/outer rings with outliers.
   This checks non-linear cluster shape and visual clarity.

Selection state, labeling state, latest SSDBCODI result, and history are scoped
by dataset so switching fixtures does not mix feedback between datasets.

## Debug Page Interaction Rules

1. Point click and rectangle selection are additive and update the plot in place
   without a full page refresh.
2. Selected points are marked with black center dots and slightly stronger
   outlines.
3. Apply Label recolors the currently selected points as a pending manual-label
   preview and returns `pending_run: true`.
4. Run and Store calls SSDBCODI with merged bootstrap + manual seeds, stores the
   result, and reloads the page with the recomputed clusters, outliers, scores,
   and seeds.

## Output Contracts

### Per-point score (`PointScores`)

```json
{
  "point_id": "ring_a_01",
  "cluster_id": "cluster_1",
  "is_outlier": false,
  "r_score": 0.91,
  "l_score": 0.88,
  "sim_score": 0.74,
  "t_score": 0.18,
  "c_dist": 0.21,
  "e_max": 0.09,
  "seed_origin_point_id": "ring_a_03"
}
```

### Run result (`SsdbcodiResult`)

```json
{
  "run_id": "ssdbcodi_run_abcdef012345",
  "algorithm": "ssdbcodi_numpy",
  "cluster_result": { "...": "ClusterResult schema" },
  "outlier_result": { "...": "OutlierResult schema" },
  "point_scores": [ "...PointScores..." ],
  "seeds": [ {"point_id": "...", "cluster_id": "...", "source": "..."} ],
  "parameters": {
    "n_clusters_bootstrap": 3,
    "min_pts": 3,
    "alpha": 0.4,
    "beta": 0.3,
    "contamination": 0.13,
    "bootstrap_used": true
  },
  "diagnostics": { "...": "..." }
}
```

## Files

```text
app/modules/ssdbcodi/
  __init__.py
  algorithm.py        pure numpy algorithm (no Flask imports)
  schemas.py          PointScores, SeedRecord, SsdbcodiResult
  service.py          bootstrap, run_ssdbcodi, SsdbcodiProvider
  store.py            SsdbcodiStore with latest_result + bounded history
  fixtures.py         deterministic debug datasets
  routes.py           Flask blueprint
  templates/ssdbcodi/index.html

tests/modules/ssdbcodi/
  test_algorithm.py
  test_service.py
  test_routes.py

docs/modules/ssdbcodi/design.md
```

## Flask Routes

```text
/modules/ssdbcodi/                         debug page with interactive plot
/modules/ssdbcodi/health                   health endpoint
/modules/ssdbcodi/api/state                latest run, selection, labeling, and history summary
/modules/ssdbcodi/api/scores               stored per-point rScore/lScore/simScore/tScore
/modules/ssdbcodi/api/result               full stored SsdbcodiResult JSON
/modules/ssdbcodi/api/run                  POST: re-run SSDBCODI with current labels and store result
/modules/ssdbcodi/api/select               POST: additive selection through the selection module
/modules/ssdbcodi/api/clear-selection      POST: clear active selection
/modules/ssdbcodi/api/reset-selection      POST: reset selection state and groups
/modules/ssdbcodi/api/groups               GET/POST: list or save selection groups
/modules/ssdbcodi/api/groups/<id>/select   POST: restore a saved selection group
/modules/ssdbcodi/api/groups/<id>          DELETE: delete a saved selection group
/modules/ssdbcodi/api/label                POST: label current selection; leaves run pending
/modules/ssdbcodi/api/clear-labels         POST: drop all manual annotations
/modules/ssdbcodi/api/reset-labels         POST: reset labeling store
/modules/ssdbcodi/api/reset                POST: clear SSDBCODI store, selection, and labels
```

All API responses use the dashboard envelope:

```json
{ "ok": true, "data": {...}, "error": null, "diagnostics": {...} }
```

## Integration with Existing Modules

| Module | Touch point | Direction |
|---|---|---|
| `data_workspace` | uses `create_dataset`, `create_feature_matrix` | reads only |
| `algorithm_adapters` | reuses deterministic KMeans for density-safe bootstrap; emits compatible `ClusterResult` / `OutlierResult` | reads + reuses schemas |
| `selection` | owns active selected/unselected state and saved selection groups for the debug page | reads + delegates writes |
| `labeling` | owns manual annotations; SSDBCODI reads `LabelingState` and sends label actions through `apply_labeling_action` | reads + delegates writes |
| `scatterplot` (future) | can render `SsdbcodiResult` with per-point cluster + outlier flag | downstream consumer |
| `metric_learning_adapter` (future) | will consume the persisted `r_score`/`l_score`/`sim_score`/`t_score` to weight constraints | downstream consumer |

## Testing

Unit tests (`tests/modules/ssdbcodi/test_algorithm.py`):

1. Pairwise distance is symmetric with zero diagonal.
2. Core distance returns the MinPts-th neighbor distance.
3. Reachability uses `max(cDist(p), cDist(q), dist(p, q))` and is symmetric.
4. SSDBSCAN expansion assigns all points and seeds keep their labels.
5. `lScore` uses nearest-neighbor `rDist`; `simScore` uses nearest labeled outlier distance and is zero when there is no labeled outlier.
6. Combined `tScore` honors `alpha`, `beta`, and `gamma = 1 - alpha - beta` with the paper-aligned positive `simScore` term.
7. Top-`contamination` outlier selection.
8. End-to-end `run_ssdbcodi_core` returns the expected schema, including labeled outlier indices.
9. Invalid inputs (no seeds, bad `min_pts`) raise `ValueError`.

Service tests (`test_service.py`):

1. Bootstrap returns one dense, non-outlier seed per KMeans cluster.
2. Bootstrap is used when no cluster labels exist and the demo fixture yields balanced normal clusters plus the known far outliers.
3. Bootstrap anchors remain active when manual labels exist; manual labels override only matching point seeds.
4. All four intermediate scores are populated per point.
5. Manual `mark_outlier` overrides force outlier state and provide labeled-outlier evidence for `simScore`.
6. Bootstrap `n_clusters` is adjustable.
7. `SsdbcodiStore` keeps a bounded history with run summaries.

Route tests (`test_routes.py`):

1. Debug page returns 200 with bootstrap-driven preview content.
2. Module appears in `/modules/`.
3. `GET /modules/ssdbcodi/` does not write run history.
4. `/api/run` returns scores; `/api/state` reflects the latest explicit run.
5. `/api/label` requires current selection, stores pending feedback only, and then `/api/run` uses merged bootstrap + user seeds.
6. Selection groups can be saved, restored, and deleted through the selection module contract.
7. Unknown selected point IDs and invalid parameters return standard error envelopes.
8. `/api/reset` clears SSDBCODI store, selection, and labels.

## Definition of Done

- [x] Pure service / algorithm logic with unit tests.
- [x] Flask debug page at `/modules/ssdbcodi/`.
- [x] At least one JSON state API endpoint.
- [x] Flask route smoke tests.
- [x] Module registered in `app/module_registry.py`.
- [x] This design document exists.
- [ ] Manual browser inspection on the debug page.
