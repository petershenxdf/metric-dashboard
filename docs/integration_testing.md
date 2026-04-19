# Incremental Integration Testing

## 1. Purpose

This document explains how to test each module in two modes:

1. standalone mode
   - test the module by itself with fixtures or mocks.

2. integrated-with-previous-modules mode
   - test the module together with modules that have already been built.

This prevents two common problems:

1. modules only work in isolation.
2. the full dashboard is integrated too late.

The grouped workflow index contract is maintained in `docs/workflows.md`.

## 2. Testing Levels

For every module, use four testing levels:

1. service tests
   - pure Python unit tests.

2. route tests
   - Flask test client checks pages and APIs.

3. standalone browser check
   - open `/modules/<module_name>/`.

4. incremental workflow browser check
   - open a `/workflows/.../` page that combines the current module with already built modules.

## 3. Main Integration Matrix

| Step | Current Module | Standalone Page | Integrated Page With Previous Modules |
| --- | --- | --- | --- |
| 0 | `dashboard_shell` | `/modules/` | `/workflows/` |
| 1 | `data_workspace` | `/modules/data-workspace/` | none yet |
| 2 | `projection` | `/modules/projection/` | `/workflows/data-projection/` |
| 3 | `algorithm_adapters` | `/modules/algorithm-adapters/` | `/workflows/default-analysis/` |
| 4 | `selection` | `/modules/selection/` | `/workflows/selection-context/`, `/workflows/analysis-selection/` |
| 5 | `labeling` | `/modules/labeling/` | `/workflows/selection-labeling/`, `/workflows/analysis-labeling/` |
| 6 | `scatterplot` | `/modules/scatterplot/` | `/workflows/scatter-selection/` and `/workflows/scatter-labeling/` |
| 6.5 | `ssdbcodi` provider diagnostics | `/modules/ssdbcodi/` | `/workflows/provider-feedback/` |
| 7 | `chatbox` | `/modules/chatbox/` | `/workflows/chat-selection/` |
| 8 | `intent_instruction` | `/modules/intent-instruction/` | `/workflows/chat-intent/` |
| 9 | `metric_learning_adapter` | `/modules/metric-learning-adapter/` | `/workflows/instruction-constraints/` |
| 10 | `refinement_orchestrator` | `/modules/refinement-orchestrator/` | `/workflows/refinement-loop/` |
| 11 | integrated dashboard | `/` | full app |

Not every workflow needs to be polished. A workflow page can be simple and diagnostic as long as it shows the interaction clearly.

Step 3 currently uses a dedicated `default_analysis_outlier_debug` fixture for
the algorithm-adapter page and `/workflows/default-analysis/`. This fixture is
not Iris; it is intentionally shaped with compact clusters and distant outlier
candidates so SSDBCODI clustering and outlier detection can both be visually
checked.

For Step 3 browser checks, confirm:

1. `/modules/algorithm-adapters/` shows SSDBCODI output through the adapter APIs.
2. SSDBCODI assignments exclude detected outliers.
3. changing `n_clusters` changes the requested cluster count.
4. `/workflows/default-analysis/` shows projection, outliers, and clusters together.

Step 4 currently uses the `selection_iris_debug` fixture and in-memory debug
state. For Step 4 browser checks, confirm:

1. `/modules/selection/` shows supported actions, sources, and modes.
2. point clicks call the `toggle` action.
3. manual action lab supports `select`, `deselect`, `replace`, `toggle`, and `clear`.
4. `/modules/selection/api/context` returns selected and unselected point IDs.
5. the saved selection group form can save the current selection by name.
6. clicking a saved group restores that group's points as the active selection.
7. deleting a saved group removes it without changing the active selection.
8. `/modules/selection/api/groups` returns saved group metadata.
9. `/workflows/selection-context/` shows Data Workspace output and selection context together.

Step 1-4 combined workflow check:

1. open `/workflows/analysis-selection/`.
2. use the dataset dropdown to switch between `wide_gap_analysis_debug` and `default_analysis_outlier_debug`.
3. confirm one SVG shows projected points, cluster colors, SSDBCODI outlier markers, and black center dots for selected points.
4. click a point and confirm it is added to the active selection.
5. drag a rectangle and confirm every point inside the region is added to the active selection.
6. select more points and confirm they are added without replacing the existing selection.
7. change `n_clusters` and confirm cluster colors/assignments update while selection still uses the same point IDs.
8. save the active selection as a group, change selection, then restore the group.
9. open `/workflows/analysis-selection/api/state` and confirm dataset, feature matrix, projection, outliers, clusters, selection, and selection context are all present.

Step 5 currently uses real selection debug state and in-memory labeling state.
For Step 5 browser checks, confirm:

1. `/modules/labeling/` shows current selected and unselected point context.
2. assigning selected points to `cluster_2` creates a cluster annotation.
3. marking selected points as outliers creates an outlier annotation.
4. structured feedback JSON updates after each annotation.
5. `/modules/labeling/api/state` returns annotation history and structured feedback.
6. `/workflows/selection-labeling/` shows selection context beside labeling output.

Step 1-5 combined workflow check:

1. open `/workflows/analysis-labeling/`.
2. use the same dataset dropdown and `n_clusters` input as `/workflows/analysis-selection/`.
3. confirm one SVG shows projected points, cluster colors, SSDBCODI outlier markers, and black center dots for selected points.
4. click a point or drag a rectangle to add points to the active selection.
5. assign the selected points to a cluster from the labeling panel.
6. mark selected points as outliers.
7. confirm annotation history and structured feedback JSON update on the same page.
8. open `/workflows/analysis-labeling/api/state` and confirm dataset, feature matrix, projection, outliers, clusters, selection, selection context, groups, and labeling are all present.

Current Step 1-5 labeling rules:

1. allowed labels on `/workflows/analysis-labeling/` are the current cluster labels, such as `cluster_1`, `cluster_2`, `cluster_3`, plus `outlier`.
2. assigning a point to `cluster_N` updates the effective cluster state and removes that point from the effective outlier set.
3. assigning a point to `outlier` updates the effective outlier state and removes that point from effective cluster assignments.
4. the API keeps baseline provider outputs as `raw_clusters` and `raw_outliers`, label-aware provider outputs as `provider_clusters` and `provider_outliers`, and final display state as `clusters` and `outliers`.

Expected visual result:

1. assigning selected points to `cluster_N` changes their point color to that cluster color after reload.
2. assigning selected points to `outlier` changes them to the pink outlier marker after reload.
3. selected points remain indicated by small black center dots after labeling.
4. the data preview table shows `Effective Cluster`, `Effective Outlier`, and the manual label history for each affected point.

Step 6 currently uses the same wide-gap analysis fixture as the Step 1-5
workflow. For Step 6 browser checks, confirm:

1. `/modules/scatterplot/` shows the render payload and SVG plot.
2. clicking a point toggles selection through scatterplot's selection boundary.
3. dragging a rectangle adds all points inside the region through the same selection boundary.
4. saved selection groups can be created, restored, and deleted from the scatter workflows.
5. changing `n_clusters` reruns analysis and updates available cluster label options.
6. `/workflows/scatter-selection/` exposes projection, analysis, selection, selection groups, and render payload together.
7. `/workflows/scatter-labeling/` lets selected points become cluster or outlier labels through labeling.
8. `/workflows/scatter-labeling/api/state` includes baseline analysis, label-aware provider analysis, final effective analysis, selection groups, labeling state, and render payload.

### Step 6.5 Provider Diagnostics

`/modules/ssdbcodi/` is the standalone provider test page.
`/workflows/provider-feedback/` verifies the promoted `algorithm_adapters`
boundary beside standalone SSDBCODI scores.

Expected behavior:

1. `/workflows/provider-feedback/` reports `active_provider: ssdbcodi` through
   `algorithm_adapters`.
2. `/workflows/provider-feedback/api/state` includes both the adapter
   `AnalysisResult` and standalone `SsdbcodiResult`.
3. The standalone result includes seed records and per-point `rScore`,
   `lScore`, `simScore`, and `tScore`.
4. Adapter results continue to use `ClusterResult`, `OutlierResult`, and
   `AnalysisResult`.

Standalone SSDBCODI behavior:

1. `GET /modules/ssdbcodi/` returns a preview but does not append run history.
2. `POST /modules/ssdbcodi/api/run` stores a run and exposes scores through
   `/api/scores` and `/api/result`.
3. Default bootstrap uses density-safe KMeans center seeds and should not
   promote the demo fixture's far outliers as normal seeds. Bootstrap seeds
   remain active after manual labels unless the same point is explicitly
   relabeled or marked outlier.
4. Click selection and rectangle selection add to the active selection through
   the `selection` module without forcing a full page refresh or jumping back
   to the top of the page.
5. Selected points use the same black center-dot marker as the Step 1-6 pages.
6. Saved selection groups can be created, restored, and deleted.
7. Labeling controls are limited to `cluster_1...cluster_n` plus `outlier` and
   send actions through the `labeling` module.
8. Apply Label only saves pending feedback; Run and Store re-runs and stores
   SSDBCODI, and `rScore`, `lScore`, `simScore`, and `tScore` remain visible
   per point.
9. The dataset dropdown can switch among `demo`, `moons`, and `circles`; each
   dataset keeps separate selection, label, and result state.

## 4. Allowed Alternate Build Path

The recommended order is safe, but it is not mandatory.

It is allowed to build `chatbox` and `intent_instruction` early if they use mock context.

Allowed early path:

```text
dashboard_shell
  -> intent_instruction with mock selection and label context
  -> chatbox with mock selection and label context
  -> chat-intent workflow
```

Rules for this alternate path:

1. Chatbox must not call clustering.
2. Chatbox must not call outlier detection.
3. Chatbox must not call metric learning.
4. Intent module must output structured instructions only.
5. Mock selection and label context must be clearly labeled in the Flask page.
6. When the real selection and labeling modules exist, the mock context should be replaceable through a small boundary.

It is also allowed to build `labeling` early after `data_workspace` if it uses mock selection context. The labeling page must clearly mark whether selected point IDs are mock or real.

## 5. Mock vs Real Dependency Rule

If a module depends on a module that is not built yet, use a fixture or mock provider.

The page must display the dependency mode:

```text
dependency_mode: mock
```

or

```text
dependency_mode: real
```

Examples:

1. Chatbox before selection exists:
   - use mock selected/unselected IDs.

2. Labeling before selection exists:
   - use mock selected IDs and clearly mark them as mock.

3. Scatterplot before algorithm adapters exist:
   - use mock cluster and outlier assignments.

4. Refinement orchestrator before real metric learning exists:
   - use mock metric update output.

## 6. Workflow Page Standard

Each workflow page should show:

1. which modules are included.
2. which dependencies are real.
3. which dependencies are mocked.
4. visible inputs.
5. visible outputs.
6. JSON payloads passed between modules.
7. a clear success or failure state.

## 7. Completion Rule

A module can move from standalone to integrated only when:

1. its standalone page works.
2. its API route tests pass.
3. its dependency mode is clear.
4. its first workflow with previous modules works.
