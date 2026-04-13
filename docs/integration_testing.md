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
| 5 | `labeling` | `/modules/labeling/` | `/workflows/selection-labeling/` |
| 6 | `scatterplot` | `/modules/scatterplot/` | `/workflows/scatter-selection/` and `/workflows/scatter-labeling/` |
| 7 | `chatbox` | `/modules/chatbox/` | `/workflows/chat-selection/` |
| 8 | `intent_instruction` | `/modules/intent-instruction/` | `/workflows/chat-intent/` |
| 9 | `metric_learning_adapter` | `/modules/metric-learning-adapter/` | `/workflows/instruction-constraints/` |
| 10 | `refinement_orchestrator` | `/modules/refinement-orchestrator/` | `/workflows/refinement-loop/` |
| 11 | integrated dashboard | `/` | full app |

Not every workflow needs to be polished. A workflow page can be simple and diagnostic as long as it shows the interaction clearly.

Step 3 currently uses a dedicated `default_analysis_outlier_debug` fixture for
the algorithm-adapter page and `/workflows/default-analysis/`. This fixture is
not Iris; it is intentionally shaped with compact clusters and distant outlier
candidates so Local Outlier Factor and KMeans can both be visually checked.

For Step 3 browser checks, confirm:

1. `/modules/algorithm-adapters/` shows Local Outlier Factor output.
2. KMeans assignments exclude detected outliers.
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
3. confirm one SVG shows projected points, cluster colors, LOF outlier markers, and thin selected-point rings.
4. click a point and confirm it is added to the active selection.
5. drag a rectangle and confirm every point inside the region is added to the active selection.
6. select more points and confirm they are added without replacing the existing selection.
7. change `n_clusters` and confirm cluster colors/assignments update while selection still uses the same point IDs.
8. save the active selection as a group, change selection, then restore the group.
9. open `/workflows/analysis-selection/api/state` and confirm dataset, feature matrix, projection, outliers, clusters, selection, and selection context are all present.

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
