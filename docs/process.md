# Development Process

## 1. Core Rule

Build the project as a local Flask app from the beginning.

Every module must be independently testable in code and independently visible in Flask.

Unit tests are required, but they are not enough. For a dashboard project, each module also needs a browser-visible debug page that shows whether the module is behaving correctly.

## 2. Standard Module Development Loop

For every module, follow this loop:

1. Read:
   - `README.md`
   - `docs/overview.md`
   - `docs/flask_app.md`
   - `docs/workflows.md`
   - `docs/modules/<module_name>/design.md`
   - `docs/module_debug_checklist.md`
   - `docs/integration_testing.md`
   - `docs/state_and_api_contracts.md`

2. Define contracts:
   - schemas
   - service inputs
   - service outputs
   - Flask API responses

3. Build pure module logic:
   - `service.py`
   - `schemas.py`
   - small helpers

4. Add fixtures:
   - predictable sample data
   - classic datasets when helpful
   - mock downstream outputs when the real downstream module is not ready

5. Add unit tests:
   - valid inputs
   - invalid inputs
   - edge cases
   - deterministic behavior

6. Add Flask module page:
   - `routes.py`
   - `templates/<module_name>/index.html`
   - `static/<module_name>/...` if needed

7. Add Flask API endpoints:
   - health endpoint
   - state endpoint
   - action endpoint if interactive

8. Add Flask smoke tests:
   - module page returns 200.
   - module API returns expected JSON.

9. Open the page locally:
   - inspect the visible output.
   - confirm the page communicates what is real and what is mocked.

10. Update docs:
   - design notes
   - route list
   - manual check instructions

## 3. Definition of Done for a Module

A module is done only when all of these are true:

1. It has documented input and output contracts.
2. It has pure service logic that can be tested without Flask.
3. Unit tests pass.
4. It has a Flask debug page under `/modules/<module_name>/`.
5. It has at least one JSON/debug API endpoint.
6. Flask route smoke tests pass.
7. A developer can open the module page in the browser and visually inspect behavior.
8. The module does not import unrelated module internals.
9. Any integration with other modules happens through schemas, services, APIs, or orchestrated workflows.
10. Its design document is updated.

## 4. Global Test Commands

Use these during development:

```powershell
python -m compileall app tests
python -m unittest discover -s tests
python run.py
```

After `python run.py`, manually inspect the relevant pages:

```text
http://127.0.0.1:5000/
http://127.0.0.1:5000/modules/
http://127.0.0.1:5000/modules/<module_name>/
```

## 5. Build Order

### Step 0: Flask App Shell and Module Lab

Build first:

```text
dashboard_shell
module registry
module lab index
base templates
base CSS
```

Why:

Every later module needs a place to appear visually. The Flask app shell should exist before module work continues.

Tasks:

1. Create `run.py`.
2. Create `create_app()`.
3. Create `/`, `/health`, `/modules/`, and `/workflows/`.
4. Create module registry.
5. Create a simple base template and shared CSS.
6. Add a placeholder card for every planned module.

Unit tests:

1. App factory returns a Flask app.
2. `/health` returns OK.
3. `/modules/` returns 200.

Flask visual check:

Open `/modules/` and confirm all module cards are visible.

Completion:

The local Flask app runs and can host module pages.

---

### Step 1: Data Workspace

Build:

```text
shared schemas
data_workspace
data workspace Flask page
```

Why:

All later modules need stable point IDs, feature vectors, and dataset state.

Tasks:

1. Define `Point`, `Dataset`, and `FeatureMatrix`.
2. Normalize raw input into point objects.
3. Generate stable IDs if missing.
4. Create classic dataset fixtures, such as Iris sample.
5. Add `/modules/data-workspace/`.
6. Add `/modules/data-workspace/api/dataset`.
7. Show dataset table and feature matrix preview in Flask.

Unit tests:

1. Load fixture data.
2. Reject empty input.
3. Reject missing features.
4. Reject duplicate IDs.
5. Preserve point order.
6. Return valid feature matrix.

Flask visual check:

Open `/modules/data-workspace/` and confirm:

1. point IDs are visible.
2. feature names are visible.
3. feature matrix preview matches the fixture.
4. JSON API link works.

Completion:

The dataset can be inspected in the browser and consumed by later modules.

---

### Step 2: Projection

Build:

```text
projection
MDS service
projection Flask page
data-projection workflow page
```

Why:

The scatterplot needs 2D coordinates. Projection should be visually inspected before scatterplot work.

Tasks:

1. Implement MDS behind a service function.
2. Input `FeatureMatrix`.
3. Output `ProjectionResult`.
4. Add `/modules/projection/`.
5. Add `/modules/projection/api/projection`.
6. Render an SVG scatterplot from fixture data.
7. Add `/workflows/data-projection/` to show data table and projection together.

Unit tests:

1. Every input point gets one coordinate.
2. Coordinates are finite.
3. Projection is deterministic for fixed input.
4. Invalid matrix is rejected.

Flask visual check:

Open `/modules/projection/` and confirm:

1. points appear in SVG.
2. coordinate table is visible.
3. Iris sample visually separates in a plausible way.
4. JSON API returns projection coordinates.

Completion:

Projection can be inspected through Flask without scatterplot, clustering, or chatbox.

---

### Step 3: Algorithm Adapters

Build:

```text
algorithm_adapters
clustering adapter
outlier adapter
adapter Flask page
default-analysis workflow page
```

Why:

The app must show default clusters and outliers, but existing algorithms should stay isolated behind adapters.

Tasks:

1. Wrap existing clustering algorithm.
2. Wrap existing outlier detection algorithm.
3. Convert dashboard schemas to algorithm input.
4. Convert algorithm output back to point-ID-based schemas.
5. Add `/modules/algorithm-adapters/`.
6. Add APIs for cluster and outlier output.
7. Show adapter diagnostics in Flask.
8. Add `/workflows/default-analysis/`.
9. Keep algorithm implementation behind a provider boundary so future algorithms can replace the current default.

Current implementation:

1. `run_default_analysis()` now uses `SsdbcodiProvider`.
2. SSDBCODI emits cluster assignments and outlier flags in one provider run.
3. The legacy `SequentialLofThenKMeansProvider` remains available explicitly.
4. `n_clusters` can be adjusted through the Flask page or query string.
5. The `ssdbcodi` module also exists as a dedicated debug page. It preserves
   dashboard-facing `ClusterResult` / `OutlierResult` schemas while exposing
   paper-aligned intermediate scores and selection/labeling integration. Its
   debug page keeps bootstrap seeds active across runs (manual cluster labels
   are the only per-point output locks; bootstrap seed points' final
   `cluster_id` may shift under the weighted-distance rule), separates
   pending label entry from Run and Store execution, and includes demo/moons/
   circles fixtures for shape-specific testing.
6. The debug fixture is `default_analysis_outlier_debug`, which intentionally contains visible outlier candidates.

Unit tests:

1. Adapter passes expected matrix or representation to algorithms.
2. Cluster assignments map to known point IDs.
3. Outlier scores map to known point IDs.
4. Invalid algorithm output is rejected.
5. Outliers are excluded before clustering.
6. `n_clusters` changes the requested cluster count.

Flask visual check:

Open `/modules/algorithm-adapters/` and confirm:

1. cluster assignments are visible.
2. outlier scores are visible.
3. diagnostics show which algorithm is being called.
4. execution order is clearly shown as outlier detection before clustering.
5. changing `n_clusters` updates SSDBCODI bootstrap cluster output.
6. the page explains the current provider and future algorithm slot.

Open `/workflows/default-analysis/` and confirm:

1. projection, outliers, and clusters are visible together.
2. outliers are visually distinct.
3. JSON payloads for outliers and clusters are inspectable.

Completion:

Default cluster and outlier results can be inspected in Flask.

---

### Step 4: Selection

Build:

```text
selection
selection store
selection Flask page
```

Why:

Selection is the grounding layer for phrases like "these points" and "unselected points".

Tasks:

1. Store selected point IDs.
2. Support select, deselect, replace, and clear.
3. Return selected and unselected IDs.
4. Add `/modules/selection/`.
5. Add selection API endpoints.
6. Build a clickable point list in Flask.
7. Add `/workflows/selection-context/`.
8. Preserve action `source`, `mode`, and metadata fields for future selection gestures.
9. Support named selection groups so a user can save the current point set and restore it later.

Current implementation:

1. Supports `select`, `deselect`, `replace`, `toggle`, and `clear`.
2. Supports action sources such as `api`, `point_click`, `lasso`, `rectangle`, `manual_list`, and `workflow_fixture`.
3. Stores local debug state in memory.
4. Exposes selection state and downstream selection context separately.
5. `/workflows/selection-context/` shows Data Workspace point IDs converted into selection context.
6. Saves reusable named selection groups.
7. Selecting a saved group replaces the active selection with that group's point IDs.

Unit tests:

1. Select known points.
2. Reject unknown IDs.
3. Clear selection.
4. Return selected/unselected IDs.
5. Deselect, replace, and toggle points.
6. Preserve future gesture metadata.
7. Save, select, and delete named selection groups.
8. Reject duplicate selection group names.

Flask visual check:

Open `/modules/selection/` and confirm:

1. clicking points changes selection state.
2. selected/unselected JSON updates.
3. clear selection works.
4. manual action lab can apply select, deselect, replace, toggle, and clear.
5. supported action/source/mode values are visible.
6. saving the current selection creates a visible named group.
7. clicking a saved group name restores that point selection.
8. deleting a saved group removes it without changing the active selection.

Open `/workflows/selection-context/` and confirm:

1. Data Workspace point IDs are visible.
2. selection context JSON is visible.
3. selected and unselected counts match the module page.

Open `/workflows/analysis-selection/` and confirm:

1. data, projection, algorithm results, and selection use the same point IDs.
2. the dataset dropdown can switch between a sparse selection-friendly fixture and the original outlier debug fixture.
3. clicking projected points adds them to the active selection and shows black center dots.
4. rectangle selection adds all points inside the region to the active selection.
5. outliers remain visually distinct from selected points.
6. changing `n_clusters` reruns clustering without breaking selection.
7. saved selection groups can be restored from the combined workflow page.

Completion:

Selection can be exercised in Flask without labeling, scatterplot, or chatbox.

---

### Step 5: Labeling

Build:

```text
labeling
labeling Flask page
selection-labeling workflow page
```

Why:

Selection only says which points are active. Labeling says what those selected points mean: same class, assigned cluster, outlier, or not outlier.

Tasks:

1. Define manual annotation and structured feedback schemas.
2. Accept selection context from the selection module.
3. Support assigning selected points to an existing cluster.
4. Support creating a new class label for selected points.
5. Support marking selected points as outliers or not outliers.
6. Add `/modules/labeling/`.
7. Add `/modules/labeling/api/state`.
8. Add `/modules/labeling/api/apply`.
9. Add `/workflows/selection-labeling/`.
10. Add `/workflows/analysis-labeling/` to test Steps 1-5 on one visual layer.
11. Show annotation history and structured feedback JSON in Flask.

Current implementation:

1. Uses real selection debug state as input.
2. Stores local manual annotation history in memory.
3. Supports `assign_cluster`, `assign_new_class`, `mark_outlier`, and `mark_not_outlier`.
4. Converts manual annotations into structured feedback instructions.
5. `/workflows/selection-labeling/` shows selection context and labeling output together.
6. `/workflows/analysis-labeling/` shows data, projection, outliers, clusters, selection, and labeling together.

Unit tests:

1. selected points become `assign_cluster`.
2. selected points become `is_outlier`.
3. empty selection is rejected.
4. unknown point IDs are rejected.
5. reset clears annotation state.
6. unselected explicit point IDs are rejected.

Flask visual check:

Open `/modules/labeling/` and confirm:

1. selected point IDs are visible.
2. assigning a cluster creates an annotation.
3. marking outliers creates an annotation.
4. structured feedback JSON is visible.
5. dependency mode clearly says mock or real selection.

Open `/workflows/selection-labeling/` and confirm:

1. selection context JSON is visible.
2. structured feedback JSON is visible.
3. annotation history matches labels created in `/modules/labeling/`.

Open `/workflows/analysis-labeling/` and confirm:

1. one SVG shows projected points, cluster colors, SSDBCODI outliers, and selected points.
2. click selection and rectangle selection add points to the active selection.
3. labeling controls only allow `cluster_1...cluster_n` and `outlier`.
4. assigning `cluster_N` updates effective cluster state and frontend point colors.
5. assigning `outlier` updates effective outlier state and frontend outlier markers.
6. structured feedback JSON updates on the same page.
7. `/workflows/analysis-labeling/api/state` includes Step 1-5 state in one payload.

Completion:

Manual labels can be created from selected points and inspected in Flask before chatbox or metric learning exists.

---

### Step 6: Scatterplot

Build:

```text
scatterplot
scatterplot Flask page
scatter-selection workflow page
scatter-labeling workflow page
```

Why:

After data, projection, adapter output, selection, and labeling exist, scatterplot can combine them visually.

Tasks:

1. Render projected points.
2. Color points by cluster.
3. Mark outliers.
4. Support click selection and rectangle selection through the selection module.
5. Render manual labels when label state exists.
6. Send label actions to labeling when that workflow is active.
7. Add `/modules/scatterplot/`.
8. Add `/workflows/scatter-selection/`.
9. Add `/workflows/scatter-labeling/`.
10. Preserve saved selection groups from Step 4 in scatter workflows.
11. Preserve adjustable `n_clusters` from Step 3 in scatter workflows.

Unit tests:

1. Build render payload correctly.
2. Preserve point IDs.
3. Mark selected points correctly.
4. Include cluster, outlier, and manual label fields.
5. Verify rectangle selection preserves `source: "rectangle"`.
6. Verify selection groups can be saved, restored, and deleted from scatter workflows.
7. Verify `n_clusters` changes cluster label options in scatter-labeling.

Flask visual check:

Open `/modules/scatterplot/` and confirm:

1. points render.
2. cluster colors are visible.
3. outlier markers are visible.
4. clicking points updates selection state.
5. dragging a rectangle adds points inside the region to selection.
6. saving, restoring, and deleting a selection group works.
7. changing `n_clusters` reruns analysis and updates label options.
8. label actions update labeling state when using the labeling workflow.

Completion:

The scatterplot module can be visually tested before chatbox work, and the
Step 1-6 workflow preserves every completed upstream interaction that it
integrates: algorithm controls, selection gestures, selection groups, and manual
labeling.

---

### Step 6.5: Provider Feedback Diagnostics

Build:

```text
ssdbcodi
ssdbcodi Flask page
provider-feedback workflow page
selection + labeling integration
per-point score persistence
```

Why:

SSDBCODI is the active default provider behind `algorithm_adapters`. Keeping it
as a separate registered module lets the team inspect scores and feedback
behavior interactively while the adapter boundary remains stable.

Current implementation:

1. Computes `cDist`, `rDist`, `lScore`, `simScore`, and `tScore` from the
   paper, plus `rScore = exp(-min rDist to any seed)` (a simplified
   nearest-seed reachability instead of the paper's Prim back-trace `Emax`).
   Class assignment uses a custom weighted-distance rule
   `score(p, c) = w * rDistNorm(p, nearest_seed_of_c)
                + (1 - w) * euclDistNorm(p, nearest_seed_of_c)`
   with `w = rscore_weight` (default 0.5, user-configurable). Back-trace,
   the random-forest classifier, and the local smoothing pass from the
   paper are not used.
2. Bootstrap: density-safe KMeans seeds, centroid-nearest points promoted to
   normal seed inputs. Bootstrap seeds remain available as seeds across
   runs; only manual cluster annotations lock the final `cluster_id` of the
   labeled point.
3. Debug page at `/modules/ssdbcodi/` with `demo`, `moons`, `circles` fixtures.
4. Uses existing `selection` and `labeling` stores scoped per dataset.
5. Pending labels are separate from Run & Store: `POST /api/label` saves
   feedback; `POST /api/run` recomputes and persists results.
6. Per-point scores persisted in `SsdbcodiStore` for downstream metric-learning
   consumption.
7. Output schemas reuse `ClusterResult` / `OutlierResult` from shared schemas.
8. The `SsdbcodiProvider` implements the `AnalysisProvider` protocol and backs
   `algorithm_adapters.run_default_analysis()` by default.
9. `/workflows/provider-feedback/` compares the adapter-facing
   `AnalysisResult` with standalone `SsdbcodiResult` score diagnostics.

Provider boundary:

- `algorithm_adapters` already defines the `AnalysisProvider` protocol.
- The default provider is `SsdbcodiProvider`; `SequentialLofThenKMeansProvider`
  remains as an explicit legacy provider for comparison.
- All downstream code (scatterplot, workflows, metric-learning) continues to
  work because the output schemas are unchanged.

---

### Step 7: Chatbox

Build:

```text
chatbox
chatbox Flask page
mock selection, label, and instruction context
refinement strategy selector (metric_learning | direct_ssdbcodi)
```

Why:

Chatbox needs selection context and may benefit from recent label context, but should not own selection, labeling, algorithms, or the structured instruction state. It also exposes the refinement strategy selector so the user can choose Path A or Path B per refinement without touching orchestrator internals.

Tasks:

1. Build chat UI.
2. Display current selection context and selection groups.
3. Display recent manual label context when available.
4. Display the current `StructuredInstruction` panel (read from intent instruction).
5. Display suggestion chips derived from dataset context and the active strategy. Chips for `split_cluster` and `reclassify_outlier` are only shown when strategy is `direct_ssdbcodi`.
6. Submit user message with a truncated history window (default last 3 turns) plus context.
7. Show assistant response including router category.
8. Expose a refinement strategy toggle near the input.
9. Add `/modules/chatbox/`.
10. Add APIs for message submission, context, history, strategy selection, and reset.
11. Support mock selection, label, and instruction context for standalone testing.

Unit tests:

1. Empty messages are rejected.
2. Message payload includes selection context and selection groups.
3. Message payload includes label context when available.
4. Message payload includes the active refinement strategy.
5. History window is truncated to the configured N turns.
6. Chatbox does not call clustering or outlier detection.
7. Chatbox does not mutate selection, labeling, or structured instruction state.
8. Strategy toggle changes the strategy attached to subsequent messages but does not mutate instruction state.

Flask visual check:

Open `/modules/chatbox/` and confirm:

1. chat input works and messages appear in history.
2. selection and label context are visible.
3. strategy selector is visible and switchable.
4. suggestion chips produce valid intents when clicked; `split_cluster`/`reclassify_outlier` chips only appear under `direct_ssdbcodi`.
5. the `StructuredInstruction` preview panel updates after actionable messages.
6. response clearly shows whether the LLM provider is real or mocked.

Completion:

Chatbox can be manually tested in Flask with mock selection, label, and instruction context.

---

### Step 8: Intent Instruction

Build:

```text
intent_instruction
router + extractor + LLM provider protocol
intent Flask page
chat-intent workflow page
```

Why:

User language must become structured instructions before metric learning is touched. The module must be robust to off-topic, ambiguous, and partial messages.

Tasks:

1. Define `StructuredInstruction` schema and `InstructionDelta` schema.
2. Implement two-stage pipeline: router first, extractor only on actionable messages.
3. Define `LlmProvider` protocol; implement `MockLlmProvider` for tests and `LocalQwenProvider` (qwen2.5-14b via Ollama or vLLM) as default runtime. Cloud providers (`ClaudeProvider`, `OpenAIProvider`) slot into the same protocol.
4. Prompts use JSON-schema constrained output so small models can produce valid deltas.
5. Resolve group references (`selected_points`, `selection_group`, `cluster`, `outlier_set`, `point_id`).
6. Generate clarification prompts for ambiguous and partial messages.
7. Emit all eight Phase 1 intents: `feature_weight`, `group_similar`, `group_dissimilar`, `merge_clusters`, `anchor_point`, `ignore_cluster`, `split_cluster`, `reclassify_outlier`. Path-specific acceptance is enforced downstream by the adapters, not by the extractor.
8. Forward only the last N turns (default 3) plus the current instruction snapshot to the LLM, not full chat history.
9. Add `/modules/intent-instruction/` with route, compile, state, reset, and examples APIs.
10. Add `/workflows/chat-intent/`.

Unit tests (router):

1. Off-topic messages like "today's weather" become `off_topic`.
2. Meta queries like "how many clusters are there" become `meta_query`.
3. "move these together" with empty selection becomes `on_topic_ambiguous`.
4. Clear actionable messages become `on_topic_actionable`.

Unit tests (extractor, with MockLlmProvider):

1. Grouping messages become `group_similar` deltas.
2. Separating messages become `group_dissimilar` deltas.
3. Merge messages become `merge_clusters` deltas.
4. Feature-importance messages become `feature_weight` deltas.
5. Anchor references become `anchor_point` deltas.
6. Ignore-cluster messages become `ignore_cluster` deltas.
7. Split-cluster messages become `split_cluster` deltas.
8. Reclassify-outlier messages become `reclassify_outlier` deltas.
9. Applying a delta to an `StructuredInstruction` produces the expected next state.

Flask visual check:

Open `/modules/intent-instruction/` and confirm:

1. example messages grouped by intent can be submitted.
2. router category and confidence are visible.
3. delta JSON and resulting `StructuredInstruction` state are both visible.
4. clarification cases are clear and do not mutate state.
5. active provider (mock, local qwen, cloud) is clearly labeled.

Completion:

Intent parsing is visible and debuggable before metric-learning integration, and the LLM provider is swappable through settings.

---

### Step 9A: Metric-Learning Adapter (Path A)

Build:

```text
metric_learning_adapter
constraint_builder (pure)
MetricLearnerProvider protocol (IdentityProvider, ItmlProvider)
constraint preview Flask page
```

Why:

Path A turns structured feedback into a learned Mahalanobis metric. Pair constraints from labeling and chat are collected in a single `ConstraintSet`, fed to ITML, and the resulting `L = chol(M)` is applied as a linear pre-transform to the feature matrix so projection and algorithm adapters can be reused unchanged. When SSDBCODI is the active provider, its per-point `tScore` values are available as auxiliary signal for constraint weighting.

Tasks:

1. Build `constraint_builder` as a pure function module:
   - Labeling `assign_cluster` annotations become intra-label must-link pairs.
   - `group_similar`, `group_dissimilar`, `merge_clusters`, `anchor_point` become sampled pair constraints (bounded by `max_pairs_per_intent`, default 50).
   - `feature_weight` populates a `feature_scale` dict, not pair lists.
   - `ignore_cluster` excludes that cluster from all pair generation.
   - `split_cluster` and `reclassify_outlier` are rejected with `intent_deferred` and a `suggested_strategy: "direct_ssdbcodi"` hint pointing to Path B.
   - Detect conflicting must-link / cannot-link pairs and report them.
2. Define `MetricLearnerProvider` protocol and implement:
   - `IdentityProvider` - returns `M = I`, used for cold start or empty constraints.
   - `ItmlProvider` - wraps `metric-learn` ITML, accepts must-link / cannot-link pairs and `feature_scale` via pre-scaling of `X`.
3. Output a `LearnedMetric` containing `M`, `L = chol(M)`, provider name, constraint count, diagnostics.
4. Applying the metric is a linear pre-transform: `X' = X · L` (with `feature_scale` folded in).
5. Add `/modules/metric-learning-adapter/` with `constraints`, `fit`, and `providers` APIs.
6. Show instruction input, annotation input, `ConstraintSet`, and learned `M` preview.
7. Add `/workflows/instruction-constraints/` to show the full Path A constraint preview with real selection and labeling state.

Unit tests:

1. `group_similar` produces must-link pairs bounded by sampling cap.
2. `group_dissimilar` produces cannot-link pairs.
3. `merge_clusters` produces cross-cluster must-link pairs.
4. `feature_weight` populates `feature_scale`, not pair lists.
5. `anchor_point` produces must-link pairs from anchor to every target.
6. `ignore_cluster` excludes that cluster's points.
7. `split_cluster` and `reclassify_outlier` return `intent_deferred` with a Path B hint.
8. Conflicting must-link/cannot-link pairs are reported.
9. Labeling `assign_cluster` annotations merge with chat-derived constraints into the same `ConstraintSet`.
10. `IdentityProvider.fit` returns `M = I`.
11. `ItmlProvider.fit` with similar pairs reduces their distance under the learned metric.

Flask visual check:

Open `/modules/metric-learning-adapter/` and confirm:

1. sample instruction plus sample annotations produce a visible `ConstraintSet` with pair count and conflict list.
2. the active provider is visible and switchable.
3. fit produces a visible `M` preview and transformed feature matrix preview.
4. a `split_cluster` instruction produces a clear "intent deferred" error that names Path B.

Completion:

Path A feedback compilation can be inspected before the refinement loop, and the learned metric is usable as a pre-transform by projection and algorithm adapters.

---

### Step 9B: Direct Feedback Adapter (Path B)

Build:

```text
direct_feedback_adapter
plan_builder (pure)
DirectFeedbackPlan schema
direct feedback preview Flask page
```

Why:

Path B turns the same structured feedback into SSDBCODI-native inputs instead of a metric. SSDBCODI is semi-supervised and accepts seeds, `n_clusters`, contamination, and feature scales directly, so feedback can drive the algorithm without learning a Mahalanobis metric. Keeping Path B as its own adapter (and its own orchestrator in Step 10B) makes each strategy independently testable and keeps their error codes and debug pages clean.

Tasks:

1. Build `plan_builder` as a pure function module that compiles `StructuredInstruction` plus labeling annotations into a `DirectFeedbackPlan` (seed updates, `feature_scale`, `param_overrides`, `excluded_clusters`, `merged_cluster_groups`).
2. Intent-to-plan mapping:
   - `feature_weight` → `feature_scale` entry.
   - `group_similar` → seed updates with a shared `cluster_id`.
   - `group_dissimilar` → seed updates with distinct `cluster_id` values.
   - `merge_clusters` → `merged_cluster_groups` entry and relabeled seeds.
   - `anchor_point` → single seed update for the anchor.
   - `ignore_cluster` → add to `excluded_clusters`.
   - `split_cluster` → increment `param_overrides.n_clusters` and add interior seeds (Path B-native, not deferred).
   - `reclassify_outlier` → seed update with `is_outlier: true`/`false` (Path B-native, not deferred).
3. Labeling annotations override conflicting chat-derived seeds on the same point.
4. Detect conflicts (point assigned to two clusters, point assigned and marked outlier, excluded cluster that is also referenced elsewhere, etc.).
5. Add `/modules/direct-feedback-adapter/` with `plan` and `preview` APIs.
6. Add `/workflows/instruction-ssdbcodi/` to show the full Path B plan preview with real selection and labeling state.

Unit tests:

1. `feature_weight` populates `feature_scale`, not seeds.
2. `group_similar` produces `seed_updates` with a shared `cluster_id`.
3. `group_dissimilar` produces `seed_updates` with distinct `cluster_id`s.
4. `merge_clusters` produces a `merged_cluster_groups` entry and relabels seeds.
5. `anchor_point` produces a single seed update for the anchor.
6. `ignore_cluster` adds the cluster to `excluded_clusters`.
7. `split_cluster` increments `n_clusters` in `param_overrides` and adds interior seeds.
8. `reclassify_outlier` produces a seed update with the correct `is_outlier` flag.
9. Labeling `assign_cluster` annotations override conflicting chat-derived seeds on the same point.
10. Contradictory assignments are reported in `conflicts`.

Flask visual check:

Open `/modules/direct-feedback-adapter/` and confirm:

1. sample instruction plus sample annotations produce a visible `DirectFeedbackPlan`.
2. a `split_cluster` instruction is accepted (no `intent_deferred` error) and shows up in `param_overrides`.
3. a `reclassify_outlier` instruction appears as an outlier seed update.
4. contradictory seed assignments are listed in the conflicts panel.

Completion:

Path B feedback compilation can be inspected before the refinement loop, and all Phase 1 intents (including `split_cluster` and `reclassify_outlier`) produce valid plans.

---

### Step 10A: Metric Refinement Orchestrator (Path A)

Build:

```text
metric_refinement_orchestrator
Path A refinement history + rollback
Path A refinement timeline Flask page
```

Why:

Path A's orchestrator coordinates modules for the metric-learning update strategy but should not contain their internal logic. It also records each run so users can inspect and revert changes. Keeping this separate from the Path B orchestrator keeps each strategy's step list, error codes, and history easy to debug.

Tasks:

1. Accept a refinement trigger with `strategy: "metric_learning"` from labeling, intent instruction, or a manual refine button.
2. Call `metric_learning_adapter.build_constraints`, then `fit`.
3. Apply the returned `L` to the feature matrix.
4. Trigger updated projection on the transformed matrix.
5. Rerun clustering and outlier detection through algorithm adapters on the transformed matrix.
6. Record each completed run (constraints, learned metric, downstream run IDs, instruction version) in this orchestrator's own history.
7. Support rollback to a prior run.
8. Reject triggers that contain `split_cluster` or `reclassify_outlier` intents with a clear `intent_deferred` error and a pointer to Path B.
9. Add `/modules/metric-refinement-orchestrator/` with run, history, rollback, and reset APIs.
10. Add `/workflows/metric-refinement-loop/`.

Unit tests:

1. Actionable instruction triggers the flow in the expected step order.
2. Empty instruction plus empty annotations produces an identity-metric run (no-op visible as a run).
3. `split_cluster` / `reclassify_outlier` intent returns `intent_deferred` with a Path B suggestion before metric fit.
4. Metric-fit failure returns diagnostics and leaves prior active run untouched.
5. Projection failure returns diagnostics and preserves prior run.
6. History is appended only on success.
7. Rollback restores a prior run without recomputation.
8. Path A reset clears only Path A history.

Flask visual check:

Open `/modules/metric-refinement-orchestrator/` and confirm:

1. timeline shows each step.
2. intermediate payloads (constraint set, metric metadata) are visible.
3. failure and success states are understandable.
4. history list shows prior runs and offers rollback buttons.
5. deferred-intent errors name the intent clearly and suggest Path B.

Completion:

The Path A update loop can be debugged visually, and every successful Path A run is reversible before full dashboard integration.

---

### Step 10B: Direct Refinement Orchestrator (Path B)

Build:

```text
direct_refinement_orchestrator
Path B refinement history + rollback
Path B refinement timeline Flask page
```

Why:

Path B's orchestrator coordinates a different update sequence: `direct_feedback_adapter` builds a plan, `feature_scale` is applied as `X' = X · S`, SSDBCODI re-runs with merged seeds and param overrides, and projection is rerun only when feature geometry changed. The two orchestrators are intentionally separate so each strategy has its own step list, own history, and own debug surface.

Tasks:

1. Accept a refinement trigger with `strategy: "direct_ssdbcodi"` from labeling, intent instruction, or a manual direct-refine button.
2. Call `direct_feedback_adapter.build_plan` to produce a `DirectFeedbackPlan`.
3. Apply `feature_scale` (diagonal pre-scale) to the feature matrix when present.
4. Merge `seed_updates` with existing bootstrap seeds and labeling outlier overrides.
5. Call `algorithm_adapters.run_default_analysis` with the merged seeds and `param_overrides` so SSDBCODI re-runs with the new configuration.
6. Trigger an updated projection on the transformed matrix when `feature_scale` changed; reuse projection when only seeds or `n_clusters` changed.
7. Record each completed run in this orchestrator's own history.
8. Support rollback to a prior Path B run.
9. Add `/modules/direct-refinement-orchestrator/` with run, history, rollback, and reset APIs.
10. Add `/workflows/direct-refinement-loop/`.

Unit tests:

1. Actionable instruction triggers the Path B step order.
2. `split_cluster` intent produces a plan with `n_clusters += 1` and new seeds; the run completes without an `intent_deferred` error.
3. `reclassify_outlier` intent flips the point's outlier state on the downstream analysis output.
4. `feature_weight` intent populates `feature_scale` and the feature matrix is pre-scaled before SSDBCODI.
5. Seeds from `anchor_point` intent are merged with bootstrap seeds.
6. `ignore_cluster` intent removes that cluster's seeds from the merged set for this run.
7. `merge_clusters` intent relabels seeds of the absorbed cluster(s).
8. Projection is reused when only seed updates change and feature geometry is unchanged.
9. SSDBCODI run failure returns diagnostics and preserves prior run.
10. History is appended only on success.
11. Rollback restores a prior Path B run without recomputation.
12. Path B reset clears only Path B history.

Flask visual check:

Open `/modules/direct-refinement-orchestrator/` and confirm:

1. timeline shows the Path B step list with `plan_build`, `feature_scale`, `ssdbcodi_run`, `projection`, `effective_analysis`.
2. `DirectFeedbackPlan` preview is visible (seed updates, feature_scale, param overrides).
3. a `split_cluster` instruction produces a visible new cluster on the downstream analysis.
4. a `reclassify_outlier` instruction flips the target point's outlier state.
5. history list shows prior Path B runs and offers rollback buttons.

Completion:

The Path B update loop can be debugged visually, and `split_cluster` / `reclassify_outlier` produce visible changes to the analysis state without `intent_deferred` errors.

---

### Step 11: Strategy Comparison

Build:

```text
/workflows/strategy-comparison/
read-only comparison between Path A and Path B runs on the same feedback snapshot
```

Why:

Path A and Path B use the same structured feedback but different update mechanisms. The comparison workflow runs the same feedback through both orchestrators and renders their outputs side-by-side so the two strategies can be evaluated on the same data.

Tasks:

1. Snapshot current `StructuredInstruction` and labeling annotations.
2. Call `metric_refinement_orchestrator.run(strategy="metric_learning")` and `direct_refinement_orchestrator.run(strategy="direct_ssdbcodi")` on the snapshot.
3. Render both results on one page: two SVG plots side-by-side, two cluster/outlier tables, and key differences (e.g., which points changed cluster, which were reclassified as outliers).
4. Surface each run's timeline and any `intent_deferred` errors from Path A (so it's clear which intents Path B absorbed).
5. Do not own refinement history; read from both orchestrators' histories.

Unit tests:

1. Comparison endpoint runs both orchestrators and returns both timelines.
2. When the feedback contains `split_cluster`, Path A run reports `intent_deferred` and Path B run reports a new cluster.
3. Comparison payload includes per-point diff summary between the two outputs.

Flask visual check:

Open `/workflows/strategy-comparison/` and confirm:

1. both plots render using the same projection axes for fair visual comparison.
2. per-point diff table highlights points whose cluster assignment or outlier flag differs between the two paths.
3. Path A errors are clearly labeled rather than silently omitted.

Completion:

Developers can run the same feedback through both update strategies on one page and see which points differ between Path A and Path B.

---

### Step 12: Integrated Dashboard

Build:

```text
integrated dashboard
```

Why:

Only integrate after individual modules and workflow demos work.

Tasks:

1. Compose data, projection, adapters, selection, labeling, scatterplot, chatbox, intent, both Path A and Path B adapters, both orchestrators, and the strategy comparison workflow.
2. Keep dashboard shell thin.
3. Show current state clearly.
4. Make refinement loop visible, including the active strategy.

Unit tests:

1. Integrated route returns 200.
2. APIs return coherent state.
3. Irrelevant chat does not trigger refinement.
4. Actionable chat under `metric_learning` strategy triggers Path A flow.
5. Actionable chat under `direct_ssdbcodi` strategy triggers Path B flow.

Flask visual check:

Open `/` and confirm:

1. scatterplot appears.
2. selecting points updates selection and chatbox context.
3. direct label actions create structured feedback.
4. chat instruction produces structured output.
5. switching the strategy toggle routes the next refinement to the matching orchestrator.
6. valid instruction updates the visible state.

Completion:

The first complete local human-in-the-loop workflow works in Flask, with both update strategies independently exercisable and comparable.

## 6. What Not To Do Early

1. Do not add deployment infrastructure.
2. Do not add a heavy frontend framework.
3. Do not add a production database.
4. Do not let modules communicate through hidden globals.
5. Do not let chatbox call algorithms directly.
6. Do not let scatterplot own selection truth.
7. Do not let scatterplot or chatbox own label truth.
8. Do not rewrite existing clustering or outlier logic.
9. Do not skip browser-visible module pages.

## 7. Milestones

### Milestone 1: Local Module Lab

Goal:

Flask app runs and lists all modules under `/modules/`.

### Milestone 2: Data and Projection Visible

Goal:

Data workspace and projection can be opened in Flask, and `/workflows/data-projection/` shows their interaction.

### Milestone 2.5: Default Analysis Visible

Goal:

Algorithm adapters can be opened in Flask, SSDBCODI outputs are visible through the adapter schemas, and `/workflows/default-analysis/` shows data, projection, outliers, and clusters together.

### Milestone 3: Selection and Labeling

Goal:

Selection and labeling work in Flask, and selected points can become manual cluster/outlier annotations.

Current status:

Selection works in Flask, supports named selection groups, `/workflows/selection-context/` exposes reusable selected/unselected context, `/workflows/analysis-selection/` connects Steps 1-4 on one visual testing page, and `/workflows/analysis-labeling/` connects Steps 1-5 for full select-and-label testing.

### Milestone 4: Scatterplot Labeling

Goal:

Scatterplot renders projected points, default clusters, outliers, selection state, and manual label state.

Current status:

Scatterplot has a working module page, render-payload API, scatter-selection workflow, and scatter-labeling workflow. It renders state owned by previous modules and sends selection/label actions back through their module boundaries. Step 6 must preserve prior workflow capabilities when integrated: rectangle selection, saved selection groups, and adjustable cluster count are part of the acceptance check.

### Milestone 4.5: Provider Feedback Diagnostics

Goal:

SSDBCODI module works as a standalone debug page, uses the same selection/labeling stores as Step 4-5, persists per-point scores, implements the active `AnalysisProvider` default, and is visible through `/workflows/provider-feedback/`.

Current status:

Working. Debug page at `/modules/ssdbcodi/` with three fixtures, selection and labeling integration, Run/Store separation, and per-point score persistence.

### Milestone 5: Chat and Intent

Goal:

Chatbox receives selection/label context and intent module outputs structured instructions.

### Milestone 6A: Path A Refinement Loop

Goal:

Structured instruction flows through `metric_learning_adapter` and `metric_refinement_orchestrator`, producing a learned metric, a retransformed feature matrix, and rerun projection + SSDBCODI output.

### Milestone 6B: Path B Refinement Loop

Goal:

The same structured instruction flows through `direct_feedback_adapter` and `direct_refinement_orchestrator`, producing SSDBCODI-native input updates (seeds, `n_clusters`, feature scales, labeled outliers) and rerun SSDBCODI output. `split_cluster` and `reclassify_outlier` work end-to-end without `intent_deferred` errors.

### Milestone 6.5: Strategy Comparison

Goal:

`/workflows/strategy-comparison/` runs both paths on the same feedback snapshot and renders their analysis outputs side-by-side with a per-point diff.

### Milestone 7: Integrated Dashboard

Goal:

The full local dashboard supports at least one complete refinement cycle under each strategy.
