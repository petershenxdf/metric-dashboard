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
http://127.0.0.1:5001/
http://127.0.0.1:5001/modules/
http://127.0.0.1:5001/modules/<module_name>/
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

Manual labels can be created from selected points and inspected in Flask before chatbox or rule interpretation exists.

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
6. Per-point scores persisted in `SsdbcodiStore` for rule-panel diagnostics
   and LLM evidence.
7. Output schemas reuse `ClusterResult` / `OutlierResult` from shared schemas.
8. The `SsdbcodiProvider` implements the `AnalysisProvider` protocol and backs
   `algorithm_adapters.run_default_analysis()` by default.
9. `/workflows/provider-feedback/` compares the adapter-facing
   `AnalysisResult` with standalone `SsdbcodiResult` score diagnostics.

Provider boundary:

- `algorithm_adapters` already defines the `AnalysisProvider` protocol.
- The default provider is `SsdbcodiProvider`; `SequentialLofThenKMeansProvider`
  remains as an explicit legacy provider for comparison.
- All downstream code (scatterplot, workflows, rule panel) continues to
  work because the output schemas are unchanged.

---

### Step 7: Chatbox

Build:

```text
chatbox
chatbox Flask page
mock selection, label, and instruction context
chat-selection workflow page
```

Current implementation:

1. `/modules/chatbox/` renders chat history, selection context, selection groups, label context, a mock `StructuredInstruction` preview panel, full-coverage suggestion chips, and example messages.
2. Selection context, selection groups, and label context are read from the real `selection` and `labeling` debug stores — chatbox never mutates them.
3. The intent provider is pluggable via the `IntentProvider` protocol. Step 7 ships `MockIntentProvider` (deterministic keyword-based router + intent extractor); Step 8 replaces the chatbox-side mock boundary with the real `intent_instruction` module, and Step 8.5 later replaces the module's mock backend with a live model runtime.
4. The mock `StructuredInstruction` state lives in the provider, not in chatbox, so the chatbox service can be tested (and verified) to not mutate instruction state.
5. `/modules/chatbox/api/messages` builds a `ChatMessagePayload` containing the last N turns (default 3), the selection context, selection groups, and label context, then returns the provider's `ChatResponse` with `router_category`, optional `delta`, `current_instruction_version`, and optional `intent_type`.
6. `/workflows/chat-selection/` combines selection, labeling, and chatbox state on one page for Step 7 acceptance.

Why:

Chatbox needs selection context and may benefit from recent label context, but should not own selection, labeling, algorithms, structured instruction state, or downstream path choice. Step 7 should stay focused on intake.

Tasks:

1. Build chat UI.
2. Display current selection context and selection groups.
3. Display recent manual label context when available.
4. Display the current `StructuredInstruction` panel (read from intent instruction).
5. Display suggestion chips derived from dataset context. Chips for `split_cluster` and `reclassify_outlier` remain visible for legacy coverage, but should not imply that old branching update strategies are still the active roadmap.
6. Submit user message with a truncated history window (default last 3 turns) plus context.
7. Show assistant response including router category.
8. Add `/modules/chatbox/`.
9. Add APIs for message submission, context, history, and reset.
10. Support mock selection, label, and instruction context for standalone testing.

Unit tests:

1. Empty messages are rejected.
2. Message payload includes selection context and selection groups.
3. Message payload includes label context when available.
4. Message payload is strategy-agnostic.
5. History window is truncated to the configured N turns.
6. Chatbox does not call clustering or outlier detection.
7. Chatbox does not mutate selection, labeling, or structured instruction state.
8. Suggestion chips include the legacy full feedback vocabulary.

Flask visual check:

Open `/modules/chatbox/` and confirm:

1. chat input works and messages appear in history.
2. selection and label context are visible.
3. suggestion chips produce valid intents when clicked, including `split_cluster` and `reclassify_outlier`.
4. the `StructuredInstruction` preview panel updates after actionable messages.
5. response clearly shows whether the provider is real or mocked.

Completion:

Chatbox can be manually tested in Flask with mock instruction state and real selection / label context, and Step 7 remains cleanly separated from downstream path choice.

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

This legacy module turns user language into structured instructions and remains
useful as provider/runtime infrastructure. The active product direction no
longer feeds these instructions into an update path; it uses DeepSeek
to explain generated rule cards.

Tasks:

1. Define `StructuredInstruction` schema and `InstructionDelta` schema.
2. Implement two-stage pipeline: router first, extractor only on actionable messages.
3. Define `LlmProvider` protocol and ship `MockLlmProvider` as the Step 8 backend. Real providers are intentionally deferred to Step 8.5 so the compilation boundary can be stabilized first.
4. Define the JSON-schema-shaped delta contract that future real providers must satisfy.
5. Resolve group references (`selected_points`, `selection_group`, `cluster`, `outlier_set`, `point_id`).
6. Generate clarification prompts for ambiguous messages. Keep `partial` reserved in the shared schema surface for Step 8.5 draft accumulation and follow-up prompting.
7. Emit all eight Phase 1 intents: `feature_weight`, `group_similar`, `group_dissimilar`, `merge_clusters`, `anchor_point`, `ignore_cluster`, `split_cluster`, `reclassify_outlier`. This remains legacy compiler coverage; the active roadmap no longer expands these intents into old update-strategy adapters.
8. Forward only the last N turns (default 3) plus the current instruction snapshot to the backend, not full chat history.
9. Add `/modules/intent-instruction/` with route, compile, state, reset, and examples APIs.
10. Add `/workflows/chat-intent/`.

Current implementation:

1. `app/modules/intent_instruction/` ships the two-stage pipeline (`router.py`, `extractor.py`) behind a thin `IntentInstructionProvider` service that satisfies the chatbox `IntentProvider` protocol. This lets the Step 7 chatbox swap its mock provider for the real intent module without any chatbox-side code change.
2. Two protocols stack intentionally: the inner `LlmProvider` (route + extract) is owned by intent_instruction; the outer `IntentProvider` (respond / current_snapshot / reset) is the chatbox boundary. `IntentInstructionProvider` is the adapter between them.
3. `MockLlmProvider` is the only backend that ships in Step 8. It is deterministic, keyword-driven, and is the tested default for the debug page and `/workflows/chat-intent/`. Step 8.5 and later can add live local or cloud providers through the same `LlmProvider` protocol without changing the rest of the module.
4. `StructuredInstruction` is owned by intent_instruction; the `InstructionSnapshot` view shared with chatbox was promoted to `app/shared/schemas.py` so both modules consume it without layering violations.
5. `DatasetContext` bundles dataset_id, feature_names, cluster_ids, selection_group_names, and selected/unselected point ids so the router and extractor can resolve references like "these points" or "cluster 2" deterministically.
6. The extractor emits `InstructionDelta`s whose operations carry a `pending` constraint_id placeholder; the service rewrites them to real IDs (`c1`, `c2`, ...) inside `apply_delta` and advances the version counter in `IntentInstructionStore`.
7. Off-topic, meta-query, and ambiguous messages do not mutate state. Actionable messages produce a delta, advance the version, and return the resulting `ChatResponse`. `split_cluster` and `reclassify_outlier` remain supported as legacy intent outputs but do not define the next build direction.
8. `/modules/intent-instruction/` exposes `/health`, `/api/route`, `/api/compile`, `/api/state`, `/api/reset`, and `/api/examples`. `/workflows/chat-intent/` wires the real `intent_instruction` module boundary into a chatbox shell so the structured instruction state can be observed across multiple turns while the backend is still deterministic.

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
5. active backend label is clearly shown, with Step 8 expected to remain mock.

Completion:

Intent compilation is visible and debuggable before any real-model dependency is introduced. Step 8.5 is the first stage where a live provider becomes mandatory.

---

### Step 8.5: Intent Runtime Validation

Build:

```text
intent runtime validation workflow
real-model LLM provider runtime
SSDBCODI score grounding
structured conversation memory + draft state
evaluation suite + provider diagnostics
```

Why:

Step 8 proves the compiler boundary, but it still uses a deterministic mock
backend. The project needs a real-model gate that validates wording robustness,
memory, relevance filtering, partial-information accumulation,
SSDBCODI-aware planning, and schema-valid structured output. This provider
foundation is reused by the new rule-interpretation workflow. At this stage the
model can suggest and explain label or selection ideas, but only manual UI
controls mutate labeling and selection.

Tasks:

1. Implement a real `LlmProvider` runtime with default config `provider: deepseek`, `model: deepseek-v4-pro`, `base_url: https://api.deepseek.com`.
2. Keep the runtime provider-agnostic so Ollama models or online providers can be swapped through configuration instead of branching the workflow design.
3. Embed the real `scatterplot`, `selection`, `labeling`, and SSDBCODI module boundaries into `/workflows/intent-runtime-validation/` so the live model is validated against the actual visible plot state and score diagnostics.
4. Add structured conversation memory owned by `intent_instruction`: append-only transcript, rolling summary, working memory, extracted facts with provenance and confidence, incomplete instruction draft, unresolved slots, and irrelevant-turn log.
5. Activate `partial` handling: extract usable fragments from incomplete user messages, store them in structured form, and ask one focused follow-up question instead of discarding the turn.
6. Distinguish `off_topic`, `meta_query`, `relevant_but_incomplete`, `actionable`, and correction/overwrite cases before mutating structured state.
7. Validate paraphrase robustness for wording variants such as `how many clusters`, `how many class`, and `what classes do we have`.
8. Validate visual grounding cases such as `these points`, `that cluster`, `the selected group`, and label-aware outlier references against the real plot state.
9. Promote incomplete multi-turn feedback to final `StructuredInstruction` state only when the required fields are present; otherwise keep it in draft form.
10. Add `/workflows/intent-runtime-validation/` as a composite visual lab: real scatterplot, real selection/labeling context, SSDBCODI scores/seeds/diagnostics, chat intake, provider/model controls, memory panels, draft state, final structured output, and evaluation diagnostics on one page.
11. Define replayable evaluation packs for paraphrases, meta-queries, irrelevant turns, partial completion, multi-turn memory, contradiction/correction, visual grounding, state drift, and provider timeout/failure.
12. Record explicit pass/fail gates that must be satisfied before rule interpretation starts.

Current design status:

1. Implemented as `/workflows/intent-runtime-validation/` with a runtime-configurable live provider and persisted session artifacts.
2. The default live model is DeepSeek V4 Pro, but the runtime config remains provider-agnostic for local and online models.
3. Prompt templates are file-backed under `prompts/intent_instruction/ollama/` and the rendered prompts are persisted per session for debugging.
4. Grounded context now includes real selection state, saved selection groups, manual labels, effective cluster/outlier state, SSDBCODI scores/seeds/diagnostics, recent chat turns, structured memory, and point-level catalog data.
5. The visible chat surface is direct AI reply only. The AI is constrained to planning and suggestions; label/selection mutation remains manual until the next suggestion-review gate is built.

Validation suites:

1. Paraphrase suite: same intent, varied wording and grammar.
2. Meta-query suite: cluster/class synonyms and count questions.
3. Relevant-vs-irrelevant suite: on-topic fragments are retained; off-topic chatter is logged without polluting instruction state.
4. Partial-information suite: incomplete messages update the draft and produce a focused follow-up.
5. Multi-turn memory suite: later turns can resolve earlier references without replaying the entire raw transcript.
6. Correction suite: updated user statements replace or confirm tentative facts without silently overwriting confirmed instruction state.
7. Provider failure suite: timeout, invalid JSON, and unavailable model states surface clear diagnostics.

Flask visual check:

Open `/workflows/intent-runtime-validation/` and confirm:

1. a real scatterplot is embedded as the main visual surface, not a detached text-only context panel.
2. click selection and rectangle selection update selection context immediately on the same page.
3. current label state is visible and stays consistent with the plot and context panels.
4. provider health and active model are clearly visible.
5. the memory view separates transcript, summary, confirmed facts, tentative facts, and irrelevant turns.
6. incomplete messages update a draft panel instead of mutating final instruction state.
7. clarification prompts ask for the next missing piece rather than repeating the whole request.
8. completed multi-turn feedback promotes into final structured output.
9. evaluation diagnostics show which validation packs passed or failed.
10. visual references such as `these points`, `that cluster`, and current outliers can be audited by comparing the plot, context panels, and structured output on one screen.
11. both user and assistant turns render in the chat thread so the visible UI matches the persisted transcript and chat-state JSON.
12. SSDBCODI `rScore`, `lScore`, `simScore`, `tScore`, seed, and outlier diagnostics are visible and included in the AI context.

Completion:

Rule interpretation should not begin until a real-model runtime is wired
through `LlmProvider`, the default DeepSeek V4 Pro path works, and the
validation gates for provider health, memory, structured extraction, SSDBCODI
grounding, and UI clarity all pass. The next product surface after this gate is
the read-only rule panel, not automatic updates.

---

### Step 8.6: Rule Panel

Build:

```text
rule_panel
decision-tree surrogate rule generator
rule cards for clusters and anomalies
rule-panel validation workflow
```

Why:

The project direction after Step 8.5 is rule explanation. The next product
surface should explain the current SSDBCODI output. A shallow decision-tree
surrogate can turn SSDBCODI cluster and anomaly assignments into
human-readable feature-threshold rules while leaving the original clustering
and anomaly provider unchanged. The decision tree is only a rule extractor; it
does not perform clustering or outlier/anomaly detection.

Tasks:

1. Create `app/modules/rule_panel/`.
2. Read the existing dataset, feature matrix, SSDBCODI cluster result, anomaly
   result, point scores, and optional labeling state through existing module
   contracts.
3. Train deterministic shallow decision-tree surrogates using SSDBCODI output
   as fixed targets:
   - one multi-class or one-vs-rest tree for SSDBCODI cluster labels,
   - one binary tree for SSDBCODI anomaly-vs-normal flags when anomalies exist.
4. Convert tree paths into rule cards with:
   - target kind (`cluster` or `anomaly`),
   - target id,
   - raw dataset feature-threshold conditions,
   - support count,
   - coverage,
   - purity,
   - matched point ids,
   - exception point ids.
5. Use the uploaded `wine.mat` file as the default Step 8.6 visual and test
   dataset, with rule conditions shown as raw wine features such as `alcohol`
   and `proline`, not projected `x/y` coordinates.
6. Add `/modules/rule-panel/` with rules, config, and health APIs.
7. Add `/workflows/rule-panel-validation/` to show scatterplot, SSDBCODI
   diagnostics, and generated rule cards together.
8. Keep the rule panel read-only. It does not mutate selection, labeling,
   SSDBCODI output, or projection.

Unit tests:

1. Rules are generated for every fixture cluster when enough points exist.
2. Anomaly rules are generated when outlier flags exist.
3. Rule conditions reference only known raw feature names from the source data.
4. Matched point ids satisfy every rule condition.
5. Coverage and purity are deterministic for fixed input.
6. Low-purity rules expose exception point ids.
7. Invalid tree config returns a consistent error.

Flask visual check:

Open `/modules/rule-panel/` and confirm:

1. one rule-card group appears for each current cluster,
2. anomaly rules appear when current outliers exist,
3. every rule card shows thresholds, support, coverage, purity, matched points,
   and exceptions,
4. changing tree depth or minimum leaf size changes rule complexity,
5. the page clearly says rules are explanations of current output, not new
   cluster truth.
6. the page shows `wine_mat`, the raw wine feature list, feature usage, and rule
   warning summaries.

Completion:

Rule cards explain the current SSDBCODI output in the browser without changing
or replacing the clustering or anomaly state.

---

### Step 8.7: Rule Interpretation with DeepSeek

Build:

```text
rule interpretation provider payload
DeepSeek rule parser
categorized label-guidance output
rule interpretation workflow
```

Why:

DeepSeek's new job is to interpret generated rules rather than parse free-form
feedback into update instructions. The model should classify each
interpretation into an auditable label/refinement category, cite evidence from
the rule cards, and recommend what the user should label next.

Tasks:

1. Reuse the existing DeepSeek provider configuration from Step 8.5.
2. Build a rule-interpretation request payload from `RuleSet`, SSDBCODI score
   summaries, feature names, current cluster/anomaly ids, computed
   `rule_guidance_metrics`, and `label_candidate_point_profiles`.
3. Compute rule metrics before the prompt: support, coverage, purity,
   exception rate, pair intersection count, pair Jaccard overlap, overlap
   shares, shared features, boundary gaps, and candidate point ids.
4. Require the model to classify output into one or more categories:
   - `label_priority`
   - `boundary_review`
   - `overlap_merge_signal`
   - `split_or_new_cluster_signal`
   - `anomaly_label_review`
   - `exception_relabel_review`
   - `feature_label_strategy`
   - `rule_confidence_audit`
5. Require `category_explanation`, `label_targets`, `suspicion_reasons`,
   `point_label_guidance`, `recommendation`, `quantitative_findings`, and
   `suggested_label_actions`.
6. Require evidence and action references to existing rule ids, features,
   thresholds, target ids, and point ids.
7. Validate that no unknown feature, threshold, cluster, anomaly, point id,
   rule id, action reference, or category appears in the parsed output.
8. Add `/modules/rule-panel/api/interpret` or a dedicated
   `/modules/rule-panel/api/interpretation` endpoint.
9. Add `/workflows/rule-interpretation/` to show rule cards beside categorized
   DeepSeek output.

Unit tests:

1. Valid model output is parsed into a `RuleInterpretation`.
2. Unknown categories are rejected.
3. Evidence referencing an unknown rule id is rejected.
4. Evidence referencing an unknown feature or point id is rejected.
5. Action guidance missing `recommendation`, `quantitative_findings`, or
   `suggested_label_actions` is rejected.
6. Mocked provider output can produce every interpretation category.
7. Provider errors return diagnostics without changing the `RuleSet`.

Flask visual check:

Open `/workflows/rule-interpretation/` and confirm:

1. rule cards are visible before interpretation,
2. interpretation categories are shown as focus cards with one-sentence
   explanations,
3. the main panel answers "which points to label", "why these points need
   checking", and "how to label them",
4. every interpretation cites rule evidence and quantitative findings in audit
   details,
5. warnings are shown when rules are broad, low-purity, contradictory, or have
   no sample-level overlap for merge review.

Completion:

DeepSeek can turn generated rules into categorized, grounded label/refinement
recommendations without owning or changing any dashboard state.

Current status:

Working. `/workflows/rule-interpretation/` shows wine rule cards beside
categorized recommendations. The primary UI is a three-part label guidance
panel: label targets, suspicion reasons, and point-level label guidance. Audit
details still expose quantitative findings, suggested label actions, evidence,
request payload, and provider diagnostics.
`/modules/rule-panel/api/interpretation` exposes the same auditable contract.
The workflow includes one button per interpretation category; clicking a
category sends `focus_category` into the provider payload and returns a focused
label/refinement recommendation for that category. The local default is
deterministic `mock`; `provider_kind=deepseek` uses DeepSeek V4 Pro
(`deepseek-v4-pro`) with thinking enabled and `reasoning_effort=high`, using
the prompt at `prompts/rule_interpretation/deepseek/label_guidance_prompt.txt`.
Provider fallback never mutates the `RuleSet`.

---

### Step 8.8: Integrated Rule Dashboard

Build:

```text
integrated rule dashboard
scatterplot + SSDBCODI diagnostics + rule panel + rule interpretation
```

Current Step 8.8 integration surface:

```text
/workflows/wine-dashboard/
```

This working workflow uses the uploaded `wine.mat` dataset and composes Data
Workspace, Projection, Algorithm Adapters/SSDBCODI, Selection, Labeling,
Scatterplot, Rule Panel, and DeepSeek-ready rule interpretation. The existing
chatbox UI is intentionally excluded.

Why:

Only integrate after rule generation and rule interpretation are visible in
their own pages. The integrated dashboard should keep the same module-first
discipline: it composes outputs, it does not hide logic inside the shell.

Tasks:

1. Compose the existing scatterplot, SSDBCODI diagnostics, generated rules, and
   rule interpretation preview on `/workflows/wine-dashboard/`.
2. Keep the dashboard shell thin; generated data should come from rule-panel
   and existing module APIs.
3. Let the user inspect a cluster or anomaly in the plot and see the matching
   rule card and interpretation.
4. Surface rule quality warnings clearly.
5. Keep direct label/selection controls explicit and manual.
6. Do not add an update-strategy toggle.

Unit tests:

1. Integrated route returns 200.
2. Integrated state includes render payload, SSDBCODI diagnostics, `RuleSet`,
   and latest `RuleInterpretation`.
3. Rule-panel provider failures do not break base scatterplot rendering.
4. LLM interpretation failures surface diagnostics without hiding rule cards.

Flask visual check:

Open `/workflows/wine-dashboard/` and confirm:

1. scatterplot appears,
2. each cluster has a visible rule summary,
3. anomalies have separate rule summaries,
4. selection and labeling controls update the effective SSDBCODI-facing state,
5. rule conditions use raw wine feature names, not projected coordinates,
6. the page has no chatbox surface.

Completion:

The dashboard explains current clusters and anomalies through rule cards and
categorized LLM interpretation, without claiming that the LLM changed the
underlying analysis.

Current status:

Working through `/workflows/wine-dashboard/`. The page is now the Step 8.8
integrated rule dashboard: it composes wine data, projection, SSDBCODI,
selection, labeling, scatterplot rendering, decision-tree rule cards, and
rule interpretation. The route accepts `provider_kind=deepseek|mock` and
`focus_category=<category>`. By default it follows `.env` provider settings;
with the current DeepSeek configuration it attempts `deepseek-v4-pro` and
shows provider diagnostics, model name, and fallback status directly on the
page. Tests use `provider_kind=mock` so automated checks do not consume
DeepSeek tokens.

DeepSeek token usage does not guarantee that usable JSON was returned. The
provider now records missing-content and malformed-JSON response metadata
(`finish_reason`, message keys, usage, and attempt errors), and the rule
interpreter retries once with the same `deepseek-v4-pro` model in direct JSON
mode if the initial thinking JSON call spends tokens but produces no valid
final JSON.

The Rule Interpretation panel is category-first. It shows one selectable
category control per interpretation category; the integrated dashboard defaults
to `label_priority` rather than a broad overview request. Each category either presents
concrete candidate points or an explicit "no typical case" state. The
user-facing guidance must be plain-language labeling advice: which wine ids to
label, why those points are suspicious or strategically useful in human terms,
and how each likely label outcome would affect merge, boundary,
split/new-cluster, anomaly, or rule-audit decisions. Quantitative metrics stay
available as audit evidence but should not be the main explanation.
Recommended point ids must be linked back to the scatterplot through visual
halos and interactive chips, so users can locate the exact wines without
manually searching the plot. This locate interaction is temporary visual focus
only and must not mutate selection or labels. User-facing terminology should
stay consistent: `outlier score`, `current analysis`, `human label`, and
`cutoff` are preferred over mixed expert terms.

Selection is a lightweight interaction. Clicking or rectangle-selecting points
must call only the selection API and update the scatterplot/selected-count UI
in place; it must not reload the whole dashboard or trigger rule
interpretation/DeepSeek. Full dashboard recomputation is reserved for actions
that change the analysis meaning, such as labels, rule parameters, category
focus, or provider controls.

---

### Removed: Old Branching Roadmap

The old post-Step-8.5 branching update roadmap is no longer the active build
order. Do not implement those modules from this process document unless the
roadmap is explicitly reopened. The current build order after Step 8.5 is only:
`rule_panel`, rule interpretation, and the integrated rule dashboard.

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

Legacy chatbox receives selection/label context and intent module outputs
structured instructions. This remains implemented history, not the next build
direction.

### Milestone 5.5: Real LLM Validation Gate

Goal:

`/workflows/intent-runtime-validation/` proves a live provider runtime
(default DeepSeek V4 Pro unless `.env` overrides it), SSDBCODI-grounded direct
AI planning, structured memory, partial-information draft handling, and
schema-valid structured output. The provider/runtime foundation is reused for
rule interpretation.

### Milestone 6: Rule Panel

Goal:

Current SSDBCODI clusters and anomalies are converted into decision-tree rule
cards with feature thresholds, coverage, purity, matched points, and exception
points.

Current status:

Working. `/modules/rule-panel/` generates explanation-only decision-tree rule
cards from SSDBCODI outputs, and `/workflows/rule-panel-validation/` exposes the
source analysis, rule cards, and deterministic interpretation preview together.
The default visual/test dataset is `wine.mat`; rule conditions and diagnostics
use raw wine feature names rather than projected coordinates.

### Milestone 6.5: Rule Interpretation

Goal:

DeepSeek parses generated rule cards into categorized label/refinement
recommendations: `label_priority`, `boundary_review`,
`overlap_merge_signal`, `split_or_new_cluster_signal`,
`anomaly_label_review`, `exception_relabel_review`,
`feature_label_strategy`, and `rule_confidence_audit`.

Current status:

Working through `/workflows/rule-interpretation/` and
`/modules/rule-panel/api/interpretation`. Model output is validated against
known categories, rule ids, raw wine features, thresholds, target ids, and point
ids before it becomes a `RuleInterpretation`. Category buttons are implemented
in Step 8.7 through the `focus_category` request path. The interpretation
contract now requires a concrete recommendation, quantitative findings, and
suggested label actions.

### Milestone 7: Integrated Dashboard

Goal:

The full local dashboard explains current clusters and anomalies through the
scatterplot, SSDBCODI diagnostics, rule cards, and categorized rule
interpretation.

Current status:

Working through `/workflows/wine-dashboard/` for Step 8.8. It uses `wine.mat`,
excludes the chatbox UI, and combines data, projection, SSDBCODI, selection,
labeling, scatterplot rendering, rule cards, and rule interpretation. The page
can use deterministic `mock` for testing or `deepseek-v4-pro` from `.env` for
live point-level guidance, and it shows fallback status explicitly.
Scatterplot selection updates are client-local after the lightweight selection
API returns, so basic point selection does not spend DeepSeek tokens or wait for
rule interpretation.
