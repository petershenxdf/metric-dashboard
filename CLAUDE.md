# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the app
python run.py
# App available at http://127.0.0.1:5000

# Run all tests
python -m unittest discover -s tests

# Run tests for a single module
python -m unittest discover -s tests/modules/data_workspace
python -m unittest discover -s tests/modules/projection

# Check for syntax errors
python -m compileall app tests
```

## Architecture

This is a local Flask dashboard for human-in-the-loop metric learning. The stack is pure Python + Flask + vanilla JS - no frontend framework, no production database.

### App Factory

`app/__init__.py` exports `create_app(enabled_modules=None)`, which calls:
1. `register_core_routes(app)` - `/`, `/health`, `/modules/`, `/workflows/`
2. `register_modules(app, enabled_modules)` - loads blueprints from the module registry
3. `register_workflows(app, enabled_modules)` - registers workflow blueprints from `app/workflows/`

### Module Registry

Each module is declared as a `ModuleInfo` entry (slug, package_name, title, purpose, status, blueprint_factory) in `app/module_registry.py`. Blueprint factories are lazy-loaded via `importlib.import_module` — modules are only imported when their blueprint is actually registered. The dashboard shell reads this registry; modules never import the dashboard shell.

### Module Contract

Every module under `app/modules/<module_name>/` must have:

| File | Purpose |
|------|---------|
| `service.py` | Pure logic, independently unit-tested |
| `schemas.py` | Module-local or re-exported shared schemas |
| `fixtures.py` | Sample data for tests and debug pages |
| `routes.py` | Flask blueprint with debug page and APIs |
| `templates/<module_name>/index.html` | Browser-visible debug page |

### Route Conventions

- Python packages: `snake_case` (e.g., `data_workspace`)
- Flask URL slugs: `kebab-case` (e.g., `data-workspace`)
- Module pages: `/modules/<module-slug>/`
- Module APIs: `/modules/<module-slug>/api/<action>`
- Workflow pages: `/workflows/<workflow-slug>/`

### API Response Envelope

All debug APIs must return this shape:

```json
{ "ok": true, "data": {}, "error": null, "diagnostics": {} }
```

Errors use `"ok": false` with `"error": { "code": "...", "message": "..." }`.

### State Ownership

State is owned by exactly one module - other modules read through contracts, never mutate:

| State | Owner |
|-------|-------|
| dataset / feature matrix | `data_workspace` |
| projection coordinates | `projection` |
| cluster assignments | `algorithm_adapters` backed by SSDBCODI; legacy LOF+KMeans remains available explicitly |
| outlier scores | `algorithm_adapters` backed by SSDBCODI `tScore`; legacy LOF score remains available explicitly |
| ssdbcodi per-point intermediate scores (`rScore`, `lScore`, `simScore`, `tScore`) | `ssdbcodi` |
| selected point IDs | `selection` |
| manual annotations | `labeling` |
| chat history | `chatbox` |
| structured instructions (shared) | `intent_instruction` |
| metric constraints (Path A) | `metric_learning_adapter` |
| direct feedback plan / active `DirectFeedbackPlan` (Path B) | `direct_feedback_adapter` |
| Path A refinement history (`metric_refinement_runs`) | `metric_refinement_orchestrator` |
| Path B refinement history (`direct_refinement_runs`) | `direct_refinement_orchestrator` |

### Current Pipeline (implemented through Step 8 compiler boundary; Step 8.5 live runtime is planned)

```
data_workspace -> projection -> algorithm_adapters -> selection -> labeling -> scatterplot
                                                                            \-> chatbox (Step 7) -> intent_instruction (Step 8) -> Step 8.5 live runtime validation
```

- `algorithm_adapters`: defaults to `SsdbcodiProvider`, preserving the same dashboard-facing `ClusterResult`, `OutlierResult`, and `AnalysisResult` schemas. `SequentialLofThenKMeansProvider` remains as an explicit legacy provider.
- `selection`: supports `select`/`deselect`/`replace`/`toggle`/`clear`, named selection groups (not semantic labels), sources include `point_click`, `rectangle`, `lasso`, `api`, `workflow_fixture`, `selection_group`.
- `labeling`: converts selected points into `assign_cluster`, `assign_new_class`, `mark_outlier`, `mark_not_outlier` annotations -> structured feedback instructions.
- `scatterplot`: builds a render payload from upstream state; does not own selection or label truth.
- `chatbox`: reads selection/selection-groups/label context from their owning modules without mutating them, forwards messages through an `IntentProvider` protocol (Step 7 ships `MockIntentProvider`; Step 8 swaps in the real `intent_instruction` module boundary), and stays strategy-agnostic. The mock `StructuredInstruction` snapshot lives inside the provider, not chatbox.

The main manual test page for the full Step 1-6 path is `/workflows/scatter-labeling/`. The Step 7 page is `/workflows/chat-selection/`.

### Chatbox Module (Step 7)

`app/modules/chatbox/` is the dialogue UI for user feedback.

- Target files: `schemas.py`, `service.py`, `store.py`, `state.py`, `fixtures.py`, `routes.py`, `templates/chatbox/index.html`, `providers/{base.py,mock.py}`.
- `IntentProvider` protocol decouples chatbox from the intent-compilation pipeline: `respond(payload) -> ChatResponse`, `current_snapshot(dataset_id)`, `reset(dataset_id)`. Step 8 (`intent_instruction`) now also satisfies the same protocol, so `/workflows/chat-intent/` swaps in the real provider without chatbox code changes while `/modules/chatbox/` keeps the mock for isolated testing.
- `ChatMessagePayload` includes: the message, dataset_id, selection context (selected/unselected point IDs), selection groups, label context, and truncated history window (default last 3 turns).
- `ChatResponse` includes: `reply`, `router_category`, optional `delta`, `current_instruction_version`, optional `intent_type`, optional followup question, and `provider_label`.
- Step 7 is intentionally strategy-agnostic. Path choice begins later, when Step 9 adapters consume the compiled instruction state.
- Suggestion chips for `split_cluster` and `reclassify_outlier` remain visible in Step 7, but are marked as Path B-only downstream intents rather than hidden.
- Debug page at `/modules/chatbox/`, workflow at `/workflows/chat-selection/`. See `docs/modules/chatbox/design.md`.

### Intent Instruction Module (Step 8)

`app/modules/intent_instruction/` converts chat text into structured instruction deltas.

- Target files: `schemas.py`, `router.py`, `extractor.py`, `service.py`, `store.py`, `state.py`, `fixtures.py`, `routes.py`, `templates/intent_instruction/index.html`, `providers/{base.py,mock.py}`.
- Two protocols stack intentionally:
  - Inner `LlmProvider` (owned here): `route(message, context, history) -> RouterResult`, `extract(message, context, history, current_instruction) -> InstructionDelta`. `MockLlmProvider` is the only provider that ships in Step 8; Step 8.5 and later can add live local or cloud providers without touching chatbox.
  - Outer `IntentProvider` (chatbox boundary): `IntentInstructionProvider` is the adapter that satisfies it using the inner `LlmProvider`.
- Router categories: `on_topic_actionable`, `on_topic_ambiguous`, `partial`, `meta_query`, `off_topic`. Current Step 8 behavior only routes `on_topic_actionable` into the extractor; `partial` remains reserved in the shared schema surface.
- Phase 1 intents (8): shared (`feature_weight`, `group_similar`, `group_dissimilar`, `merge_clusters`, `anchor_point`, `ignore_cluster`) + Path B-only downstream intents (`split_cluster`, `reclassify_outlier`). Step 8 emits all eight without choosing Path A vs Path B; adapters enforce final acceptance later.
- `StructuredInstruction` (full state, owned here) vs `InstructionSnapshot` (narrow cross-module view, in `app/shared/schemas.py`). The extractor emits `InstructionDelta`s whose `constraint_id` starts as `pending`; the service rewrites it to a real ID (`c1`, `c2`, ...) inside `apply_delta` and advances the version counter.
- Off-topic, meta-query, and ambiguous messages never mutate state.
- Debug page at `/modules/intent-instruction/` with `/health`, `/api/route`, `/api/compile`, `/api/state`, `/api/reset`, `/api/examples`. Workflow at `/workflows/chat-intent/`. See `docs/modules/intent_instruction/design.md`.

### Planned Runtime Validation Gate (Step 8.5)

`Step 8.5` is the first stage that actually connects a live model runtime.

- Default first runtime: Ollama `qwen2.5:14b` at `http://127.0.0.1:11434`.
- Provider contract stays generic so future local or online models can slot in without changing Step 7 ownership boundaries.
- `intent_instruction` owns the structured memory for this stage: append-only transcript, rolling summary, extracted facts with provenance, incomplete draft state, clarification agenda, irrelevant-turn log, and evaluation diagnostics.
- `/workflows/intent-runtime-validation/` is the pre-Step-9 gate for paraphrase robustness, meta-query alias handling, partial-information completion, relevance filtering, and provider failure behavior.

### Planned Refinement Pipeline (Steps 9-11, A/B fork)

After Step 6.5 the feedback loop forks into two parallel update strategies so they can be compared experimentally. The shared upstream stages are identical:

```
chatbox (strategy-agnostic intake)
  -> intent_instruction (emits shared + Path B-only intents)
  -> Step 8.5 live runtime validation (real provider + memory + draft completion)
  -> structured feedback
      -> refinement trigger chooses Path A or Path B
      +-- Path A: metric_learning_adapter -> metric_refinement_orchestrator
      |     (learns M; applies L = chol(M) as linear pre-transform;
      |      reruns projection and algorithm_adapters on X · L)
      +-- Path B: direct_feedback_adapter -> direct_refinement_orchestrator
            (builds DirectFeedbackPlan of seed_updates, feature_scale,
             param_overrides; reruns SSDBCODI directly with merged seeds)
  -> /workflows/strategy-comparison/ runs both paths on the same feedback
```

Intent handling rules:
- Shared intents: `feature_weight`, `group_similar`, `group_dissimilar`, `merge_clusters`, `anchor_point`, `ignore_cluster`.
- Path B-only intents: `split_cluster`, `reclassify_outlier`. Path A adapters return `intent_deferred` for these with `suggested_strategy: "direct_ssdbcodi"`; Path B handles them natively through seed updates and `n_clusters` overrides.
- Path A and Path B keep independent refinement histories and rollback stacks. Only `/workflows/strategy-comparison/` reads both.

### SSDBCODI Module (active clustering/outlier provider)

`app/modules/ssdbcodi/` implements *Semi-Supervised Density-Based Clustering with Outlier Detection Integrated* ([arXiv:2208.05561](https://arxiv.org/abs/2208.05561)) as both a separate debug module and the default provider behind `algorithm_adapters.run_default_analysis()`.

- Bootstrap: density-safe KMeans (default `k=3`, user-configurable) seeds SSDBCODI by promoting each dense cluster's centroid-nearest point to a labeled normal seed. These bootstrap anchors remain active as a stable baseline; manual labels override only the explicitly labeled point.
- Algorithm formulas follow the paper contract: symmetric `rDist = max(cDist(p), cDist(q), dist(p,q))`, `lScore` from nearest-neighbor `rDist`, `simScore` from nearest labeled outlier distance, and `tScore = alpha(1-rScore) + beta(1-lScore) + gamma*simScore`.
- The debug page uses the existing `selection` and `labeling` module stores: click and rectangle selection are additive, selected points use black center dots, saved selection groups work, and labels are limited to `cluster_1...cluster_n` plus `outlier`.
- The debug page includes multiple deterministic fixtures (`demo`, `moons`, `circles`) selected by `dataset_id`; selection, labels, and SSDBCODI store state are scoped per dataset.
- GET `/modules/ssdbcodi/` previews the current result without writing run history. `POST /modules/ssdbcodi/api/label` saves pending feedback only; `POST /modules/ssdbcodi/api/run` recomputes and stores results in `SsdbcodiStore`.
- Per-point scores `rScore`, `lScore`, `simScore`, `tScore` are persisted in `SsdbcodiStore` for downstream metric-learning consumption.
- Output schemas (`ClusterResult`, `OutlierResult`) live in `app/shared/schemas.py` (re-exported by `algorithm_adapters/schemas.py`), so downstream modules consume SSDBCODI results without changes.
- The debug page is at `/modules/ssdbcodi/`. See `docs/modules/ssdbcodi/design.md` for the full contract.

### Workflows

Workflow files live in `app/workflows/`. A workflow page connects multiple modules on one visual debug page. It does not own module internals - it orchestrates through module schemas and service calls.

The workflow registry includes `group`, `step`, and `debug_focus` metadata so `/workflows/` can act as a debug map rather than a flat link list. See `docs/workflows.md`.

Key workflows:
- `/workflows/data-projection/` - Step 1-2 core smoke test.
- `/workflows/default-analysis/` - Step 1-3 SSDBCODI-backed analysis provider smoke test.
- `/workflows/selection-context/` and `/workflows/selection-labeling/` - state boundary probes.
- `/workflows/analysis-selection/` and `/workflows/analysis-labeling/` - visual integration through Step 1-5.
- `/workflows/scatter-selection/` and `/workflows/scatter-labeling/` - Step 1-6 render/selection/labeling checks.
- `/workflows/provider-feedback/` - Step 6.5 provider diagnostics for adapter boundary plus standalone SSDBCODI scores.
- `/workflows/chat-selection/` - Step 7 chat intake reading selection, selection groups, and labeling context through a mocked intent provider.
- `/workflows/chat-intent/` - Step 8 chatbox wired to the real `IntentInstructionProvider`, exercising the full chat -> router -> extractor -> structured instruction loop with the deterministic backend.
- `/workflows/intent-runtime-validation/` - Step 8.5 live-model validation gate for provider health, memory, partial drafts, and evaluation diagnostics.

### Shared Layer (`app/shared/`)

Code that multiple modules or workflows need lives in `app/shared/`:

| File | Purpose |
|------|---------|
| `schemas.py` | `Dataset`, `FeatureMatrix`, `AnalysisResult`, `ClusterResult`, `OutlierResult`, etc. |
| `flask_helpers.py` | `api_success`, `api_error` response envelope helpers |
| `fixtures.py` | Cross-module fixture datasets (wide-gap, default analysis) used by workflows and scatterplot |
| `request_helpers.py` | Shared Flask request parsing (`n_clusters_from_request`, `dataset_id_from_request`, `apply_selection_action_or_error`) |
| `effective_analysis.py` | Pure logic to apply explicit label overrides on top of provider output |

**Layering rule:** `modules → shared` is OK. `modules → workflows` is a violation. When both modules and workflows need the same code, it belongs in `app/shared/`. Workflow files (`app/workflows/fixtures.py`, `app/workflows/effective_analysis.py`) are thin re-export shims pointing to `app/shared/` for backward compatibility.

### Module Boundaries (never cross these)

1. Scatterplot does not parse chat text and does not own selection or label truth.
2. Chatbox does not call clustering, outlier detection, projection, or metric learning.
3. Selection state is owned by the selection module only.
4. Labeling owns manual annotations; scatterplot sends label actions *to* labeling.
5. Existing clustering/outlier algorithms are only accessed through `algorithm_adapters`.
6. Modules do not import unrelated module internals - use schemas, services, APIs, or workflow pages.
7. Modules never import from `app/workflows/` - shared code goes in `app/shared/`.

### Definition of Done for a Module

A module is complete only when all of these are true:
1. Documented input/output contracts exist.
2. Pure service logic passes unit tests.
3. Flask debug page exists at `/modules/<module-slug>/`.
4. At least one JSON state API endpoint exists.
5. Flask route smoke tests pass.
6. The debug page can be opened and visually inspected in the browser.
7. Design document at `docs/modules/<module_name>/design.md` is updated.

### Docs to Read Before Starting a New Module

```
docs/overview.md
docs/flask_app.md
docs/process.md
docs/module_debug_checklist.md
docs/integration_testing.md
docs/state_and_api_contracts.md
docs/modules/<module_name>/design.md
```

这个项目的核心设计思想是模块化设计：每个模块相对独立，有自己的测试、可视化单元，又可以和其他模块整合，开发时不至于互相影响。
