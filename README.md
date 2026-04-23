# Metric Dashboard

## What This Project Is

Metric Dashboard is a local Flask dashboard for human-in-the-loop metric learning.

The app is meant to run locally, stay simple, and make every module visible for debugging. Deployment, production auth, cloud infrastructure, and complex frontend frameworks are not part of the current scope.

Run target:

```powershell
python run.py
```

Expected local URL:

```text
http://127.0.0.1:5000
```

Live runtime defaults for the Step 8.5 chat workflow are read from the repo-root
`.env` file. Supported keys:

```text
METRIC_DASHBOARD_LLM_PROVIDER
METRIC_DASHBOARD_LLM_MODEL
METRIC_DASHBOARD_OLLAMA_BASE_URL
METRIC_DASHBOARD_OLLAMA_KEEP_ALIVE
METRIC_DASHBOARD_LLM_TEMPERATURE
METRIC_DASHBOARD_LLM_TIMEOUT_SECONDS
METRIC_DASHBOARD_LLM_MAX_OUTPUT_TOKENS
METRIC_DASHBOARD_LLM_ALLOW_MOCK_FALLBACK
```

Edit `.env` when you want to change the default model or Ollama endpoint, then
restart `python run.py`. The runtime form on
`/workflows/intent-runtime-validation/` still works as a session-only override.
That same workflow also exposes a reply-mode toggle:

1. `Processed Reply`
   - shows the workflow's normal cleaned confirmation / clarification text.
2. `Direct AI Reply`
   - shows a separate freeform model-authored reply built from the same
     grounded context and memory, while leaving the actual instruction
     pipeline unchanged.

Prompt templates now live under the repo-root `prompts/` folder:

```text
prompts/
  intent_instruction/
    ollama/
      route_prompt.txt
      extract_prompt.txt
      reply_prompt.txt
```

Anything in the app that needs prompt files should read from there instead of
module-local prompt folders.

## Product Goal

The dashboard helps a user inspect high-dimensional data, select points, and give feedback that can guide metric learning.

The first product version has two main user-facing areas:

1. Scatterplot
   - Projects user data into 2D with MDS.
   - Shows default clustering results.
   - Shows default outlier detection results.
   - Colors points by cluster.
   - Marks outliers.
   - Lets the user select points.
   - Lets selected points become explicit cluster or outlier labels through the labeling module.

2. Chatbox
   - Receives feedback about selected or unselected points.
   - Converts relevant feedback into structured instructions.
   - Asks clarification when feedback is incomplete.
   - Does not directly run clustering, outlier detection, or metric learning.

Existing clustering and outlier detection algorithms are treated as fixed external logic. They should be wrapped by adapters, not redesigned or silently replaced.

## System Loop

The feedback loop forks into two update strategies after the shared upstream
stages so they can be compared experimentally.

```text
user data
  -> data workspace
  -> MDS projection
  -> clustering adapter
  -> outlier adapter
  -> scatterplot
  -> point selection
  -> direct labeling / annotation
  -> or chatbox feedback
  -> intent instruction for chat-derived feedback
  -> intent runtime validation (real provider + memory + draft completion)
  -> unified structured feedback
      -> refinement trigger chooses Path A or Path B
      +-- Path A: metric_learning_adapter -> metric_refinement_orchestrator
      |     (learns M, applies L = chol(M) as linear pre-transform,
      |      reruns projection and algorithm adapters on X · L)
      +-- Path B: direct_feedback_adapter -> direct_refinement_orchestrator
            (builds DirectFeedbackPlan of seeds / feature_scale / n_clusters,
             reruns SSDBCODI directly with merged seeds and param overrides)
  -> updated projection/clusters/outliers (from whichever path ran)
  -> updated scatterplot
```

## Main Design Principle

This is a Flask-first, module-first project.

There should be one simple local Flask app, and every module should have its own visible debug page inside that app.

Unit tests are required, but they are not enough. Because this is a dashboard, each module must also be inspectable in the browser.

Each module should have:

1. pure service logic.
2. unit tests.
3. Flask route tests.
4. a Flask debug page.
5. at least one JSON/state API endpoint.
6. fixtures or mock data for local testing.
7. a clear boundary from other modules.

## Required Local Routes

The Flask app should provide:

```text
/                                      integrated dashboard
/health                                app health
/modules/                              module lab index
/modules/<module_name>/                module debug page
/modules/<module_name>/health          module health
/modules/<module_name>/api/...         module state/action APIs
/workflows/                            workflow demo index
/workflows/<workflow_name>/            multi-module interaction demo
```

The module lab is important. It lets the developer open one module at a time and check whether it works before connecting it to the full dashboard.

## Planned Modules

| Module | Role | Debug Page |
| --- | --- | --- |
| `dashboard_shell` | Flask app, module registry, shared layout, workflow links | `/`, `/modules/`, `/workflows/` |
| `data_workspace` | Dataset loading, point IDs, feature matrix | `/modules/data-workspace/` |
| `projection` | MDS projection into 2D | `/modules/projection/` |
| `algorithm_adapters` | Adapter boundary for clustering and outlier providers, currently backed by SSDBCODI | `/modules/algorithm-adapters/` |
| `selection` | Selected and unselected point state | `/modules/selection/` |
| `labeling` | Manual point annotations, cluster labels, and outlier labels | `/modules/labeling/` |
| `scatterplot` | Point rendering, clusters, outliers, visual selection | `/modules/scatterplot/` |
| `ssdbcodi` | Active semi-supervised clustering/outlier provider plus score diagnostics | `/modules/ssdbcodi/` |
| `chatbox` | Chat UI, user feedback, clarification display, and context-aware intake | `/modules/chatbox/` |
| `intent_instruction` | Message classification and structured instruction output (shared by both paths) | `/modules/intent-instruction/` |
| `metric_learning_adapter` | **Path A**: structured instruction to pair-based metric-learning constraints and learned `M` | `/modules/metric-learning-adapter/` |
| `direct_feedback_adapter` | **Path B**: structured instruction to SSDBCODI-native `DirectFeedbackPlan` | `/modules/direct-feedback-adapter/` |
| `metric_refinement_orchestrator` | **Path A**: coordinates the metric-learning refinement loop | `/modules/metric-refinement-orchestrator/` |
| `direct_refinement_orchestrator` | **Path B**: coordinates the direct-SSDBCODI refinement loop | `/modules/direct-refinement-orchestrator/` |

## Current Implementation Status

The current working modules are:

1. `dashboard_shell`
   - app factory, registry-driven module/workflow registration, module lab, workflow lab, and placeholders.

2. `data_workspace`
   - fixture dataset, stable point IDs, feature matrix API, and debug page.

3. `projection`
   - MDS projection, projection API, SVG debug plot, and `/workflows/data-projection/`.

4. `algorithm_adapters`
   - defaults to the SSDBCODI integrated clustering/outlier provider.
   - keeps the old LOF-then-KMeans provider available as an explicit legacy provider.
   - returns the same `ClusterResult`, `OutlierResult`, and `AnalysisResult` schemas to downstream modules.
   - `n_clusters` can be adjusted from the module page, workflow page, or API query string.
   - `/workflows/default-analysis/` shows data, projection, outliers, and clusters together.

5. `selection`
   - owns selected and unselected point state.
   - supports `select`, `deselect`, `replace`, `toggle`, and `clear`.
   - supports saved selection groups, which are reusable named point sets.
   - preserves `source`, `mode`, and metadata fields for future UI gestures such as lasso and rectangle selection.
   - `/workflows/selection-context/` shows Data Workspace point IDs converted into reusable selection context.
   - `/workflows/analysis-selection/` connects Steps 1-4 on one shared visual layer with dataset switching, click selection, and rectangle selection.

6. `labeling`
   - owns manual point annotations derived from selected points.
   - supports assigning selected points to a cluster or new class.
   - supports marking selected points as outliers or not outliers.
   - converts annotations into structured feedback instructions.
   - `/workflows/selection-labeling/` shows selection context beside annotation output.
   - `/workflows/analysis-labeling/` connects Steps 1-5 on one shared visual layer: data, projection, SSDBCODI outliers, SSDBCODI clusters, selection, and labeling.
   - in the Step 1-5 workflow, labels are limited to `cluster_1...cluster_n` and `outlier`; those labels are passed into SSDBCODI and then reflected in the effective cluster/outlier state and frontend colors/markers.
   - `/workflows/analysis-labeling/` remains the main manual browser test page for the completed Step 1-5 path.

7. `scatterplot`
   - builds a point render payload from projection, cluster, outlier, selection, and label state.
   - shows projected points with cluster colors, outlier markers, selected point indicators, and manual label context.
   - preserves click selection, rectangle selection, saved selection groups, and adjustable cluster count in the Step 1-6 workflows.
   - exposes `/modules/scatterplot/api/render-payload` for downstream UI rendering.
   - `/workflows/scatter-selection/` tests scatterplot click/rectangle selection and saved groups flowing through the selection module.
   - `/workflows/scatter-labeling/` is the current main Step 1-6 manual browser test page: data, projection, algorithms, scatterplot, selection, saved groups, and labeling together.

The default algorithm-adapter fixture is `default_analysis_outlier_debug`, not Iris. It intentionally contains three compact clusters plus three distant outlier candidates so Step 3 is visually inspectable.

8. `ssdbcodi`
   - implements the active semi-supervised clustering/outlier provider and keeps an independent debug module at `/modules/ssdbcodi/`.
   - uses density-safe KMeans center seeds as stable bootstrap anchors, then merges manual labels on top so one relabel does not drop unrelated anchors.
   - includes selectable debug datasets (`demo`, `moons`, `circles`) to test separated, curved, and ring-shaped structures.
   - persists `rScore`, `lScore`, `simScore`, and `tScore` for downstream metric-learning use.
   - reuses the existing selection and labeling contracts: additive click/rectangle selection, black center dots for selected points, saved selection groups, and label controls limited to `cluster_1...cluster_n` plus `outlier`.
   - keeps label entry and execution separate: Apply Label saves pending labeling feedback; Run and Store recomputes and persists SSDBCODI.
   - returns dashboard-compatible `ClusterResult` and `OutlierResult` schemas and now backs the default `algorithm_adapters` provider boundary.
   - `/workflows/provider-feedback/` verifies the promoted provider boundary beside standalone SSDBCODI score diagnostics.

9. `chatbox`
   - dialogue UI that reads selection, selection groups, and label context from the real `selection` and `labeling` debug stores without mutating them.
   - forwards messages through a pluggable `IntentProvider` protocol; Step 7 ships `MockIntentProvider` (deterministic keyword-based router + intent extractor) so the chatbox can be tested standalone. Step 8 (`intent_instruction`) now also satisfies the same protocol, so `/workflows/chat-intent/` can wire the real provider in without any chatbox code change while `/modules/chatbox/` keeps the mock for isolated testing.
   - owns chat history only; the `InstructionSnapshot` lives inside whichever provider is active, not inside chatbox, so the module never owns instruction truth.
   - forwards a truncated history window (default last 3 turns) plus selection/label/instruction context with each message.
   - renders suggestion chips for the full Phase 1 vocabulary. `split_cluster` and `reclassify_outlier` remain visible but are marked as Path B-only downstream intents instead of being hidden.
   - fallback responses explicitly mark themselves as coming from the mock provider so users aren't confused by keyword-matcher limitations.
   - `/workflows/chat-selection/` combines selection, labeling, and chatbox state on one page as the Step 7 intake check.

10. `intent_instruction`
    - owns `StructuredInstruction` state and the two-stage router + extractor pipeline behind `IntentInstructionProvider`, which satisfies both the inner `LlmProvider` protocol (route/extract) and the outer `IntentProvider` protocol expected by chatbox.
    - Step 8 ships `MockLlmProvider` (deterministic keyword-driven router + extractor) as the only LLM provider; Step 8.5 and later can plug live local/cloud models into the same `LlmProvider` protocol without code changes elsewhere.
    - emits all eight Phase 1 intents (`feature_weight`, `group_similar`, `group_dissimilar`, `merge_clusters`, `anchor_point`, `ignore_cluster`, `split_cluster`, `reclassify_outlier`) in a strategy-agnostic way; the adapters in Steps 9A/9B enforce final acceptance.
    - off-topic, meta-query, and ambiguous messages never mutate state; actionable messages append a versioned `InstructionDelta` with a real `constraint_id`.
    - `/modules/intent-instruction/` exposes `/health`, `/api/route`, `/api/compile`, `/api/state`, `/api/reset`, `/api/examples`. `/workflows/chat-intent/` wires the real `intent_instruction` module boundary into a chatbox shell so the instruction state can be observed across multiple turns as the Step 8 compilation check.
    - `InstructionSnapshot` (shared cross-module view) was promoted to `app/shared/schemas.py` so chatbox and intent_instruction can both consume it without layering violations.

Planned next gate before Step 9:

- `Step 8.5` is intentionally separate from Step 8. It is the first stage that
  actually connects a live model runtime, defaulting to Ollama
  `qwen2.5:14b`, and validates paraphrase robustness, structured memory,
  partial-information completion, relevance filtering, and UI diagnostics
  before either downstream refinement path is trusted. The default provider,
  model, endpoint, and Ollama keep-alive window are read from `.env`, while the
  workflow page can temporarily override them for the current session. The same
  page now lets you flip the chat surface between processed replies and direct
  AI replies without changing the grounded context, memory payload, or
  underlying structured-instruction commit logic.

## Workflow Debug Map

The workflow index is grouped by debugging purpose, not just build order:

1. Core pipeline smoke tests:
   - `/workflows/data-projection/`
   - `/workflows/default-analysis/`
2. State boundary probes:
   - `/workflows/selection-context/`
   - `/workflows/selection-labeling/`
3. Visual integration tests:
   - `/workflows/analysis-selection/`
   - `/workflows/analysis-labeling/`
   - `/workflows/scatter-selection/`
   - `/workflows/scatter-labeling/`
4. Provider diagnostics:
   - `/workflows/provider-feedback/`
5. Feedback pipeline:
   - `/workflows/chat-selection/` (Step 7 chat intake + selection + labeling context)
   - `/workflows/chat-intent/` (Step 8 intent compilation)
   - `/workflows/intent-runtime-validation/` (Step 8.5 live-model validation gate)
6. Future workflows:
   - `/workflows/instruction-constraints/` (Path A: metric learning constraints preview)
   - `/workflows/instruction-ssdbcodi/` (Path B: DirectFeedbackPlan preview)
   - `/workflows/metric-refinement-loop/` (Path A end-to-end)
   - `/workflows/direct-refinement-loop/` (Path B end-to-end)
   - `/workflows/strategy-comparison/` (Path A vs Path B side-by-side)

See `docs/workflows.md` for the current workflow contract and grouping rules.

## Module Boundary Rules

1. Scatterplot does not parse chat text.
2. Chatbox does not run clustering, outlier detection, projection, or metric learning.
3. Selection state is owned by the selection module, not hidden inside scatterplot.
4. Labeling owns manual cluster/outlier annotations derived from selected points.
5. Scatterplot can expose label actions, but it must send them to labeling instead of owning label truth.
6. Intent instruction receives chat text plus context and outputs structured instructions.
7. Metric-learning adapter receives structured instructions, not raw chat text.
8. Existing algorithms are accessed only through algorithm adapters.
9. Dashboard shell composes modules but does not own module internals.
10. Integration should happen through schemas, services, APIs, or workflow pages.

## Structured Instructions

Actionable user feedback should become stable structured instructions.

Step 8 defines the shared instruction schema and emits Phase 1 intents. Step
8.5 validates that a live model can fill the same structure reliably across
multi-turn conversation before Step 9 consumes it.

Phase 1 intents:

```text
feature_weight
group_similar
group_dissimilar
merge_clusters
anchor_point
ignore_cluster
split_cluster
reclassify_outlier
needs_clarification
non_actionable
meta_query
```

Example:

```json
{
  "version": 4,
  "constraints": [
    {
      "id": "c3",
      "intent": "group_similar",
      "group_a": {"source": "selection_group", "ref": "group_001"},
      "group_b": {"source": "cluster", "ref": "cluster_2"}
    }
  ],
  "last_delta": {
    "operations": [
      {"op": "add", "constraint_id": "c3"}
    ]
  }
}
```

If the user input is vague, incomplete, irrelevant, or too general, the system
must not invent a hard constraint. Step 8 returns clarification or
non-actionable output. Step 8.5 additionally stores relevant fragments in a
draft, asks focused follow-up questions, and keeps irrelevant content out of
the active instruction state.

## Development Order

Follow the process in `docs/process.md`.

Current planned order:

1. `dashboard_shell`
   - create the Flask app shell and module lab first.

2. `data_workspace`
   - make dataset and feature matrix visible in Flask.

3. `projection`
   - make MDS output visible as an SVG/table in Flask.

4. `algorithm_adapters`
   - make SSDBCODI adapter outputs visible through the existing clustering/outlier schemas.
   - keep the legacy LOF-then-KMeans provider available for comparison.

5. `selection`
   - make selected/unselected state interactive in Flask.
   - keep action/source/mode fields extensible for future selection gestures.
   - let users save and restore named selections without turning them into semantic labels.

6. `labeling`
   - convert selected point IDs into manual cluster/outlier annotations.

7. `scatterplot`
   - render projected points, clusters, outliers, selection, and manual label state.

8. `chatbox`
   - build chat UI with mock or real selection context.

9. `intent_instruction`
   - compile messages into structured instructions (shared by both update paths; emits the full shared + Path B-only intent set) with a deterministic backend.

10. Step 8.5 runtime validation
    - connect the first live model runtime, defaulting to Ollama `qwen2.5:14b`.
    - validate memory, partial-information completion, relevance filtering, and provider diagnostics before Step 9 begins.

11. Path A: `metric_learning_adapter` / Path B: `direct_feedback_adapter`
   - 9A `metric_learning_adapter`: convert shared structured instructions into pair-based metric-learning constraints.
   - 9B `direct_feedback_adapter`: convert shared structured instructions (including `split_cluster` and `reclassify_outlier`) into a SSDBCODI-native `DirectFeedbackPlan` (seed updates, feature_scale, n_clusters, excluded/merged clusters).

12. Path A: `metric_refinement_orchestrator` / Path B: `direct_refinement_orchestrator`
    - 10A `metric_refinement_orchestrator`: fit metric `M`, apply `L = chol(M)` as linear pre-transform, rerun projection and algorithm adapters on `X · L`, track Path A rollback history.
    - 10B `direct_refinement_orchestrator`: rerun SSDBCODI directly with merged seeds and param overrides from the plan, rerun projection only when geometry changed, track Path B rollback history.

13. `strategy_comparison` workflow
    - run the same structured feedback through both orchestrators and render outputs side-by-side with a per-point diff.

14. integrated dashboard
    - combine already-tested modules and expose refinement triggers that choose which downstream path to run.

## Testing Expectations

For each module, do all three:

1. Unit tests:

```powershell
python -m unittest discover -s tests
```

2. Flask route tests:

Use Flask test client to verify debug pages and APIs.

3. Manual browser check:

```powershell
python run.py
```

Then open:

```text
http://127.0.0.1:5000/modules/<module_name>/
```

Confirm the module's visible output, state preview, and interactions work.

## Documentation Map

Read these before coding:

```text
README.md
docs/overview.md
docs/flask_app.md
docs/process.md
docs/module_debug_checklist.md
docs/integration_testing.md
docs/state_and_api_contracts.md
docs/modules/<module_name>/design.md
```

## Rule for Future AI Agents

Do not jump straight into the full dashboard.

For the current module:

1. read the relevant docs.
2. implement only that module.
3. add or update its Flask debug page.
4. add unit tests and Flask route tests.
5. make sure the module can be inspected in the browser.
6. update the module design document if behavior changes.
