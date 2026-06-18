# Metric Dashboard

## What This Project Is

Metric Dashboard is a local Flask dashboard for human-in-the-loop clustering,
anomaly detection, and rule-based explanation.

The app is meant to run locally, stay simple, and make every module visible for debugging. Deployment, production auth, cloud infrastructure, and complex frontend frameworks are not part of the current scope.

Run target:

```powershell
python run.py
```

Expected local URL:

```text
http://127.0.0.1:5001
```

The app defaults to port `5001` because macOS AirPlay / Control Center often
uses port `5000`. Override it with `PORT=5002 python run.py` when needed.

Live runtime defaults for the Step 8.5 LLM workflow are read from the repo-root
`.env` file. Supported providers are `deepseek`, `ollama`, and `mock`.
Supported keys:

```text
METRIC_DASHBOARD_LLM_PROVIDER
METRIC_DASHBOARD_LLM_MODEL
METRIC_DASHBOARD_OLLAMA_BASE_URL
METRIC_DASHBOARD_DEEPSEEK_BASE_URL
METRIC_DASHBOARD_DEEPSEEK_API_KEY
METRIC_DASHBOARD_OLLAMA_KEEP_ALIVE
METRIC_DASHBOARD_LLM_TEMPERATURE
METRIC_DASHBOARD_LLM_TIMEOUT_SECONDS
METRIC_DASHBOARD_LLM_MAX_OUTPUT_TOKENS
METRIC_DASHBOARD_LLM_ALLOW_MOCK_FALLBACK
```

Edit `.env` when you want to change the default model or provider endpoint, then
restart `python run.py`. The runtime form on
`/workflows/intent-runtime-validation/` still works as a session-only override.
For DeepSeek V4, use `METRIC_DASHBOARD_LLM_PROVIDER=deepseek` with
`METRIC_DASHBOARD_LLM_MODEL=deepseek-v4-flash` or
`METRIC_DASHBOARD_LLM_MODEL=deepseek-v4-pro`; the workflow page also exposes
two buttons for switching between Flash and Pro in the current session.
That workflow is now the last step of the original chat-intent direction. The
new direction after Step 8.5 is explanation-first: replace the chatbox surface
with a rule panel that turns current SSDBCODI clusters and anomalies into
decision-tree rules, then use DeepSeek to interpret those rules.

Prompt templates now live under the repo-root `prompts/` folder:

```text
prompts/
  intent_instruction/
    ollama/
      route_prompt.txt
      extract_prompt.txt
      reply_prompt.txt
  rule_interpretation/
    deepseek/
      label_guidance_prompt.txt
```

Anything in the app that needs prompt files should read from there instead of
module-local prompt folders. The rule-interpretation workflow uses
`deepseek-v4-pro` with thinking enabled and high reasoning effort when
`provider_kind=deepseek`.

## Product Goal

The dashboard helps a user inspect high-dimensional data, select points, review
SSDBCODI clustering/outlier results, and understand those results through
decision-tree rules plus LLM explanations.

The first product version has two main user-facing areas:

1. Scatterplot
   - Projects user data into 2D with MDS.
   - Shows default clustering results.
   - Shows default outlier detection results.
   - Colors points by cluster.
   - Marks outliers.
   - Lets the user select points.
   - Lets selected points become explicit cluster or outlier labels through the labeling module.

2. Rule Panel
   - Uses shallow decision-tree surrogates only to explain current SSDBCODI clusters and anomalies as feature-threshold rules.
   - Uses the uploaded `wine.mat` file as the default visual and test dataset for the current Step 8.6 path.
   - Shows rule conditions with raw wine feature names such as `alcohol`, `flavanoids`, and `proline`, not projected `x/y` coordinates.
   - Shows one or more rule cards for each cluster and anomaly group.
   - Displays rule quality signals such as coverage, purity, support, matched points, and exception points.
   - Uses DeepSeek to parse and explain generated rules, not to directly mutate labels, selection, clustering, or anomaly detection.

SSDBCODI remains responsible for clustering and outlier/anomaly detection.
Decision trees are explanation-only surrogate models: they learn rules from
SSDBCODI outputs, but they do not produce the real cluster assignments or
outlier flags.

## System Loop

The current post-Step-8.5 direction is explanation-first. The old branching
update roadmap has been removed from the active build plan.

```text
user data
  -> data workspace
  -> MDS projection
  -> clustering adapter
  -> outlier adapter
  -> scatterplot
  -> point selection
  -> direct labeling / annotation
  -> rule panel
      -> decision-tree surrogate rules for clusters
      -> decision-tree surrogate rules for anomalies
      -> rule cards with coverage / purity / support / exceptions
  -> DeepSeek rule interpretation
      -> next label priorities
      -> overlap / merge hypotheses
      -> split or new-cluster hypotheses
      -> anomaly and exception label review
      -> feature-threshold labeling strategy
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
/workflows/wine-dashboard/            Step 8.8 integrated rule dashboard
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
| `chatbox` | Legacy Step 7/8 intake surface; superseded by the rule-panel direction after Step 8.5 | `/modules/chatbox/` |
| `intent_instruction` | Legacy Step 8/8.5 provider runtime and prompt infrastructure; reusable for rule interpretation provider calls | `/modules/intent-instruction/` |
| `rule_panel` | Decision-tree rule cards for each SSDBCODI cluster and anomaly, plus interpretation preview | `/modules/rule-panel/` |

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
   - persists `rScore`, `lScore`, `simScore`, and `tScore` for rule-panel diagnostics and LLM evidence.
   - reuses the existing selection and labeling contracts: additive click/rectangle selection, black center dots for selected points, saved selection groups, and label controls limited to `cluster_1...cluster_n` plus `outlier`.
   - keeps label entry and execution separate: Apply Label saves pending labeling feedback; Run and Store recomputes and persists SSDBCODI.
   - returns dashboard-compatible `ClusterResult` and `OutlierResult` schemas and now backs the default `algorithm_adapters` provider boundary.
   - `/workflows/provider-feedback/` verifies the promoted provider boundary beside standalone SSDBCODI score diagnostics.

9. `chatbox`
   - legacy intake surface from the original Step 7/8 direction.
   - dialogue UI that reads selection, selection groups, and label context from the real `selection` and `labeling` debug stores without mutating them.
   - forwards messages through a pluggable `IntentProvider` protocol; Step 7 ships `MockIntentProvider` (deterministic keyword-based router + intent extractor) so the chatbox can be tested standalone. Step 8 (`intent_instruction`) now also satisfies the same protocol, so `/workflows/chat-intent/` can wire the real provider in without any chatbox code change while `/modules/chatbox/` keeps the mock for isolated testing.
   - owns chat history only; the `InstructionSnapshot` lives inside whichever provider is active, not inside chatbox, so the module never owns instruction truth.
   - forwards a truncated history window (default last 3 turns) plus selection/label/instruction context with each message.
   - renders suggestion chips for the legacy Phase 1 vocabulary. `split_cluster` and `reclassify_outlier` remain visible for historical coverage, but the active roadmap no longer routes them into old update adapters.
   - fallback responses explicitly mark themselves as coming from the mock provider so users aren't confused by keyword-matcher limitations.
   - `/workflows/chat-selection/` combines selection, labeling, and chatbox state on one page as the Step 7 intake check.

10. `intent_instruction`
    - legacy structured-intent module from the original chat-feedback direction.
    - owns `StructuredInstruction` state and the two-stage router + extractor pipeline behind `IntentInstructionProvider`, which satisfies both the inner `LlmProvider` protocol (route/extract) and the outer `IntentProvider` protocol expected by chatbox.
    - Step 8 ships `MockLlmProvider` (deterministic keyword-driven router + extractor) as the only LLM provider; Step 8.5 and later can plug live local/cloud models into the same `LlmProvider` protocol without code changes elsewhere.
    - emits all eight Phase 1 intents (`feature_weight`, `group_similar`, `group_dissimilar`, `merge_clusters`, `anchor_point`, `ignore_cluster`, `split_cluster`, `reclassify_outlier`) for the legacy chat-feedback path.
    - off-topic, meta-query, and ambiguous messages never mutate state; actionable messages append a versioned `InstructionDelta` with a real `constraint_id`.
    - `/modules/intent-instruction/` exposes `/health`, `/api/route`, `/api/compile`, `/api/state`, `/api/reset`, `/api/examples`. `/workflows/chat-intent/` wires the real `intent_instruction` module boundary into a chatbox shell so the instruction state can be observed across multiple turns as the Step 8 compilation check.
    - `InstructionSnapshot` (shared cross-module view) was promoted to `app/shared/schemas.py` so chatbox and intent_instruction can both consume it without layering violations.

Current post-Step-8.5 direction:

- Keep the current Step 8.5 live-runtime work as the provider foundation.
- Step 8.6 is implemented as `rule_panel`.
- Generate decision-tree surrogate rules from the current SSDBCODI cluster and
  anomaly outputs.
- Use `wine.mat` as the current Rule Panel visual/test fixture, with conditions
  expressed in raw wine feature names.
- Provide a deterministic interpretation preview with the same action-guidance
  schema that Step 8.7 connects to DeepSeek: recommendation, quantitative
  findings, suggested label actions, grounded evidence, and one category from
  the label/refinement taxonomy.

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
   - `/workflows/rule-panel-validation/` (scatterplot + SSDBCODI + decision-tree rule cards)
   - `/workflows/rule-interpretation/` (rule cards + categorized DeepSeek interpretation)
7. Integrated dashboards:
   - `/workflows/wine-dashboard/` (Step 8.8 wine.mat dashboard with category-first DeepSeek-ready rule interpretation, without chatbox)

See `docs/workflows.md` for the current workflow contract and grouping rules.

## Module Boundary Rules

1. Scatterplot does not parse chat text.
2. Chatbox does not run clustering, outlier detection, projection, rule generation, or LLM interpretation.
3. Selection state is owned by the selection module, not hidden inside scatterplot.
4. Labeling owns manual cluster/outlier annotations derived from selected points.
5. Scatterplot can expose label actions, but it must send them to labeling instead of owning label truth.
6. Rule panel receives computed analysis results and outputs read-only rules.
7. DeepSeek receives generated rules and returns categorized interpretations; it does not own clustering or labeling truth.
8. Rule interpretation should read like label guidance for a non-specialist user: choose a category, show concrete point ids, explain why those points are suspicious in ordinary language, and say how different labels would affect merge, boundary, split/new-cluster, anomaly, or rule-audit decisions.
9. Existing algorithms are accessed only through algorithm adapters.
10. Dashboard shell composes modules but does not own module internals.
11. Integration should happen through schemas, services, APIs, or workflow pages.

## Rule Interpretation

Generated rules should become stable, auditable explanations.

The new post-Step-8.5 plan uses decision-tree surrogates to translate
SSDBCODI-produced cluster and anomaly assignments into rules. Each rule is a
conjunction of feature thresholds that explains a current cluster or anomaly
state. The decision tree is not the clustering or outlier detector.

DeepSeek parses these generated rules into label/refinement guidance
categories:

```text
label_priority
boundary_review
overlap_merge_signal
split_or_new_cluster_signal
anomaly_label_review
exception_relabel_review
feature_label_strategy
rule_confidence_audit
```

Example:

```json
{
  "rule_set_id": "rules_001",
  "rules": [
    {
      "rule_id": "rule_cluster_2_001",
      "target_kind": "cluster",
      "target_id": "cluster_2",
      "conditions": [
        {"feature": "petal_length", "operator": ">", "threshold": 3.1}
      ],
      "coverage": 0.67,
      "purity": 0.88,
      "matched_point_ids": ["p3", "p7"],
      "exception_point_ids": ["p9"]
    }
  ]
}
```

The LLM must recommend what the user should label next. Rule-pair overlap,
boundary adjacency, exception rates, support, coverage, and purity are computed
before the prompt and included as `rule_guidance_metrics`; the model may not
invent features, thresholds, clusters, anomalies, rule IDs, or point IDs. If
two rules overlap, labels test merge/shared-boundary hypotheses. If they do
not overlap, the model must not recommend a merge from the rules alone.

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
   - legacy chat UI with mock or real selection context.

9. `intent_instruction`
   - legacy module that compiles messages into structured instructions with a deterministic backend.

10. Step 8.5 runtime validation
    - connect the live model runtime, defaulting to DeepSeek V4 Pro unless `.env` overrides it.
    - keep the provider/runtime foundation available for the new rule interpretation workflow.

11. Step 8.6 Rule Panel
    - implemented: generate decision-tree surrogate rules for each cluster and anomaly.
    - show rule cards with conditions, support, coverage, purity, matched points, and exceptions.
    - keep the panel read-only with respect to selection, labeling, and SSDBCODI state.

12. Step 8.7 Rule Interpretation
    - implemented: parse generated rule cards into categorized, quantitative `RuleInterpretation` output that guides the next labels.
    - expose `/workflows/rule-interpretation/` and `/modules/rule-panel/api/interpretation`.
    - provide one button per interpretation category; each button sends `focus_category` and returns that category's label/refinement recommendation.
    - use deterministic mock locally by default; `provider_kind=deepseek` uses the existing DeepSeek configuration.
    - classify responses into `label_priority`, `boundary_review`,
      `overlap_merge_signal`, `split_or_new_cluster_signal`,
      `anomaly_label_review`, `exception_relabel_review`,
      `feature_label_strategy`, and `rule_confidence_audit`.
    - require `recommendation`, `quantitative_findings`, and `suggested_label_actions`.
    - reject unknown categories, rule ids, raw features, thresholds, target ids, point ids, and action references.

13. Integrated Rule Dashboard
    - combine scatterplot, SSDBCODI diagnostics, rule panel, and categorized rule interpretation.
    - expose whether DeepSeek V4 Pro was used or deterministic fallback guidance is being shown.

Old branching update steps after Step 8.5 are removed from the active plan.
Do not implement them unless the roadmap is explicitly reopened.

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
http://127.0.0.1:5001/modules/<module_name>/
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
