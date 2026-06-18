# Workflow Debug Map

## Purpose

Workflow pages are not final product screens. They are controlled integration
labs that prove module boundaries before the full dashboard is assembled.

Each workflow should answer one question:

```text
Can these modules exchange their real schemas and state without hidden coupling?
```

The workflow registry keeps stable URL slugs and adds three pieces of metadata:

1. `group` - what kind of debugging the workflow supports.
2. `step` - where it sits in the build sequence.
3. `debug_focus` - the specific contract the page is meant to verify.

## Current Workflow Groups

### Core Pipeline Smoke Tests

These pages validate the forward data path with minimal interaction.

| Step | Route | Purpose |
| --- | --- | --- |
| 1-2 | `/workflows/data-projection/` | Dataset rows become stable point IDs, feature matrix rows, and MDS coordinates. |
| 1-3 | `/workflows/default-analysis/` | Projection and SSDBCODI-backed `algorithm_adapters` output align by point ID. |

### State Boundary Probes

These pages isolate state ownership before visual workflows add more moving
parts.

| Step | Route | Purpose |
| --- | --- | --- |
| 4 | `/workflows/selection-context/` | Selection owns selected/unselected point IDs and exports a read-only context. |
| 5 | `/workflows/selection-labeling/` | Labeling consumes selection context and emits manual annotations plus structured feedback. |

### Visual Integration Tests

These pages combine real data, analysis, selection, labeling, and rendering.

| Step | Route | Purpose |
| --- | --- | --- |
| 1-4 | `/workflows/analysis-selection/` | Data, projection, SSDBCODI outliers/clusters, and selection share one SVG layer. |
| 1-5 | `/workflows/analysis-labeling/` | Manual labels are passed into SSDBCODI and reflected in effective analysis state. |
| 1-6 | `/workflows/scatter-selection/` | Scatterplot render payload preserves selection behavior after composition. |
| 1-6 | `/workflows/scatter-labeling/` | Full completed loop: render, select, label, rerun effective analysis. |

### Provider Diagnostics

This page verifies Step 6.5 provider promotion and score availability.

| Step | Route | Purpose |
| --- | --- | --- |
| 6.5 | `/workflows/provider-feedback/` | Compare the `algorithm_adapters` boundary with standalone SSDBCODI scores and seed diagnostics. |

### Feedback Pipeline

Step 7 is the legacy strategy-agnostic chat intake workflow. It proves that chatbox
can read selection and labeling context and forward messages without owning any
upstream state.

Step 8 promotes the chatbox boundary from the mock provider to the real
`intent_instruction` module, but the backend inside that module is still the
deterministic `MockLlmProvider` so the compiler boundary stays easy to debug.

Step 8.5 is the real-model runtime gate from the original chat direction. It
connects `intent_instruction` to a live provider runtime (default DeepSeek V4
Pro unless `.env` overrides it), validates SSDBCODI-grounded context, and keeps
the provider/runtime foundation that the new rule-interpretation workflow can
reuse.

| Step | Route | Purpose |
| --- | --- | --- |
| 7 | `/workflows/chat-selection/` | Chat UI reads current selection and labeling context, records mock intents, and proves the intake boundary without choosing a downstream update path. |
| 8 | `/workflows/chat-intent/` | Chat text becomes structured instruction deltas at the real `intent_instruction` module boundary, still using the deterministic Step 8 backend so compilation can be debugged independently of live-model behavior. |
| 8.5 | `/workflows/intent-runtime-validation/` | Live-model validation gate: real scatterplot + selection + labeling context, SSDBCODI scores/seeds/diagnostics, real provider runtime, grounded conversation memory, and direct AI replies. |

### Rule Workflows

These routes define the new post-Step-8.5 direction. Old branching update
workflows are no longer the active plan.

| Step | Route | Purpose |
| --- | --- | --- |
| 8.6 | `/workflows/rule-panel-validation/` | Working: render current SSDBCODI diagnostics beside decision-tree rule cards for SSDBCODI clusters and anomalies. |
| 8.7 | `/workflows/rule-interpretation/` | Working: show rule cards beside categorized label guidance: which points to label, why they need checking, how to label them, and optional DeepSeek V4 Pro provider mode. |
| 8.8 | `/workflows/wine-dashboard/` | Working: integrated `wine.mat` rule dashboard with data, projection, SSDBCODI, selection, labeling, scatterplot, rule cards, category-first rule interpretation, and explicit DeepSeek/mock status; chatbox is excluded. |

### Rule Workflow Rules

1. Rule workflows are read-only with respect to clustering, labeling, and
   selection state.
2. Rule cards explain current SSDBCODI output; decision trees do not replace
   `ClusterResult`, `OutlierResult`, or `SsdbcodiResult`.
3. DeepSeek interpretation must cite rule evidence and use only known feature
   names, thresholds, point ids, cluster ids, and anomaly ids.
4. Each category view must explain what that category examines before showing
   recommendations.
5. Main guidance should prioritize label targets, suspicion reasons, and
   point-level label guidance in ordinary labeling language; quantitative
   findings belong in audit details.
6. If a category has no related candidate points, the page must show a clear
   "no typical case" state instead of inventing a recommendation.
7. Integrated dashboard pages must expose whether DeepSeek V4 Pro was actually
   used or whether deterministic fallback guidance is being displayed.
8. Rule quality warnings should be visible when coverage or purity is low.

## Ordering Rule

The workflow index is ordered for debugging, not for UI polish:

1. prove schemas,
2. prove state ownership,
3. prove visual interaction,
4. prove provider diagnostics,
5. reserve future integration points.

Do not remove an older workflow just because a later workflow covers more
surface area. Smaller workflow pages are faster to debug when the full loop
breaks.

## Route Stability Rule

Keep existing workflow slugs stable unless there is a strong reason to break
links or tests. Prefer changing the registry title, purpose, group, and
`debug_focus` metadata over renaming URLs.

## Workflow Page Standard

Each working workflow should show:

1. included real modules,
2. dependency mode,
3. visible controls,
4. visible output,
5. JSON/state payload,
6. one clear debugging claim.

Placeholder workflows should still list included modules and debug focus so
future implementation work starts from an explicit contract.
