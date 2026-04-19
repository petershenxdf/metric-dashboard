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

### Future Workflows

These placeholders are intentionally visible so future work has a planned
integration path.

| Step | Route | Purpose |
| --- | --- | --- |
| 7 | `/workflows/chat-selection/` | Chat UI receives current selection context. |
| 8 | `/workflows/chat-intent/` | Chat text becomes structured instruction deltas. |
| 9 | `/workflows/instruction-constraints/` | Structured instructions and labels become metric-learning constraints. |
| 10 | `/workflows/refinement-loop/` | Metric fit, transformed projection, rerun analysis, and rollback history. |

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
