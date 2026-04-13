# Refinement Orchestrator Module Design

## Purpose

The refinement orchestrator coordinates the end-to-end update flow after valid user feedback.

It does not own the implementation of metric learning, projection, clustering, outlier detection, labeling, chat, or scatterplot. It calls other modules in the correct order.

## Responsibilities

1. Receive actionable structured instructions.
2. Accept instructions from labeling or intent instruction.
3. Call metric-learning adapter.
4. Trigger updated projection.
5. Rerun clustering through algorithm adapters.
6. Rerun outlier detection through algorithm adapters.
7. Return updated dashboard state.
8. Provide a Flask timeline page for debugging the update flow.

## Not Responsible For

1. Parsing natural language.
2. Rendering scatterplot points.
3. Implementing metric learning internals.
4. Implementing clustering or outlier detection.
5. Owning selection state.
6. Owning manual label state.

## Target Files

```text
app/modules/refinement_orchestrator/
  __init__.py
  schemas.py
  service.py
  fixtures.py
  routes.py
  templates/refinement_orchestrator/index.html

tests/modules/refinement_orchestrator/
  test_service.py
  test_routes.py
```

## Main Flow

```text
structured instruction from labeling or intent
  -> metric-learning adapter
  -> updated metric state or representation
  -> projection
  -> clustering adapter
  -> outlier adapter
  -> dashboard state
```

## Run Result Contract

```json
{
  "refinement_run_id": "refine_001",
  "status": "completed",
  "steps": [
    {
      "name": "metric_learning",
      "status": "completed"
    },
    {
      "name": "projection",
      "status": "completed"
    }
  ],
  "outputs": {
    "projection_id": "projection_002",
    "cluster_run_id": "cluster_002",
    "outlier_run_id": "outlier_002"
  }
}
```

## Flask Routes

```text
/modules/refinement-orchestrator/                 orchestrator debug page
/modules/refinement-orchestrator/health           module health
/modules/refinement-orchestrator/api/run          run mock refinement
/workflows/refinement-loop/                       full refinement workflow demo
```

## Flask Debug Page Requirements

The page should show:

1. sample structured instruction.
2. run button.
3. timeline of each orchestration step.
4. intermediate payloads.
5. final updated state.
6. error state when a step fails.

## Testing

Unit tests:

1. actionable instruction triggers expected step order.
2. incomplete instruction stops before metric learning.
3. metric-learning failure returns diagnostics.
4. projection failure returns diagnostics.
5. algorithm adapter failures return diagnostics.

Flask route tests:

1. debug page returns 200.
2. run API returns timeline JSON.
3. invalid run request returns clear error.

Manual browser check:

1. open `/modules/refinement-orchestrator/`.
2. run sample refinement.
3. inspect the timeline.
4. verify failed and successful steps are visually clear.

## Completion Criteria

This module is complete when the refinement loop can be tested with mocks and inspected through a Flask timeline page.
