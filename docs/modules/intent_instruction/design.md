# Intent Instruction Module Design

## Purpose

The intent instruction module converts user language into stable structured instructions.

It is the boundary between vague chat text and deterministic downstream metric-learning constraints.

## Responsibilities

1. Classify user messages.
2. Decide whether the message is actionable.
3. Resolve references using selection context.
4. Compile actionable messages into structured instructions.
5. Generate clarification prompts for incomplete messages.
6. Reject irrelevant messages as constraints.
7. Provide a Flask page for trying example messages.

## Not Responsible For

1. Rendering chat UI.
2. Running metric learning.
3. Running clustering.
4. Running outlier detection.
5. Rendering scatterplot points.

## Target Files

```text
app/modules/intent_instruction/
  __init__.py
  schemas.py
  classifier.py
  compiler.py
  fixtures.py
  routes.py
  templates/intent_instruction/index.html

tests/modules/intent_instruction/
  test_classifier.py
  test_compiler.py
  test_routes.py
```

## Message Categories

1. `actionable`
2. `actionable_incomplete`
3. `high_level_preference`
4. `non_actionable`

## Structured Instruction Schema

```json
{
  "instruction_type": "same_class",
  "status": "actionable",
  "target": {
    "source": "selected_points",
    "point_ids": ["p1", "p7"]
  },
  "parameters": {},
  "explicitness": "explicit",
  "requires_followup": false,
  "followup_question": null,
  "raw_message": "I think these points should be one class"
}
```

## Supported Instruction Types

1. `same_class`
2. `different_class`
3. `split_into_n_classes`
4. `merge_groups`
5. `is_outlier`
6. `not_outlier`
7. `needs_clarification`
8. `non_actionable`

## Flask Routes

```text
/modules/intent-instruction/                       intent debug page
/modules/intent-instruction/health                 module health
/modules/intent-instruction/api/compile            compile message
/modules/intent-instruction/api/examples           example messages
/workflows/chat-intent/                            chat plus intent workflow
```

## Flask Debug Page Requirements

The page should show:

1. example message buttons.
2. free text input.
3. mock selection context editor or preview.
4. classification result.
5. structured instruction JSON.
6. clarification question when needed.

## Deterministic First Version

Start with rules-based behavior:

1. keywords for same class.
2. keywords for split into N classes.
3. keywords for outlier/not outlier.
4. selected/unselected reference resolution.
5. deterministic clarification templates.

LLM behavior can be added later behind the same schema.

## Testing

Unit tests:

1. "these points should be one class" becomes `same_class`.
2. "unselected points should be split into three classes" becomes `split_into_n_classes`.
3. "these are outliers" becomes `is_outlier`.
4. vague relevant messages require clarification.
5. irrelevant messages become `non_actionable`.
6. empty selection with "these points" requires clarification.

Flask route tests:

1. debug page returns 200.
2. compile API returns structured JSON.
3. example API returns examples.

Manual browser check:

1. open `/modules/intent-instruction/`.
2. try each example message.
3. confirm JSON output matches expectation.
4. confirm vague messages do not become hard constraints.

## Completion Criteria

This module is complete when structured instruction generation is testable in code and inspectable through Flask.

