# Chatbox Module Design

## Purpose

The chatbox module is the dialogue interface for user feedback.

It receives user messages and displays responses, but it does not run clustering, outlier detection, MDS, or metric learning.

It is one path for user feedback. Direct point labels are handled by the labeling module, while chat text is sent to intent instruction and compiled into the same structured feedback family.

## Responsibilities

1. Display conversation history.
2. Display current selection context and selection groups.
3. Display recent manual label context when available.
4. Display the current accumulated `StructuredInstruction` (delta memory).
5. Display suggestion chips generated from current dataset context.
6. Accept user messages and forward them with context to intent instruction.
7. Show router-level responses (clarification, off-topic redirect, meta-query answer).
8. Expose a **refinement strategy selector** so the user can pick Path A (`metric_learning`) or Path B (`direct_ssdbcodi`) when triggering a refinement. The selector only affects which orchestrator runs; it does not filter intent extraction.
9. Provide standalone Flask testing with mock selection, label, and instruction context.

## Not Responsible For

1. Owning selection state.
2. Owning manual label state.
3. Owning structured instruction state (that belongs to intent instruction).
4. Parsing language internally.
5. Running clustering.
6. Running outlier detection.
7. Running metric learning.
8. Updating the scatterplot directly.

## Target Files

```text
app/modules/chatbox/
  __init__.py
  schemas.py
  service.py
  fixtures.py
  routes.py
  templates/chatbox/index.html
  static/chatbox/chatbox.js
  static/chatbox/chatbox.css

tests/modules/chatbox/
  test_service.py
  test_routes.py
```

## Chat History Policy

Chat history is stored for display, not for replay into the LLM.

1. Full history is kept in memory per session for the UI.
2. Only the last N turns (default N=3) are forwarded to intent instruction with each new message.
3. The real cross-turn memory is the `StructuredInstruction` state owned by intent instruction, not the raw history.

This policy keeps prompts short and compatible with small local models.

## Message Request Contract

```json
{
  "message": "move group A closer to cluster 2",
  "selection_context": {
    "selected_point_ids": ["p1", "p7"],
    "unselected_point_ids": ["p2", "p3"],
    "selected_count": 2,
    "unselected_count": 2
  },
  "selection_groups": [
    {"group_id": "group_001", "group_name": "group A", "point_ids": ["p1", "p7"]}
  ],
  "label_context": {
    "active_annotations": []
  },
  "history_window": [
    {"role": "user", "text": "these points are similar"},
    {"role": "assistant", "text": "noted as group_similar"}
  ]
}
```

## Response Contract

```json
{
  "reply": "Added group_similar between group A and cluster 2.",
  "router_category": "on_topic_actionable",
  "delta": {
    "operations": [
      {"op": "add", "constraint_id": "c3"}
    ]
  },
  "current_instruction_version": 4,
  "requires_followup": false,
  "followup_question": null
}
```

For off-topic or ambiguous messages, `delta` is `null` and `reply` contains a redirect or clarification.

## Suggestion Chips

The chatbox generates a small set of suggestion chips per turn from dataset context. Examples:

1. "Make feature `petal_length` more important"
2. "Pull cluster 1 and cluster 3 apart"
3. "Merge group A with cluster 2"
4. "Treat p42 as a typical point for cluster 1"

Clicking a chip sends the exact phrase as a normal message. Chips cover all Phase 1 intents. Chips for `split_cluster` and `reclassify_outlier` are only shown when the active refinement strategy is Path B (`direct_ssdbcodi`), since Path A rejects those intents with `intent_deferred`.

## Flask Routes

```text
/modules/chatbox/                       chatbox debug page
/modules/chatbox/health                 module health
/modules/chatbox/api/messages           submit message
/modules/chatbox/api/context            current selection and label context
/modules/chatbox/api/history            current chat history
/modules/chatbox/api/reset              clear chat history
/workflows/chat-intent/                 chatbox plus intent parser
```

## Flask Debug Page Requirements

The page should show:

1. chat history.
2. message input and send button.
3. suggestion chips derived from current dataset context and active refinement strategy.
4. current selection context panel.
5. current label context panel when available.
6. current `StructuredInstruction` preview panel (read from intent instruction API).
7. response output with router category visible.
8. a refinement strategy toggle (`metric_learning` / `direct_ssdbcodi`) visible near the chat input so the user knows which orchestrator the next refinement trigger will hit.
9. a note showing whether selection, label, and instruction context are mocked or real.
10. a provider status badge showing which LLM is active.

## Testing

Unit tests:

1. empty message is rejected.
2. valid message creates a chat turn.
3. selection context is included in downstream payload.
4. label context is included when available.
5. history window is truncated to the configured N turns.
6. chatbox service does not call clustering or outlier detection.
7. chatbox service does not mutate selection, labeling, or structured instruction state.

Flask route tests:

1. debug page returns 200.
2. context API returns selected/unselected data.
3. history API returns stored turns.
4. message API handles valid and invalid messages.
5. reset API clears history.

Manual browser check:

1. open `/modules/chatbox/`.
2. type a message.
3. confirm the message appears in history.
4. confirm selection and label context are visible.
5. confirm response shows router category.
6. confirm suggestion chips produce valid messages when clicked.
7. confirm off-topic messages do not update the `StructuredInstruction` panel.

## Completion Criteria

This module is complete when chat interaction can be tested visibly through Flask without depending on the full dashboard, and when the UI clearly separates raw chat history from the accumulated structured instruction state.
