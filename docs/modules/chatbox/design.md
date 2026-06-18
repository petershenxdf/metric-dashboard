# Chatbox Module Design

## Purpose

The chatbox module is the legacy Step 7/8 dialogue intake surface for user
feedback. It remains documented because it exists in the current repository,
but the active post-Step-8.5 product direction is to replace this surface with
`rule_panel`.

It reads selection and labeling context, shows conversation history, and
forwards each new message to a pluggable intent provider. It does not decide
downstream behavior.

Direct point labels still belong to the labeling module. Chat text is only one
way to produce feedback, and its compiled form is owned by
`intent_instruction`.

## Responsibilities

1. Display conversation history.
2. Display current selection context and selection groups.
3. Display recent manual label context when available.
4. Display the current `InstructionSnapshot` supplied by the active provider.
5. Display suggestion chips covering the full Phase 1 feedback vocabulary.
6. Accept user messages and forward them with context to an `IntentProvider`.
7. Show router-level responses such as clarification, off-topic redirect, or meta-query answer.
8. Support standalone Flask testing with real selection/label context and a mock provider.

## Not Responsible For

1. Owning selection state.
2. Owning manual label state.
3. Owning structured instruction state.
4. Parsing language internally.
5. Choosing downstream explanation behavior.
6. Running clustering.
7. Running outlier detection.
8. Running rule generation or LLM interpretation.
9. Updating the scatterplot directly.

## Step 7 Contract

Step 7 is intentionally strategy-agnostic.

Its job is to prove that chat intake is clean:

```text
selection + selection groups + label context + recent chat turns
  -> ChatMessagePayload
  -> IntentProvider.respond(...)
  -> ChatResponse + InstructionSnapshot
```

This module no longer defines the next build direction. The next product
surface after Step 8.5 is the rule panel, not an update adapter.

## Target Files

```text
app/modules/chatbox/
  __init__.py
  schemas.py
  service.py
  store.py
  state.py
  fixtures.py
  routes.py
  templates/chatbox/index.html
  providers/
    __init__.py
    base.py      # IntentProvider protocol
    mock.py      # Step 7 MockIntentProvider (keyword-based)

tests/modules/chatbox/
  test_service.py
  test_routes.py
```

The real intent module (`intent_instruction`, Step 8) satisfies the same
`IntentProvider` protocol, so `/workflows/chat-intent/` can swap it in without
changing chatbox-side code.

## Chat History Policy

Chat history is stored for display, not as the system's real long-term memory.

1. Full history is kept in memory per dataset for the UI.
2. Only the last N turns (default N = 3) are forwarded with each new message.
3. The real cross-turn memory is the `StructuredInstruction` state owned by
   `intent_instruction`, not the raw text history.
4. When Step 8.5 is added, structured conversation memory still belongs to
   `intent_instruction`; chatbox remains a short-window intake surface only.
5. The Step 8.5 workflow may choose to display either the processed workflow
   reply or a freeform direct-AI reply, but that is a workflow presentation
   choice, not a second chatbox execution path.

This keeps prompts short and keeps the chatbox boundary simple.

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
    {"role": "assistant", "text": "recorded group_similar"}
  ]
}
```

No explanation strategy is attached here. Step 7 only captures legacy feedback.

## Response Contract

```json
{
  "reply": "Recorded group_similar.",
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

For off-topic or ambiguous messages, `delta` is `null` and `reply` contains a
redirect or clarification.

## Suggestion Chips

Suggestion chips cover all Phase 1 intents:

1. `feature_weight`
2. `group_similar`
3. `group_dissimilar`
4. `merge_clusters`
5. `anchor_point`
6. `ignore_cluster`
7. `split_cluster`
8. `reclassify_outlier`

This suggestion-chip vocabulary belongs to the legacy chat-feedback direction.
Do not extend it for the new rule-panel direction unless the chat roadmap is
explicitly reopened.

## Flask Routes

```text
/modules/chatbox/                       chatbox debug page
/modules/chatbox/health                 module health
/modules/chatbox/api/messages           submit message
/modules/chatbox/api/context            current selection and label context
/modules/chatbox/api/history            current chat history
/modules/chatbox/api/reset              clear chat history and reset mock snapshot
/modules/chatbox/api/clear              clear chat history only
/workflows/chat-selection/              Step 7 intake workflow
/workflows/chat-intent/                 Step 8 compilation workflow
/workflows/intent-runtime-validation/   Step 8.5 runtime validation workflow
```

## Flask Debug Page Requirements

The page should show:

1. Chat history.
2. Message input and send button.
3. Suggestion chips derived from current dataset context.
4. Current selection context panel.
5. Current label context panel when available.
6. Current `InstructionSnapshot` preview panel.
7. Response output with router category visible.
8. A note showing whether selection, label, and instruction context are mocked or real.
9. A provider status badge showing which provider is active.

It should not show an update-strategy toggle at Step 7.

## Testing

Unit tests:

1. Empty message is rejected.
2. Selection context is included in downstream payload.
3. Label context is included when available.
4. History window is truncated to the configured N turns.
5. Chatbox service does not call clustering or outlier detection.
6. Chatbox service does not mutate selection, labeling, or structured instruction state.
7. Suggestion chips include the legacy full feedback vocabulary.

Flask route tests:

1. Debug page returns 200.
2. Context API returns selected/unselected data.
3. History API returns stored turns.
4. Message API handles valid and invalid messages.
5. Reset API clears history and resets the mock provider snapshot.
6. Clear API clears history only.

Manual browser check:

1. Open `/modules/chatbox/`.
2. Type a message.
3. Confirm the message appears in history.
4. Confirm selection and label context are visible.
5. Confirm response shows router category.
6. Confirm suggestion chips produce valid messages when clicked.
7. Confirm off-topic messages do not update the `InstructionSnapshot` panel.

## Completion Criteria

This module is complete when chat intake can be tested visibly through Flask
without depending on the full dashboard, and when the UI clearly separates raw
chat history from the accumulated structured instruction state.

## Step 7 Status

Implemented. Current behavior:

1. `/modules/chatbox/` and `/workflows/chat-selection/` render selection
   context, selection groups, label context, a read-only instruction snapshot,
   full-coverage suggestion chips, example messages, and chat history.
2. The intent provider is pluggable via the `IntentProvider` protocol. Step 7
   ships `MockIntentProvider` (deterministic keyword router + intent
   extractor). Step 8 swaps in the real `intent_instruction` module through
   the same protocol, and Step 8.5 validates a live model runtime without
   changing the chatbox boundary.
3. Chatbox reads selection and label context from the real `selection` and
   `labeling` debug stores and never mutates them. The instruction snapshot is
   owned by the provider, not by chatbox.
4. `/workflows/intent-runtime-validation/` now renders direct AI replies only.
   The chatbox still only displays and forwards grounded payloads; label and
   selection mutation remains owned by the explicit workflow controls.
5. Future work should move the product-facing surface to `rule_panel`, where
   generated decision-tree rules are interpreted by DeepSeek.
