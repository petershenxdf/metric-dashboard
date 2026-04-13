# Chatbox Module Design

## Purpose

The chatbox module is the dialogue interface for user feedback.

It receives user messages and displays responses, but it does not directly run clustering, outlier detection, MDS, or metric learning.

## Responsibilities

1. Display conversation history.
2. Display current selection context.
3. Accept user messages.
4. Send message plus selection context to the intent instruction module when available.
5. Show clarification or confirmation responses.
6. Provide standalone Flask testing with mock selection context.

## Not Responsible For

1. Owning selection state.
2. Parsing language internally beyond basic request handling.
3. Running clustering.
4. Running outlier detection.
5. Running metric learning.
6. Updating the scatterplot directly.

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

## Message Request Contract

```json
{
  "message": "I think these points should be one class",
  "selection_context": {
    "selected_point_ids": ["p1", "p7"],
    "unselected_point_ids": ["p2", "p3"],
    "selected_count": 2,
    "unselected_count": 2
  }
}
```

## Response Contract

```json
{
  "reply": "I interpreted this as same_class.",
  "structured_instruction": null,
  "requires_followup": false
}
```

If the intent module is not implemented yet, the response must clearly say that parsing is mocked or not connected.

## Flask Routes

```text
/modules/chatbox/                       chatbox debug page
/modules/chatbox/health                 module health
/modules/chatbox/api/messages           submit message
/modules/chatbox/api/context            current selection context
/workflows/chat-intent/                 chatbox plus intent parser
```

## Flask Debug Page Requirements

The page should show:

1. chat history.
2. message input.
3. current selection context panel.
4. response output.
5. structured instruction JSON preview if available.
6. a note showing whether selection context is mocked or real.

## Testing

Unit tests:

1. empty message is rejected.
2. valid message creates a chat turn.
3. selection context is included in downstream payload.
4. chatbox service does not call clustering or outlier detection.

Flask route tests:

1. debug page returns 200.
2. context API returns selected/unselected data.
3. message API handles valid and invalid messages.

Manual browser check:

1. open `/modules/chatbox/`.
2. type a message.
3. confirm the message appears in history.
4. confirm selection context is visible.
5. confirm response states whether intent parsing is active or mocked.

## Completion Criteria

This module is complete when chat interaction can be tested visibly through Flask without depending on the full dashboard.

