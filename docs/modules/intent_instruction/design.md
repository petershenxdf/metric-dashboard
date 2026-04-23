# Intent Instruction Module Design

## Purpose

The intent instruction module compiles chat-derived feedback into stable
structured instructions.

It is the boundary between conversational text and deterministic downstream
feedback objects. It should stay strategy-agnostic: Path A / Path B acceptance
starts after this module, not inside it.

Step 8 proves the compiler boundary with a deterministic backend. Step 8.5 is
the first stage that actually connects a live model runtime and validates
memory, relevance filtering, and partial-information accumulation.

## Responsibilities

1. Route user messages through a robustness stage before extraction.
2. Decide whether a message is actionable, ambiguous, off-topic, or a meta-query.
3. Resolve references using selection context, selection groups, and existing label context.
4. Compile actionable messages into structured feedback instructions through a replaceable backend.
5. Produce delta updates to an evolving `StructuredInstruction` state instead of regenerating it from scratch.
6. Generate clarification prompts for incomplete or ambiguous messages.
7. Reject irrelevant messages as constraints.
8. Provide a Flask page for trying example messages.

## Not Responsible For

1. Rendering chat UI.
2. Choosing Path A vs Path B.
3. Running metric learning.
4. Running clustering.
5. Running outlier detection.
6. Rendering scatterplot points.
7. Owning manual label state.
8. Owning selection state or selection groups.

## Step 8 Contract

Step 8 proves the compilation boundary:

```text
ChatMessagePayload
  -> router
  -> extractor
  -> InstructionDelta
  -> StructuredInstruction state
```

This step should answer one question:

```text
Can chat feedback become versioned, replayable structured instructions
without entangling the compiler with downstream refinement policy?
```

## Step 8.5 Runtime Validation Gate

Step 8 alone is not enough for the project goals because it still runs on
`MockLlmProvider`.

Step 8.5 is the first stage that connects a live runtime (default Ollama
`qwen2.5:14b`) and validates five things before Step 9:

1. paraphrase robustness across different wording and structure,
2. conversation memory that is stored in structured, auditable form,
3. partial-information extraction plus focused follow-up questions,
4. relevance filtering between irrelevant chatter and relevant-but-incomplete feedback,
5. UI visibility of provider state, memory state, draft state, and final output.

See `docs/modules/intent_instruction/runtime_validation.md` for the full Step
8.5 design. The Step 8.5 workflow reads its default provider/model/base URL
from the repo-root `.env` file and lets the user override them per session on
the runtime page. Prompt templates are loaded from the repo-root
`prompts/intent_instruction/ollama/` directory. That same page now offers a
processed/raw reply toggle:

1. `processed` shows the normal workflow reply after routing, extraction,
   required-slot checks, and deterministic confirmation handling.
2. `raw` shows a separate freeform model-authored reply for the same grounded
   request.
3. Both modes still use the same `ChatMessagePayload`, `DatasetContext`,
   memory payload, and `StructuredInstruction` commit path.

## Target Files

```text
app/modules/intent_instruction/
  __init__.py
  schemas.py
  router.py
  extractor.py
  service.py
  store.py
  providers/
    base.py
    mock.py
    ollama.py
  fixtures.py
  routes.py
  templates/intent_instruction/index.html

prompts/
  intent_instruction/
    ollama/
      route_prompt.txt
      extract_prompt.txt
      reply_prompt.txt
    # cloud providers remain swappable follow-on work

tests/modules/intent_instruction/
  test_router.py
  test_extractor.py
  test_providers.py
  test_service.py
  test_routes.py
```

## Two-Stage Pipeline

The module never sends raw text straight to an extractor.

### Stage A: Router

Current Step 8 categories:

1. `on_topic_actionable` - proceed to extraction.
2. `on_topic_ambiguous` - still build a draft candidate so missing slots can be
   tracked and a focused clarification can be returned.
3. `meta_query` - the user is asking about current state, not giving feedback.
4. `off_topic` - polite redirect with suggested example phrases.

`partial` remains reserved in the shared schema surface, but the current
implementation represents the same idea through structured draft state plus
`on_topic_ambiguous` follow-up handling.

### Stage B: Extractor

Actionable messages reach the extractor for final delta creation. Ambiguous but
relevant messages also reach the extractor in Step 8.5 so the system can store
an incomplete `proposed_delta`, keep grounded fragments, and ask for the next
missing slot without committing the final instruction state.

The extractor emits an `InstructionDelta`, not a fully regenerated instruction
state. The service layer applies the delta, allocates stable constraint IDs,
and advances the version counter.

## Supported Intent Types (Phase 1)

The intent module emits eight structured intent types.

### Shared intents

1. `feature_weight`
2. `group_similar`
3. `group_dissimilar`
4. `merge_clusters`
5. `anchor_point`
6. `ignore_cluster`

### Path B-only downstream intents

7. `split_cluster`
8. `reclassify_outlier`

The compiler still emits all eight intents. It does not pre-filter by path and
does not add a strategy-specific deferral note. Path A / Path B adapters make
the acceptance decision later.

## Structured Instruction Schema

A `StructuredInstruction` is the current accumulated state:

```json
{
  "version": 3,
  "constraints": [
    {
      "id": "c1",
      "intent": "feature_weight",
      "feature": "petal_length",
      "direction": "increase"
    },
    {
      "id": "c2",
      "intent": "group_similar",
      "group_a": {"source": "selection_group", "ref": "group_001"},
      "group_b": {"source": "cluster", "ref": "cluster_2"}
    }
  ],
  "last_delta": {
    "operations": [
      {"op": "add", "constraint_id": "c2"}
    ]
  },
  "confidence": 0.87,
  "router_category": "on_topic_actionable",
  "clarification_needed": false,
  "clarification_question": null,
  "raw_message": "move group A closer to cluster 2"
}
```

Delta operations are `add`, `remove`, and `modify`.

## Group Reference Schema

Constraints reference groups through a compact reference object:

```json
{"source": "selection_group", "ref": "group_001"}
{"source": "cluster", "ref": "cluster_2"}
{"source": "outlier_set", "ref": "current"}
{"source": "selected_points", "ref": "current"}
{"source": "point_id", "ref": "p42"}
```

This keeps the structure stable even when selection or cluster contents change
between turns.

## Chat History Handling

Step 8 does not treat full raw history as memory.

1. The real memory is the `StructuredInstruction` itself.
2. Each turn the extractor receives the last N turns (default 3), the current
   instruction snapshot, the new message, and a `DatasetContext` summary.
3. The model/backend outputs only the delta.
4. The service layer applies the delta to the instruction state.

Step 8.5 extends this with structured conversation memory owned by
`intent_instruction`: transcript, rolling summary, extracted facts with
provenance, incomplete draft state, and irrelevant-turn logging. Chatbox still
owns only UI history.

The Step 8.5 raw-mode chat display must therefore be treated as diagnostics,
not as an alternate compiler. The module still commits only the normal
structured outputs:

1. `RouterResult`
2. `InstructionDelta`
3. `StructuredInstruction`

## Two Protocol Layers

There are two protocols in this module.

### Inner: `LlmProvider`

The pluggable router/extractor backend:

```python
class LlmProvider(Protocol):
    label: str

    def route(
        self,
        message: str,
        context: DatasetContext,
        history: Sequence[Turn],
        memory_context: Mapping[str, object] | None = None,
    ) -> RouterResult: ...

    def extract(
        self,
        message: str,
        context: DatasetContext,
        history: Sequence[Turn],
        current_instruction: StructuredInstruction,
        memory_context: Mapping[str, object] | None = None,
    ) -> InstructionDelta: ...
```

Step 8 ships `MockLlmProvider` only: a deterministic keyword-based backend that
makes the pipeline fully testable without external dependencies. Step 8.5 adds
the first live provider runtime, defaulting to Ollama `qwen2.5:14b`, while
keeping the same protocol open to future local or cloud providers. The default
runtime config is file-backed through `.env`, so local model changes do not
require code edits.

### Outer: `IntentProvider`

The chatbox-facing protocol:

```python
class IntentProvider(Protocol):
    label: str
    def respond(self, payload: ChatMessagePayload) -> ChatResponse: ...
    def current_snapshot(dataset_id: str) -> InstructionSnapshot: ...
    def reset(dataset_id: str) -> None: ...
```

`IntentInstructionProvider` satisfies this outer protocol by:

1. Building `DatasetContext` from `ChatMessagePayload`.
2. Routing the message.
3. Extracting an `InstructionDelta` only for actionable messages.
4. Applying the delta to the module-owned store.
5. Returning a `ChatResponse` plus a narrower `InstructionSnapshot`.

## Robustness Guarantees

1. Vague references such as "these points" require a non-empty selection or selection group.
2. Off-topic messages never mutate instruction state.
3. Meta-queries never mutate instruction state.
4. The frontend exposes both the last response and the current `StructuredInstruction` state.
5. Suggestion chips on the chatbox debug page are generated so users can exercise every Phase 1 intent.

## Flask Routes

```text
/modules/intent-instruction/                       intent debug page
/modules/intent-instruction/health                 module health
/modules/intent-instruction/api/route              run router only
/modules/intent-instruction/api/compile            run full pipeline, return delta
/modules/intent-instruction/api/state              current StructuredInstruction
/modules/intent-instruction/api/reset              clear instruction state
/modules/intent-instruction/api/examples           example messages
/workflows/chat-intent/                            chat plus intent workflow
/workflows/intent-runtime-validation/              Step 8.5 live-model validation gate
```

## Flask Debug Page Requirements

The page should show:

1. Example message buttons grouped by intent type.
2. Free text input.
3. Dataset context summary.
4. Router classification result and confidence.
5. Structured instruction state preview.
6. Last delta JSON.
7. Clarification question when needed.
8. Provider status showing whether the backend is mock or real.

It should not expose a refinement-strategy selector at Step 8.

## Testing

Unit tests (router):

1. `"today's weather"` becomes `off_topic`.
2. `"how many clusters are there"` becomes `meta_query`.
3. `"move these together"` with empty selection becomes `on_topic_ambiguous`.
4. `"make petal_length more important"` becomes `on_topic_actionable`.

Unit tests (extractor, with `MockLlmProvider`):

1. Grouping messages become `group_similar` deltas.
2. Separating messages become `group_dissimilar` deltas.
3. Merge messages become `merge_clusters` deltas.
4. Feature-importance messages become `feature_weight` deltas.
5. Anchor references become `anchor_point` deltas.
6. Ignore-cluster messages become `ignore_cluster` deltas.
7. Split-cluster messages become `split_cluster` deltas.
8. Reclassify-outlier messages become `reclassify_outlier` deltas.
9. Applying a delta to a `StructuredInstruction` produces the expected next state.

Flask route tests:

1. Debug page returns 200.
2. Route API returns router category for valid input.
3. Compile API returns delta JSON for actionable input.
4. Compile API returns clarification JSON for ambiguous input.
5. State API returns current instruction.
6. Reset API clears state.

Manual browser check:

1. Open `/modules/intent-instruction/`.
2. Try each example message.
3. Confirm router category matches expectation.
4. Confirm delta and resulting state are both visible.
5. Confirm vague messages return clarification, not constraints.
6. Confirm off-topic messages do not mutate state.

## DatasetContext Schema

```python
@dataclass(frozen=True)
class DatasetContext:
    dataset_id: str
    feature_names: tuple[str, ...]
    cluster_ids: tuple[str, ...]
    outlier_point_ids: tuple[str, ...]
    selection_group_names: tuple[str, ...]
    selection_groups: tuple[Mapping[str, Any], ...]
    label_annotations: tuple[Mapping[str, Any], ...]
    analysis_context: Mapping[str, Any]
    selected_point_ids: tuple[str, ...]
    unselected_point_ids: tuple[str, ...]
```

The `IntentInstructionProvider` builds this from `ChatMessagePayload` plus the
grounded workflow state. In Step 8.5 that means the live provider can see:

1. current selected and unselected points,
2. saved selection groups,
3. manual label annotations,
4. effective cluster and outlier state,
5. point-level catalog data used for grounding and audit.

## Completion Criteria

This module is complete when:

1. Router and extractor are independently testable with `MockLlmProvider`.
2. Phase 1 intents produce valid deltas that apply cleanly to instruction state.
3. Flask debug page exposes router result, delta, and current instruction state.
4. `LlmProvider` is documented so real backends can be plugged in later without touching `service.py`.
5. `IntentInstructionProvider` satisfies the chatbox `IntentProvider` protocol.
6. Path B-only intents are emitted here without forcing a path decision at compile time.
7. Step 8.5 design is documented so the first live runtime can be added without changing Step 7 ownership boundaries.
