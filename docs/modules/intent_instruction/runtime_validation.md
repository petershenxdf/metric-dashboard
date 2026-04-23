# Intent Runtime Validation (Step 8.5)

## Purpose

Step 8.5 is the first stage that actually connects a live model runtime.

Step 8 already proves the chat-to-structure compiler boundary, but it still
uses `MockLlmProvider`. That is not enough for the project goals. Before Step 9
adapters consume chat-derived instructions, the system must prove that a real
model can:

1. understand wording variants,
2. maintain usable cross-turn memory,
3. separate relevant from irrelevant content,
4. convert incomplete user messages into structured draft state,
5. ask focused follow-up questions until the instruction is usable,
6. emit schema-valid structured output that later modules can trust.

This document defines that gate.

## Scope

Step 8.5 covers:

1. live provider runtime integration,
2. runtime configuration and provider abstraction,
3. real scatterplot + selection + labeling integration,
4. conversation memory design,
5. relevance filtering and partial-information handling,
6. UI and workflow requirements,
7. evaluation packs and acceptance gates before Step 9.

Step 8.5 does not:

1. choose Path A vs Path B,
2. run metric learning,
3. run SSDBCODI refinement,
4. bypass the existing `StructuredInstruction` schema,
5. move memory ownership into chatbox.

Step 8.5 must, however, consume the real upstream visual state. It is not a
text-only lab. The workflow should embed the real `scatterplot`, `selection`,
and `labeling` module boundaries so language grounding can be validated against
the same visual state the user actually sees.

## Current Implementation Status

The current repository already ships a working first Step 8.5 implementation:

1. `/workflows/intent-runtime-validation/` is live as the composite validation
   workflow.
2. The workflow uses the real grounded `selection`, saved `selection_groups`,
   real `labeling` annotations, and the effective cluster/outlier state derived
   from the current analysis view.
3. The live default provider is Ollama `qwen2.5:14b`.
4. Runtime defaults are read from the repo-root `.env` file, while the workflow
   form can override those values for the current in-memory session.
5. The chat panel exposes two reply display modes on the same workflow page:
   - `processed`: show the workflow's normal final reply after route/extract
     handling and required-slot gating.
   - `raw`: show a separate freeform model-authored reply for the same turn.
6. Prompt templates are file-backed and loaded from:
   - `prompts/intent_instruction/ollama/route_prompt.txt`
   - `prompts/intent_instruction/ollama/extract_prompt.txt`
   - `prompts/intent_instruction/ollama/reply_prompt.txt`
7. Runtime artifacts are persisted per session under:
   - `runtime_data/intent_runtime_validation/<dataset_id>/<session_id>/`
8. Persisted artifacts include runtime config, chat state, grounded state,
   memory state, provider diagnostics, interaction history, and the exact last
   route/extract/reply prompts sent to the live model.

The current implementation also has one important known risk: some live
multi-turn slot-answer cases can still hit route-timeout fallback even when the
final structured result is correct. That means the semantic path may succeed
while the route stage is not yet as stable as the extract stage.

## Default Runtime

The default first live runtime is:

```text
provider: ollama
model: qwen2.5:14b
base_url: http://127.0.0.1:11434
keep_alive: 30m
```

Those values are the local defaults shipped in `.env`. Supported keys:

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

Editing `.env` changes the next app start's default runtime. The workflow UI
still allows session-scoped overrides without editing files.

This is only the default. The design must leave room for:

1. other Ollama models,
2. remote HTTP-served local models,
3. online models such as OpenAI / Anthropic / future providers.

The provider abstraction must avoid hard-coding Ollama-specific assumptions
into the workflow design.

## Runtime Contract

The live runtime still satisfies the existing inner `LlmProvider` protocol:

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

Step 8.5 adds runtime-facing configuration around that protocol:

```json
{
  "provider_kind": "ollama",
  "model_name": "qwen2.5:14b",
  "base_url": "http://127.0.0.1:11434",
  "keep_alive": "30m",
  "timeout_seconds": 45,
  "temperature": 0.1,
  "max_output_tokens": 800,
  "allow_mock_fallback": true,
  "response_mode": "processed"
}
```

Required runtime behaviors:

1. health check,
2. explicit model label,
3. model keep-alive so repeated turns do not cold-load Ollama on every message,
4. timeout handling,
5. invalid-JSON diagnostics,
6. prompt-template visibility for debugging,
7. persisted prompt artifacts for replay,
8. room for future authentication when the provider is online.

`response_mode` is intentionally a display-only runtime setting. Switching
between `processed` and `raw` must not change:

1. the grounded context sent to the provider,
2. the transcript/history window sent to the provider,
3. the structured memory payload,
4. the final `StructuredInstruction` state transition rules.

## Prompt and Artifact Debugging

Prompt debugging is a first-class Step 8.5 requirement, not an afterthought.

Current implementation rules:

1. route, extract, and reply prompts are stored as standalone text templates in
   `prompts/intent_instruction/ollama/`,
2. the workflow persists the exact last rendered route, extract, and reply
   prompt text for each session,
3. the diagnostics payload reports the prompt template paths plus the last
   prompt text and prompt size,
4. a prompt change should therefore be auditable without scraping console logs.

This is important because wording failures can come from three different
layers:

1. route classification,
2. extraction / slot filling,
3. service-side final reply generation.

Those layers should be debugged separately.

## Step Boundary

The intended sequence is:

```text
Step 7 chatbox intake
  -> Step 8 deterministic compiler boundary
  -> Step 8.5 live runtime validation
  -> Step 9A / Step 9B adapters
```

Meaning:

1. Step 7 proves intake.
2. Step 8 proves the structure contract.
3. Step 8.5 proves that a live runtime can reliably fill that same contract.
4. Step 9 starts only after Step 8.5 passes.

## Upstream Module Integration

Step 8.5 must integrate the real upstream modules that ground language:

1. `scatterplot`
   - Real plot render payload.
   - Real point positions, cluster colors, and outlier markers.
   - Real click and rectangle selection interactions.

2. `selection`
   - Real active selection state.
   - Real selection groups.
   - Real selected/unselected counts and point IDs.

3. `labeling`
   - Real manual annotations.
   - Real label context exposed to chat/runtime.
   - Real point-level cluster/outlier overrides already visible in the plot.

This workflow should not create parallel mock copies of those states. The value
of Step 8.5 is precisely that the user can point at what they see on the plot,
change selection, apply labels, and then verify whether the live model grounded
the message correctly.

### Ownership Rule

Even when embedded on one page, ownership stays unchanged:

1. `scatterplot` owns rendering only.
2. `selection` owns selection truth.
3. `labeling` owns manual annotations.
4. `chatbox` owns chat history display and message intake.
5. `intent_instruction` owns memory, draft, facts, routing, and final
   structured instruction state.

Step 8.5 is therefore a composite workflow, not a state-ownership merge.

## Memory Design

The system should not treat raw chat history as its only memory.

Step 8.5 introduces a structured memory owned by `intent_instruction`.

### Memory Layers

1. Transcript log
   - Append-only.
   - Every user and assistant turn is stored.
   - Used for audit and replay, not as the full prompt every time.

2. Rolling summary
   - Short model-facing summary of the current conversation state.
   - Updated after each turn.
   - Used to keep prompts compact.

3. Working memory
   - Current dataset id, visible features, active selection groups, current
     cluster/class aliases, latest referenced point IDs, latest clarification
     target, and other short-lived context.

4. Extracted facts
   - Structured facts derived from user turns.
   - Each fact stores provenance, confidence, and status.

5. Instruction draft
   - Structured but incomplete candidate instruction.
   - Holds relevant fields that are present now plus a list of missing fields.

6. Irrelevant-turn log
   - Tracks off-topic chatter and other discarded turns for audit.
   - Does not feed the active draft except through a short summary when needed.

### Fact Schema

Suggested shape:

```json
{
  "fact_id": "fact_012",
  "source_turn_id": "turn_008",
  "kind": "target_cluster",
  "value": "cluster_2",
  "confidence": 0.81,
  "status": "tentative"
}
```

Status values should include at least:

```text
tentative
confirmed
retracted
ignored
```

### Draft Schema

Suggested shape:

```json
{
  "draft_id": "draft_003",
  "intent": "group_similar",
  "filled_slots": {
    "group_a": {"source": "selection_group", "ref": "group_001"}
  },
  "missing_slots": ["group_b"],
  "relevant_fragments": [
    {
      "turn_id": "turn_010",
      "text": "make this group closer",
      "status": "usable_but_incomplete"
    }
  ],
  "ignored_fragments": [
    {
      "turn_id": "turn_010",
      "text": "by the way the plot colors are weird",
      "reason": "ui_comment_not_instruction"
    }
  ]
}
```

The key rule is simple:

Incomplete instructions live in draft state, not in final
`StructuredInstruction`.

## Relevance and Turn Classification

Step 8.5 must separate message handling into these practical buckets:

1. `off_topic`
   - Irrelevant to clustering, class structure, outliers, features, or current
     dataset state.
   - Logged, answered politely, does not mutate draft or final instruction.

2. `meta_query`
   - User is asking about current state rather than giving a new constraint.
   - Example: `how many clusters`, `how many class`, `what classes do we have`.
   - Answered from current state, no instruction mutation.

3. `relevant_but_incomplete`
   - Contains useful information but not enough to finalize the instruction.
   - Extract usable fragments into draft state and ask one focused follow-up.

4. `actionable`
   - Contains enough information to build a valid `InstructionDelta`.
   - Draft can be promoted to final structured instruction.

5. `correction_or_override`
   - Revises an earlier tentative or confirmed fact.
   - Requires provenance-aware update rather than blind overwrite.

The system must explicitly distinguish:

```text
irrelevant
relevant but incomplete
relevant and actionable
```

That distinction is central to this step.

## Reply Interpretation Rule

The raw route output shown in diagnostics is not always the final assistant text
shown in the chatbox.

Current pipeline behavior:

1. route produces an intermediate `RouterResult`,
2. extract may still run to build or complete draft state,
3. required-slot gating may override the raw route clarification,
4. successful commits currently end with a deterministic service-side action
   confirmation rather than a free-form model-authored confirmation.

The new raw-mode toggle exists precisely so both layers can be audited on the
same page:

1. `processed` mode shows the user-facing workflow reply,
2. `raw` mode shows a separate direct-AI reply generated from the same
   grounded context and memory,
3. both modes still come from the same grounded request and same committed
   instruction state.

So if diagnostics show a raw clarification such as "Do you want me to split or
group these points within cluster 2?", the chatbox may still show a different
final reply if:

1. the route was upgraded by draft-memory resolution,
2. extraction produced a valid delta,
3. the service committed the delta and returned its own confirmation text.

## Wording Robustness

The runtime must normalize wording variants before downstream handling.

Minimum alias families:

1. `cluster`, `clusters`
2. `class`, `classes`
3. `group`, `groups`
4. `category`, `categories`

Example expectation:

```text
how many clusters
how many class
how many classes are there
what classes do we have
```

All of the above should route to the same meta-query family when the user is
asking for current cluster/class count or names.

This does not mean every alias is semantically identical everywhere. The router
should normalize enough to answer the intent correctly while preserving the raw
turn for audit.

## Partial-Information Workflow

When the user message is incomplete, the system should follow this sequence:

1. classify the turn,
2. extract relevant fragments,
3. update extracted facts,
4. update or create an `InstructionDraft`,
5. list missing required fields,
6. ask one focused clarification question,
7. wait for the next turn,
8. merge the next turn into the draft,
9. promote to final `StructuredInstruction` only when required fields are complete.

Example:

```text
User: make these more similar
System:
  - usable fragment: target group = current selection
  - missing field: comparison target
  - ask: "Which cluster or group should they be closer to?"
```

Later:

```text
User: cluster 2
System:
  - fills missing slot group_b = cluster_2
  - promotes draft to final group_similar instruction
```

## Structured Output Rule

Step 8.5 must continue to emit the same downstream-facing structures defined by
Step 8:

1. `RouterResult`
2. `InstructionDelta`
3. `StructuredInstruction`

It may add intermediate state such as memory, facts, and drafts, but it must
not invent a separate downstream payload just for the live runtime.

That keeps Step 9 adapters stable.

## UI and Workflow Design

Step 8.5 gets its own workflow page:

```text
/workflows/intent-runtime-validation/
```

The page should feel like a deliberate pre-Step-9 lab, not a one-off debug
form. It must validate whether language is grounded in the actual visual state,
not in a detached JSON panel alone.

### Page Layout

Recommended layout:

1. Top control band
   - dataset selector,
   - provider selector,
   - model name,
   - health status,
   - run-evaluation controls,
   - reset controls,
   - current dependency mode badges (`real` for scatterplot / selection /
     labeling, runtime provider clearly labeled).

2. Main visual area
   - a wide real `scatterplot` panel as the visual anchor of the page,
   - visible point IDs on hover or inspection,
   - cluster color legend,
   - outlier legend,
   - selection-state cues,
   - current label overlays when present.

3. Interaction rail beside the plot
   - current selection context,
   - selection groups,
   - current label context,
   - lightweight label actions or read-only label history,
   - current visual references such as `selected_points`, `cluster_2`,
     `outlier_set`, and saved group names.

4. Conversation panel
   - chat surface,
   - example prompts,
   - processed/raw reply toggle,
   - per-turn router outcome,
   - assistant clarification or answer.

5. Memory and output panel
   - transcript summary,
   - confirmed facts,
   - tentative facts,
   - irrelevant-turn log,
   - current clarification target,
   - draft state with missing slots,
   - latest `InstructionDelta`,
   - current `StructuredInstruction`.

6. Evaluation and diagnostics panel
   - pack list,
   - per-scenario status,
   - latency summary,
   - schema-validity summary,
   - provider failure diagnostics,
   - latest replay transcript and diff against expected outcome.

### Why Scatterplot Must Be Embedded

Without the real plot on the same page, several critical validations are weak:

1. whether `these points` really maps to the visible selected set,
2. whether `that cluster` matches what the user sees on screen,
3. whether label changes are reflected before the next model turn,
4. whether the model can answer visual meta-queries grounded in current state,
5. whether a human can audit grounding errors without switching workflows.

Step 8.5 therefore needs a real, embedded scatterplot rather than a text-only
selection summary.

### Visual State Requirements

The plot panel must visibly encode:

1. current cluster assignments,
2. outlier markers,
3. active selection,
4. saved selection group availability,
5. manual label overrides when present,
6. stable point identity for audit.

At minimum, the user must be able to look at the page and verify:

1. what points are currently selected,
2. what cluster a referenced point/group belongs to,
3. whether a manual label already changed the effective state,
4. whether the model's interpreted target matches the visual target.

### Interaction Requirements

The page must support these interaction loops without leaving the workflow:

1. click or rectangle-select points on the plot,
2. observe selection context update immediately,
3. optionally save or restore a selection group,
4. apply or inspect manual labels,
5. send a natural-language message about the currently visible state,
6. inspect memory/draft/final output,
7. adjust the message or selection and rerun.

This loop is the core of Step 8.5. The workflow should feel like a live
grounding workbench, not like separate modules stitched together loosely.

### UI Rules

1. Chatbox remains the intake surface; this workflow adds visibility, not a new
   ownership model.
2. Provider controls must not hard-code Ollama into the page title or layout.
3. The scatterplot must be the primary visual anchor of the page, not a small
   debug thumbnail.
4. Selection and labeling panels must reflect real module state, not derived
   approximations.
5. Memory panels must clearly separate final state from tentative state.
6. Draft state must be visibly distinct from committed `StructuredInstruction`.
7. Irrelevant turns must be inspectable without cluttering the main chat flow.
8. The page must compose naturally with earlier and later module layouts so it
   can be used beside selection, labeling, scatterplot, and future Step 9
   workflows.
9. A user should be able to audit any grounding error by comparing three things
   on one screen: the plot, the current context, and the model's interpreted
   structured state.
10. Reply-mode switching must stay inside the same workflow UI; it should not
    fork into a second Step 8.5 page or a second provider pipeline.

## Visual Validation Scenarios

Step 8.5 should include explicit manual and replayable scenarios that exercise
the visual grounding path, not only the text path.

### Scenario Family A: Selection Grounding

1. Click-select a few points and ask `make these closer to cluster 2`.
   - Expected: the selected points become `group_a`; `cluster_2` resolves from
     the current plot state.

2. Rectangle-select a region and ask `these should be separate from that group`.
   - Expected: active selection becomes one group, referenced saved group or
     labeled group becomes the other.

3. Change selection after one turn, then ask a follow-up.
   - Expected: working memory must reflect whether `these` refers to the newest
     visible selection or to an already-confirmed draft target.

### Scenario Family B: Label Grounding

1. Manually assign selected points to `cluster_3`, then ask `how many clusters`.
   - Expected: the answer reflects the current effective visual state used by
     the workflow.

2. Mark points as outliers, then ask `ignore the outliers`.
   - Expected: the model uses current label context and visible outlier state.

3. Ask about `that outlier` immediately after labeling.
   - Expected: the model can ground the reference in current label context and
     visible marker state.

### Scenario Family C: Visual Meta-Queries

1. `how many clusters`
2. `how many class`
3. `what classes do we have`
4. `which points are selected now`
5. `is the selected group already labeled`

Expected behavior:

1. no mutation to final instruction state,
2. answer grounded in current scatterplot + selection + labeling state,
3. alias normalization works across cluster/class/group wording.

### Scenario Family D: Partial Visual References

1. User says `move these closer`.
   - Expected: current selection is extracted, missing comparison target remains
     in draft.

2. User says `split that cluster`.
   - Expected: currently referenced cluster is extracted if resolvable; if not,
     clarification asks which visible cluster is meant.

3. User mixes usable and unusable content: `these should be similar, also the
   blue color is ugly`.
   - Expected: instruction fragment retained, UI complaint routed to irrelevant
     log.

### Scenario Family E: Correction and Drift

1. User selects new points and says `actually I meant these`.
   - Expected: prior tentative selection-based fact is retracted or replaced
     with provenance.

2. User relabels points between turns.
   - Expected: later turns use the updated label context and the workflow makes
     that state transition visible.

3. User restores a saved selection group and refers to it by name.
   - Expected: saved group grounding beats stale temporary selection context.

### Scenario Family F: Provider and Rendering Failure

1. Provider unavailable while plot is still visible.
   - Expected: visual state remains inspectable; runtime panel shows failure.

2. Invalid JSON from provider.
   - Expected: no silent mutation; diagnostics panel shows schema failure.

3. Dataset or plot state changes mid-session.
   - Expected: stale drafts are either invalidated or flagged for confirmation.

## Evaluation Packs

Step 8.5 should ship replayable evaluation packs.

Minimum packs:

1. Paraphrase pack
   - Same intent, different wording and syntax.

2. Meta-query alias pack
   - `cluster` / `class` / `group` wording variants.

3. Irrelevant pack
   - Off-topic text, UI commentary, mixed relevant + irrelevant turns.

4. Partial-information pack
   - First turn incomplete, later turns fill missing slots.

5. Multi-turn memory pack
   - Later turns refer back to earlier groups, clusters, or selections.

6. Correction pack
   - User revises a prior statement and the system updates draft/facts
     correctly.

7. Provider failure pack
   - timeout,
   - invalid JSON,
   - unavailable model,
   - schema mismatch.

8. Visual grounding pack
   - real scatterplot, real selection changes, real label changes, and expected
     grounded references across turns.

9. State-drift pack
   - dataset switch, selection change, relabeling, or group restore between
     turns to ensure stale grounding is handled safely.

## Acceptance Gates Before Step 9

Step 9 should not begin until Step 8.5 passes these gates.

Required gates:

1. Live provider health succeeds for the default Ollama runtime.
2. Structured output is schema-valid on every curated validation case.
3. Meta-query alias cases behave consistently across wording variants.
4. Irrelevant turns do not pollute final instruction state.
5. Relevant-but-incomplete turns create draft state instead of being discarded.
6. Multi-turn completion can promote a draft to final structured instruction.
7. Corrections can update tentative or confirmed facts with provenance intact.
8. Real scatterplot, selection, and labeling state are visibly embedded on the
   same page and stay consistent with chat/runtime state.
9. UI clearly exposes provider state, memory state, draft state, and final state.

Suggested quantitative gates:

1. 100% schema-valid JSON on the validation suite.
2. 95%+ accuracy on meta-query alias cases.
3. 90%+ accuracy on paraphrase intent routing.
4. 90%+ accuracy on relevant-vs-irrelevant separation.
5. 100% of incomplete instructions remain drafts until required slots are filled.
6. 100% of visual-grounding evaluation cases must show the expected selected
   points / cluster references in the audit panels before Step 9 starts.

## Additional Design Requirements

The user marked this step as the highest-risk part of the project. The design
should therefore include a few extra safeguards:

1. Prompt and schema versioning
   - every evaluation result should record which prompt/schema version produced it.

2. Replayability
   - saved transcripts and evaluation packs should be rerunnable after prompt or
     model changes.

3. Provenance
   - every extracted fact and promoted instruction should point back to turn IDs.

4. Human visibility
   - uncertain updates should remain visible as tentative state instead of
     silently becoming committed state.

5. Provider diagnostics
   - latency, timeout, and invalid-JSON failures should be first-class outputs,
     not hidden console details.

6. Swap-ready runtime config
   - switching from `qwen2.5:14b` to another Ollama model or an online provider
   should change config, not workflow semantics.

7. Visual auditability
   - every grounded reference should be inspectable against the real plot and
     current selection/label state on the same page.

8. State drift safety
   - if selection, dataset, or labels change after a draft was formed, the
     workflow should make that drift visible rather than silently reusing stale
     grounding.

## Summary

Step 8.5 is where the project stops pretending a deterministic mock is enough.
It is the gate that proves the real model can understand user language, build
structured memory, recover from incomplete input, ignore irrelevant noise, and
produce trustworthy structured output before the refinement adapters are asked
to act on it.
