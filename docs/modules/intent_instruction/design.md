# Intent Instruction Module Design

## Purpose

The intent instruction module converts user language into stable structured instructions.

It is the boundary between vague chat text and deterministic downstream metric-learning constraints.

Chat-derived instructions should use the same feedback instruction family as manual labels from the labeling module.

## Responsibilities

1. Route user messages through a robustness stage before extraction.
2. Decide whether the message is actionable, ambiguous, partial, off-topic, or a meta-query.
3. Resolve references using selection context, selection groups, and existing label context.
4. Compile actionable messages into structured feedback instructions via a replaceable LLM provider.
5. Produce delta updates to an evolving `StructuredInstruction` state instead of regenerating it from scratch.
6. Generate clarification prompts for incomplete or ambiguous messages.
7. Reject irrelevant messages as constraints.
8. Provide a Flask page for trying example messages.

## Not Responsible For

1. Rendering chat UI.
2. Running metric learning.
3. Running clustering.
4. Running outlier detection.
5. Rendering scatterplot points.
6. Owning manual label state.
7. Owning selection state or selection groups.

## Target Files

```text
app/modules/intent_instruction/
  __init__.py
  schemas.py
  router.py
  extractor.py
  providers/
    base.py
    mock.py
    local_qwen.py
    cloud_claude.py
  fixtures.py
  routes.py
  templates/intent_instruction/index.html

tests/modules/intent_instruction/
  test_router.py
  test_extractor.py
  test_providers.py
  test_routes.py
```

## Two-Stage Pipeline

The module never sends raw text straight to an extractor. Every message goes through two stages:

### Stage A: Router

Classifies the message into one of:

1. `on_topic_actionable` - proceed to extraction.
2. `on_topic_ambiguous` - extractor is skipped; a clarification question is returned.
3. `partial` - extractor returns a partial instruction with missing fields flagged.
4. `meta_query` - the user is asking about current state, not giving feedback; return an informational answer.
5. `off_topic` - polite redirect with suggested example phrases tied to the current dataset.

The router is a small classifier prompt that runs on the same LLM provider but with short output.

### Stage B: Extractor

Only runs when Stage A returns `on_topic_actionable` or `partial`.

Produces a `StructuredInstruction` delta constrained by JSON schema.

## Supported Intent Types (ITML-Aligned, Phase 1)

These intents map cleanly to pair-based metric learning constraints:

1. `feature_weight` - increase, decrease, or ignore a feature. Implemented through pre-scaling, not pair constraints.
2. `group_similar` - two groups should be closer together.
3. `group_dissimilar` - two groups should be farther apart.
4. `merge_clusters` - two or more existing clusters should be treated as one.
5. `anchor_point` - one reference point attracts a target group.
6. `ignore_cluster` - a cluster should be excluded from metric updates this round.

Plus non-extracting router outcomes:

7. `needs_clarification`
8. `non_actionable`
9. `meta_query`

## Deferred Intent Types (Phase 2)

The following intents do not map cleanly to metric-only updates and are intentionally excluded from Phase 1:

1. `split_cluster` - requires changing the clustering algorithm's `k` or running sub-clustering. Metric change alone does not force KMeans to split a cluster.
2. `reclassify_outlier` - LOF uses a fixed contamination threshold, so metric changes may not move a point across the boundary.

These intents will be revisited after the clustering and outlier detection providers are swapped for algorithms that can act on these signals directly. They are not blockers for Phase 1.

## Structured Instruction Schema

A `StructuredInstruction` is the current accumulated state, a list of constraint entries:

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

Delta operations are `add`, `remove`, and `modify`. Small models only need to emit the delta, not the full state.

## Group Reference Schema

Constraints reference groups through a small reference object rather than inline point IDs:

```json
{"source": "selection_group", "ref": "group_001"}
{"source": "cluster", "ref": "cluster_2"}
{"source": "outlier_set", "ref": "current"}
{"source": "selected_points", "ref": "current"}
{"source": "point_id", "ref": "p42"}
```

This keeps the structure stable when selection or cluster contents change between turns.

## Chat History Handling

The module does not feed full chat history to the LLM. The rule is:

1. The real memory is the `StructuredInstruction` itself.
2. Each turn the extractor receives `last_N_turns` (default 3), the current instruction snapshot, the new message, and a `dataset_context` summary (feature names, cluster IDs, selection group names).
3. The model outputs only the delta.
4. Service layer applies the delta to the instruction state and records it in refinement history.

This keeps prompts short and works for small local models like qwen2.5-14b.

## LLM Provider Protocol

```python
class LlmProvider(Protocol):
    def route(self, message: str, context: DatasetContext, history: list[Turn]) -> RouterResult: ...
    def extract(self, message: str, context: DatasetContext, history: list[Turn],
                current_instruction: StructuredInstruction) -> InstructionDelta: ...
```

Built-in providers:

1. `MockLlmProvider` - deterministic, used in unit tests.
2. `LocalQwenProvider` - default; qwen2.5-14b via Ollama or vLLM, JSON-schema constrained output.
3. `LocalSmallProvider` - qwen2.5-7b or phi fallback for constrained GPU memory.
4. `ClaudeProvider` - cloud fallback or quality upgrade path.
5. `OpenAIProvider` - cloud alt.

Provider is chosen from `settings.json` or environment variable. Module code depends only on the protocol.

## Robustness Guarantees

1. Low-confidence extractor output triggers clarification rather than silent constraint creation.
2. Vague references ("these points") require a non-empty selection or selection group; otherwise the router returns `on_topic_ambiguous` with a clarification question.
3. Off-topic messages return a polite redirect that lists example phrases derived from the current dataset state.
4. The frontend exposes a preview panel showing the current `StructuredInstruction`; users can edit or remove constraints directly.
5. Suggestion chips on the chatbox debug page are generated from `dataset_context` and produce known-valid intents when clicked.

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
```

## Flask Debug Page Requirements

The page should show:

1. example message buttons grouped by intent type.
2. free text input.
3. mock selection context and selection groups editor.
4. router classification result and confidence.
5. structured instruction state preview (editable).
6. last delta JSON.
7. clarification question when needed.
8. provider status (which LLM is active, mock vs real).

## Testing

Unit tests (router):

1. "today's weather" becomes `off_topic`.
2. "how many clusters are there" becomes `meta_query`.
3. "move these together" with empty selection becomes `on_topic_ambiguous`.
4. "make petal_length more important" becomes `on_topic_actionable`.

Unit tests (extractor, with MockLlmProvider):

1. "these points should be together" becomes `group_similar` delta.
2. "push cluster 1 away from cluster 3" becomes `group_dissimilar` delta.
3. "merge clusters 1 and 2" becomes `merge_clusters` delta.
4. "make feature sepal_width less important" becomes `feature_weight` delta with `direction: decrease`.
5. "ignore cluster 5" becomes `ignore_cluster` delta.
6. "treat p42 as a typical example for cluster 2" becomes `anchor_point` delta.
7. delta apply produces the expected `StructuredInstruction` state.

Unit tests (provider contract):

1. MockLlmProvider satisfies the `LlmProvider` protocol.
2. Provider factory returns the configured provider.

Flask route tests:

1. debug page returns 200.
2. route API returns router category for any input.
3. compile API returns delta JSON for valid input.
4. compile API returns clarification JSON for ambiguous input.
5. state API returns current instruction.
6. reset API clears state.

Manual browser check:

1. open `/modules/intent-instruction/`.
2. try each example message.
3. confirm router category matches expectation.
4. confirm delta and resulting state are both visible.
5. confirm vague messages return clarification, not constraints.
6. confirm off-topic messages do not mutate state.

## Completion Criteria

This module is complete when:

1. Router and extractor are independently testable with `MockLlmProvider`.
2. Phase 1 intents produce valid deltas that apply cleanly to instruction state.
3. Flask debug page exposes router result, delta, and current instruction state.
4. Provider protocol allows swapping local and cloud models without touching service code.
5. Split and reclassify intents are explicitly documented as deferred and not emitted by the extractor.
