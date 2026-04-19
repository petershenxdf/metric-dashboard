from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import List

from app.modules.chatbox.providers.mock import MockIntentProvider
from app.modules.chatbox.schemas import (
    ChatMessagePayload,
    ChatResponse,
    MockInstructionSnapshot,
    REFINEMENT_STRATEGIES,
)
from app.modules.chatbox.service import (
    build_payload,
    clear_history,
    create_chatbox_store,
    get_chatbox_state,
    set_strategy,
    submit_message,
    suggestion_chips_for_strategy,
)
from app.modules.selection.schemas import SelectionContext, SelectionGroup


@dataclass
class SpyIntentProvider:
    """Deterministic provider that records invocations but does not mutate
    the chatbox store or any shared module state."""

    label: str = "spy"
    calls: List[ChatMessagePayload] = field(default_factory=list)
    snapshot_version: int = 0

    def current_snapshot(self, dataset_id: str) -> MockInstructionSnapshot:
        return MockInstructionSnapshot(version=self.snapshot_version, constraints=())

    def respond(self, payload: ChatMessagePayload) -> ChatResponse:
        self.calls.append(payload)
        self.snapshot_version += 1
        return ChatResponse(
            reply="noted",
            router_category="on_topic_actionable",
            delta={"operations": [{"op": "add", "constraint_id": "c1"}]},
            current_instruction_version=self.snapshot_version,
            provider_label=self.label,
        )

    def reset(self, dataset_id: str) -> None:
        self.snapshot_version = 0


class ChatboxServiceTests(unittest.TestCase):
    def setUp(self):
        self.context = SelectionContext(
            dataset_id="chatbox_test",
            selected_point_ids=("p1", "p2"),
            unselected_point_ids=("p3", "p4"),
        )
        self.empty_context = SelectionContext(
            dataset_id="chatbox_test",
            selected_point_ids=(),
            unselected_point_ids=("p1", "p2"),
        )
        self.store = create_chatbox_store("chatbox_test")
        self.provider = SpyIntentProvider()

    def test_empty_message_is_rejected(self):
        with self.assertRaises(ValueError):
            submit_message(
                self.store,
                self.provider,
                message="   ",
                selection_context=self.context,
            )

    def test_payload_includes_selection_context_and_groups(self):
        group = SelectionGroup(
            group_id="group_001",
            group_name="Group A",
            dataset_id="chatbox_test",
            point_ids=("p1", "p2"),
        )
        payload = build_payload(
            self.store,
            "make petal_length more important",
            selection_context=self.context,
            selection_groups=[group],
            label_context={},
        )
        self.assertEqual(payload.selected_point_ids, ("p1", "p2"))
        self.assertEqual(payload.unselected_point_ids, ("p3", "p4"))
        self.assertEqual(len(payload.selection_groups), 1)
        self.assertEqual(payload.selection_groups[0]["group_id"], "group_001")

    def test_payload_includes_label_context_when_available(self):
        payload = build_payload(
            self.store,
            "push cluster 1 away from cluster 2",
            selection_context=self.context,
            selection_groups=[],
            label_context={"annotation_count": 2, "active_annotations": [{"id": "a1"}]},
        )
        self.assertEqual(payload.label_context["annotation_count"], 2)
        self.assertEqual(len(payload.label_context["active_annotations"]), 1)

    def test_payload_includes_active_refinement_strategy(self):
        set_strategy(self.store, "direct_ssdbcodi")
        payload = build_payload(
            self.store,
            "split cluster 2",
            selection_context=self.context,
            selection_groups=[],
            label_context={},
        )
        self.assertEqual(payload.strategy, "direct_ssdbcodi")

    def test_history_window_is_truncated(self):
        for index in range(5):
            submit_message(
                self.store,
                self.provider,
                message=f"message {index}",
                selection_context=self.context,
            )
        # Store has 5 user + 5 assistant turns = 10 total.
        self.assertEqual(len(self.store.turns), 10)

        payload = build_payload(
            self.store,
            "next message",
            selection_context=self.context,
            selection_groups=[],
            label_context={},
            history_window_size=3,
        )
        self.assertEqual(len(payload.history_window), 3)
        # The window contains the most recent turns.
        self.assertEqual(payload.history_window[-1]["turn_id"], self.store.turns[-1].turn_id)

    def test_service_does_not_mutate_selection_or_label_context(self):
        original_selected = tuple(self.context.selected_point_ids)
        original_unselected = tuple(self.context.unselected_point_ids)
        label_context = {"annotation_count": 1, "active_annotations": [{"id": "a1"}]}
        label_context_snapshot = dict(label_context)

        submit_message(
            self.store,
            self.provider,
            message="merge clusters 1 and 2",
            selection_context=self.context,
            label_context=label_context,
        )

        self.assertEqual(self.context.selected_point_ids, original_selected)
        self.assertEqual(self.context.unselected_point_ids, original_unselected)
        self.assertEqual(label_context, label_context_snapshot)

    def test_service_does_not_mutate_provider_snapshot_directly(self):
        """The chatbox service never writes to instruction state; only the
        provider does. Verify by swapping to a provider that stays pinned
        at version 0 and observing submit_message reports that version."""

        @dataclass
        class FrozenProvider:
            label: str = "frozen"
            calls: int = 0

            def current_snapshot(self, dataset_id: str) -> MockInstructionSnapshot:
                return MockInstructionSnapshot(version=0, constraints=())

            def respond(self, payload):
                self.calls += 1
                return ChatResponse(
                    reply="ok",
                    router_category="on_topic_actionable",
                    delta=None,
                    current_instruction_version=0,
                    provider_label=self.label,
                )

            def reset(self, dataset_id: str) -> None:
                pass

        frozen = FrozenProvider()
        _, response = submit_message(
            self.store,
            frozen,
            message="merge clusters 1 and 2",
            selection_context=self.context,
        )
        state = get_chatbox_state(self.store, frozen)
        self.assertEqual(response.current_instruction_version, 0)
        self.assertEqual(state.instruction_snapshot.version, 0)
        self.assertEqual(frozen.calls, 1)

    def test_strategy_toggle_changes_payload_not_turns(self):
        set_strategy(self.store, "direct_ssdbcodi")
        before = len(self.store.turns)
        payload = build_payload(
            self.store,
            "treat these as similar",
            selection_context=self.context,
            selection_groups=[],
            label_context={},
        )
        # Building a payload should not add turns or produce chat history.
        self.assertEqual(len(self.store.turns), before)
        self.assertEqual(payload.strategy, "direct_ssdbcodi")

    def test_unsupported_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            set_strategy(self.store, "no_such_strategy")

    def test_clear_history_empties_turns_but_keeps_strategy(self):
        set_strategy(self.store, "direct_ssdbcodi")
        submit_message(
            self.store,
            self.provider,
            message="ignore cluster 5",
            selection_context=self.context,
        )
        self.assertGreater(len(self.store.turns), 0)

        clear_history(self.store)
        self.assertEqual(len(self.store.turns), 0)
        self.assertEqual(self.store.strategy, "direct_ssdbcodi")

    def test_suggestion_chips_hide_path_b_intents_under_metric_learning(self):
        chips = suggestion_chips_for_strategy("metric_learning")
        intent_types = {chip.intent_type for chip in chips}
        self.assertNotIn("split_cluster", intent_types)
        self.assertNotIn("reclassify_outlier", intent_types)

    def test_suggestion_chips_include_path_b_intents_under_direct_ssdbcodi(self):
        chips = suggestion_chips_for_strategy("direct_ssdbcodi")
        intent_types = {chip.intent_type for chip in chips}
        self.assertIn("split_cluster", intent_types)
        self.assertIn("reclassify_outlier", intent_types)

    def test_empty_selection_still_allows_message_without_reference(self):
        """An actionable message that does not reference selection should
        still be accepted even if no points are currently selected."""
        _, response = submit_message(
            self.store,
            self.provider,
            message="make petal_length more important",
            selection_context=self.empty_context,
        )
        self.assertEqual(response.router_category, "on_topic_actionable")

    def test_supported_strategies_constant(self):
        self.assertEqual(set(REFINEMENT_STRATEGIES), {"metric_learning", "direct_ssdbcodi"})


class MockIntentProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = MockIntentProvider()
        self.context = SelectionContext(
            dataset_id="chatbox_mock",
            selected_point_ids=("p1",),
            unselected_point_ids=("p2",),
        )
        self.empty_context = SelectionContext(
            dataset_id="chatbox_mock",
            selected_point_ids=(),
            unselected_point_ids=("p1",),
        )
        self.store = create_chatbox_store("chatbox_mock")

    def test_off_topic_message_does_not_mutate_snapshot(self):
        _, response = submit_message(
            self.store,
            self.provider,
            message="What's the weather today?",
            selection_context=self.context,
        )
        self.assertEqual(response.router_category, "off_topic")
        self.assertEqual(self.provider.current_snapshot(self.store.dataset_id).version, 0)

    def test_meta_query_does_not_mutate_snapshot(self):
        _, response = submit_message(
            self.store,
            self.provider,
            message="How many clusters are there?",
            selection_context=self.context,
        )
        self.assertEqual(response.router_category, "meta_query")
        self.assertEqual(self.provider.current_snapshot(self.store.dataset_id).version, 0)

    def test_ambiguous_message_without_selection_asks_for_clarification(self):
        _, response = submit_message(
            self.store,
            self.provider,
            message="Push these points apart",
            selection_context=self.empty_context,
        )
        self.assertEqual(response.router_category, "on_topic_ambiguous")
        self.assertTrue(response.requires_followup)

    def test_actionable_message_emits_delta_and_advances_version(self):
        _, response = submit_message(
            self.store,
            self.provider,
            message="Make petal_length more important",
            selection_context=self.context,
        )
        self.assertEqual(response.router_category, "on_topic_actionable")
        self.assertEqual(response.intent_type, "feature_weight")
        self.assertIsNotNone(response.delta)
        self.assertEqual(self.provider.current_snapshot(self.store.dataset_id).version, 1)

    def test_split_cluster_under_metric_learning_emits_deferred_note(self):
        self.store.strategy = "metric_learning"
        _, response = submit_message(
            self.store,
            self.provider,
            message="Split cluster 2 into two",
            selection_context=self.context,
        )
        self.assertEqual(response.intent_type, "split_cluster")
        self.assertIn("deferred", response.reply.lower())


if __name__ == "__main__":
    unittest.main()
