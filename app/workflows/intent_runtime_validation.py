from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

from flask import Blueprint, jsonify, render_template, request

from app.modules.chatbox.schemas import DEFAULT_HISTORY_WINDOW
from app.modules.chatbox.service import (
    clear_history,
    create_chatbox_store,
    get_chatbox_state,
    submit_message,
    suggestion_chips,
)
from app.modules.intent_instruction.service import IntentInstructionProvider
from app.modules.labeling.service import apply_labeling_action, clear_annotations
from app.modules.labeling.state import (
    get_debug_store_for_context as get_labeling_store_for_context,
    reset_debug_store_for_context as reset_labeling_store_for_context,
)
from app.modules.selection.http_helpers import optional_point_ids_from_payload
from app.modules.selection.service import (
    delete_selection_group,
    save_selection_group,
    select_selection_group,
)
from app.shared.chat_grounding import build_grounded_chat_context
from app.shared.flask_helpers import api_error, api_success
from app.shared.request_helpers import (
    apply_selection_action_or_error,
    n_clusters_from_request,
    request_payload,
    selection_groups_payload,
)


DEPENDENCY_MODE = "shared grounded context plus embedded scatterplot, selection, labeling, and chat runtime"


@dataclass
class _WorkflowRuntime:
    provider: IntentInstructionProvider = field(default_factory=IntentInstructionProvider)
    stores: Dict[str, object] = field(default_factory=dict)
    last_responses: Dict[str, Mapping[str, object] | None] = field(default_factory=dict)

    def store_for(self, dataset_id: str):
        if dataset_id not in self.stores:
            self.stores[dataset_id] = create_chatbox_store(dataset_id)
        return self.stores[dataset_id]

    def remember_response(self, dataset_id: str, response) -> None:
        self.last_responses[dataset_id] = response.to_dict()

    def last_response_for(self, dataset_id: str):
        return self.last_responses.get(dataset_id)

    def reset(self, dataset_id: str) -> None:
        self.stores[dataset_id] = create_chatbox_store(dataset_id)
        self.last_responses[dataset_id] = None
        self.provider.reset(dataset_id)


_runtime = _WorkflowRuntime()


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "intent_runtime_validation_workflow",
        __name__,
        template_folder="templates",
        url_prefix="/workflows/intent-runtime-validation",
    )

    @blueprint.get("/")
    def index():
        grounded = _grounded_state()
        store = _runtime.store_for(grounded.selection_context.dataset_id)
        provider = _runtime.provider
        state = get_chatbox_state(store, provider)
        instruction = provider.store.get(grounded.selection_context.dataset_id)
        return render_template(
            "workflows/intent_runtime_validation.html",
            grounded=grounded,
            state=state,
            instruction=instruction,
            chips_payload=[chip.to_dict() for chip in suggestion_chips()],
            provider_label=provider.label,
            llm_label=provider.llm.label,
            dependency_mode=DEPENDENCY_MODE,
            history_window=DEFAULT_HISTORY_WINDOW,
            runtime_payload=_runtime_payload(grounded, state, instruction),
            last_response=_runtime.last_response_for(grounded.selection_context.dataset_id),
            allowed_labels=_allowed_labels(grounded.n_clusters),
        )

    @blueprint.get("/api/state")
    def state_api():
        grounded = _grounded_state()
        store = _runtime.store_for(grounded.selection_context.dataset_id)
        provider = _runtime.provider
        state = get_chatbox_state(store, provider)
        instruction = provider.store.get(grounded.selection_context.dataset_id)
        return jsonify(
            api_success(
                _runtime_payload(grounded, state, instruction),
                diagnostics={
                    "dependency_mode": DEPENDENCY_MODE,
                    "provider": provider.label,
                    "llm": provider.llm.label,
                },
            )
        )

    @blueprint.post("/api/messages")
    def messages_api():
        body: Mapping = request.get_json(silent=True) or {}
        message = body.get("message", "")
        grounded = _grounded_state()
        store = _runtime.store_for(grounded.selection_context.dataset_id)
        provider = _runtime.provider

        try:
            forwarded, response = submit_message(
                store,
                provider,
                message=message,
                selection_context=grounded.selection_context,
                selection_groups=grounded.selection_groups,
                label_context=grounded.label_context_payload(),
                history_window_size=DEFAULT_HISTORY_WINDOW,
            )
        except ValueError as exc:
            return jsonify(api_error("invalid_chat_message", str(exc))), 400

        _runtime.remember_response(grounded.selection_context.dataset_id, response)
        refreshed = _grounded_state()
        state = get_chatbox_state(store, provider)
        instruction = provider.store.get(refreshed.selection_context.dataset_id)
        return jsonify(
            api_success(
                {
                    "forwarded_payload": forwarded.to_dict(),
                    "response": response.to_dict(),
                    "runtime": _runtime_payload(refreshed, state, instruction),
                },
                diagnostics={
                    "dependency_mode": DEPENDENCY_MODE,
                    "provider": provider.label,
                    "llm": provider.llm.label,
                },
            )
        )

    @blueprint.post("/api/reset")
    def reset_api():
        grounded = _grounded_state()
        _runtime.reset(grounded.selection_context.dataset_id)
        store = _runtime.store_for(grounded.selection_context.dataset_id)
        provider = _runtime.provider
        state = get_chatbox_state(store, provider)
        instruction = provider.store.get(grounded.selection_context.dataset_id)
        return jsonify(
            api_success(
                _runtime_payload(grounded, state, instruction),
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/clear")
    def clear_api():
        grounded = _grounded_state()
        store = _runtime.store_for(grounded.selection_context.dataset_id)
        clear_history(store)
        provider = _runtime.provider
        state = get_chatbox_state(store, provider)
        instruction = provider.store.get(grounded.selection_context.dataset_id)
        return jsonify(
            api_success(
                _runtime_payload(grounded, state, instruction),
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/toggle")
    def toggle_api():
        return _selection_action_response("toggle")

    @blueprint.post("/api/select")
    def select_api():
        return _selection_action_response("select")

    @blueprint.post("/api/clear-selection")
    def clear_selection_api():
        return _selection_action_response("clear")

    @blueprint.post("/api/groups")
    def save_group_api():
        grounded = _grounded_state()
        payload = request_payload(request)
        try:
            group = save_selection_group(
                grounded.selection_store,
                group_name=payload.get("group_name", ""),
                point_ids=optional_point_ids_from_payload(payload),
                metadata={"workflow": "intent-runtime-validation"},
            )
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        return jsonify(
            api_success(
                {
                    "group": group.to_dict(),
                    "groups": selection_groups_payload(grounded.selection_store),
                },
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/groups/<group_id>/select")
    def select_group_api(group_id: str):
        grounded = _grounded_state()
        try:
            result = select_selection_group(grounded.selection_store, group_id)
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        refreshed = _grounded_state()
        return jsonify(
            api_success(
                {
                    "selection": result.to_dict(),
                    "runtime": _runtime_payload(
                        refreshed,
                        get_chatbox_state(_runtime.store_for(refreshed.selection_context.dataset_id), _runtime.provider),
                        _runtime.provider.store.get(refreshed.selection_context.dataset_id),
                    ),
                },
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.delete("/api/groups/<group_id>")
    def delete_group_api(group_id: str):
        grounded = _grounded_state()
        try:
            group = delete_selection_group(grounded.selection_store, group_id)
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        return jsonify(
            api_success(
                {
                    "deleted_group": group.to_dict(),
                    "groups": selection_groups_payload(grounded.selection_store),
                },
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/label")
    def label_api():
        payload = request_payload(request)
        grounded = _grounded_state()
        store = get_labeling_store_for_context(grounded.selection_context)
        try:
            action = str(payload.get("action", ""))
            label_value = payload.get("label_value")
            _validate_label(action, label_value, grounded.n_clusters)
            annotation = apply_labeling_action(
                store,
                grounded.selection_context,
                action=action,
                label_value=label_value,
            )
        except ValueError as exc:
            return jsonify(api_error("invalid_labeling_action", str(exc))), 400

        refreshed = _grounded_state()
        return jsonify(
            api_success(
                {
                    "annotation": annotation.to_dict(),
                    "runtime": _runtime_payload(
                        refreshed,
                        get_chatbox_state(_runtime.store_for(refreshed.selection_context.dataset_id), _runtime.provider),
                        _runtime.provider.store.get(refreshed.selection_context.dataset_id),
                    ),
                },
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/clear-labels")
    def clear_labels_api():
        grounded = _grounded_state()
        clear_annotations(get_labeling_store_for_context(grounded.selection_context))
        refreshed = _grounded_state()
        return jsonify(
            api_success(
                _runtime_payload(
                    refreshed,
                    get_chatbox_state(_runtime.store_for(refreshed.selection_context.dataset_id), _runtime.provider),
                    _runtime.provider.store.get(refreshed.selection_context.dataset_id),
                ),
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/reset-labels")
    def reset_labels_api():
        grounded = _grounded_state()
        reset_labeling_store_for_context(grounded.selection_context)
        refreshed = _grounded_state()
        return jsonify(
            api_success(
                _runtime_payload(
                    refreshed,
                    get_chatbox_state(_runtime.store_for(refreshed.selection_context.dataset_id), _runtime.provider),
                    _runtime.provider.store.get(refreshed.selection_context.dataset_id),
                ),
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    return blueprint


def _selection_action_response(action_name: str):
    grounded = _grounded_state()
    payload = request_payload(request)
    result, error = apply_selection_action_or_error(
        grounded.selection_store,
        action_name,
        payload,
        metadata={"workflow": "intent-runtime-validation"},
    )
    if error is not None:
        return jsonify(api_error("invalid_selection_action", error)), 400

    refreshed = _grounded_state()
    state = get_chatbox_state(_runtime.store_for(refreshed.selection_context.dataset_id), _runtime.provider)
    instruction = _runtime.provider.store.get(refreshed.selection_context.dataset_id)
    return jsonify(
        api_success(
            {
                "selection": result.to_dict(),
                "runtime": _runtime_payload(refreshed, state, instruction),
            },
            diagnostics={"dependency_mode": DEPENDENCY_MODE},
        )
    )


def _runtime_payload(grounded, state, instruction):
    dataset_id = grounded.selection_context.dataset_id
    return {
        "state": grounded.state_payload(),
        "chat_state": state.to_dict(),
        "current_instruction": instruction.to_dict(),
        "runtime_diagnostics": {
            "dependency_mode": DEPENDENCY_MODE,
            "provider": _runtime.provider.label,
            "llm": _runtime.provider.llm.label,
            "mode": "deterministic_visual_grounding_lab",
        },
        "memory_state": {
            "turn_count": len(state.turns),
            "history_window": DEFAULT_HISTORY_WINDOW,
            "recent_turns": [turn.to_dict() for turn in state.turns[-DEFAULT_HISTORY_WINDOW:]],
            "last_response": _runtime.last_response_for(dataset_id),
            "draft_state": {
                "pending_clarification": (
                    _runtime.last_response_for(dataset_id).get("followup_question")
                    if _runtime.last_response_for(dataset_id)
                    and _runtime.last_response_for(dataset_id).get("requires_followup")
                    else None
                ),
            },
        },
        "evaluation_results": {
            "grounding_checks": {
                "selection_auditable": True,
                "labeling_auditable": True,
                "visible_cluster_count": len(grounded.cluster_ids),
                "visible_outlier_count": len(grounded.outlier_point_ids),
                "saved_group_count": len(grounded.selection_groups),
            },
            "status": "manual_runtime_validation_ready",
        },
    }


def _allowed_cluster_labels(n_clusters: int):
    return [f"cluster_{index}" for index in range(1, n_clusters + 1)]


def _allowed_labels(n_clusters: int):
    return [*_allowed_cluster_labels(n_clusters), "outlier"]


def _validate_label(action: str, label_value, n_clusters: int) -> None:
    if action == "assign_cluster" and label_value in set(_allowed_cluster_labels(n_clusters)):
        return
    if action == "mark_outlier":
        return
    allowed = ", ".join([*_allowed_cluster_labels(n_clusters), "outlier"])
    raise ValueError(f"label must be one of: {allowed}")


def _grounded_state():
    return build_grounded_chat_context(n_clusters=n_clusters_from_request())
