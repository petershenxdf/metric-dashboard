from __future__ import annotations

from typing import Any, Mapping

from flask import Blueprint, jsonify, render_template, request

from app.modules.chatbox.schemas import DEFAULT_HISTORY_WINDOW
from app.modules.chatbox.service import (
    clear_history,
    get_chatbox_state,
    submit_message,
    suggestion_chips,
)
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
from app.workflows.intent_runtime_support import (
    IntentRuntimeManager,
    build_memory_context,
    build_memory_state,
)
from app.modules.ssdbcodi.service import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_CONTAMINATION,
    DEFAULT_MIN_PTS,
    DEFAULT_RSCORE_WEIGHT,
)
from app.workflows.intent_response_display import (
    build_display_chat_state,
    last_display_response,
)


DEPENDENCY_MODE = "visual grounding workflow with runtime-configurable live model and persisted session artifacts"

_runtime = IntentRuntimeManager()


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
        session = _runtime.session_for(
            grounded.selection_context.dataset_id,
            _config_updates_from_request(),
        )
        state = get_chatbox_state(session.chat_store, session.provider)
        instruction = session.provider.store.get(grounded.selection_context.dataset_id)
        runtime_payload = _runtime_payload(grounded, session, state, instruction)
        return render_template(
            "workflows/intent_runtime_validation.html",
            grounded=grounded,
            state=build_display_chat_state(
                state,
                session.interactions,
                session.config.response_mode,
            ),
            instruction=instruction,
            chips_payload=[chip.to_dict() for chip in suggestion_chips()],
            provider_label=session.provider.label,
            llm_label=session.provider.llm.label,
            dependency_mode=DEPENDENCY_MODE,
            history_window=DEFAULT_HISTORY_WINDOW,
            runtime_payload=runtime_payload,
            last_response=runtime_payload.get("last_response"),
            allowed_labels=_allowed_labels(grounded.n_clusters),
            runtime_config=session.config.to_dict(),
            ssdbcodi_params=grounded.ssdbcodi_params,
            ssdbcodi_score_lookup=_ssdbcodi_score_lookup(grounded),
        )

    @blueprint.get("/api/state")
    def state_api():
        grounded = _grounded_state()
        session = _runtime.session_for(
            grounded.selection_context.dataset_id,
            _config_updates_from_request(),
        )
        state = get_chatbox_state(session.chat_store, session.provider)
        instruction = session.provider.store.get(grounded.selection_context.dataset_id)
        runtime_payload = _runtime_payload(grounded, session, state, instruction)
        return jsonify(api_success(runtime_payload, diagnostics=runtime_payload["runtime_diagnostics"]))

    @blueprint.post("/api/runtime-config")
    def runtime_config_api():
        grounded = _grounded_state()
        session = _runtime.session_for(grounded.selection_context.dataset_id)
        try:
            session = _runtime.apply_config(
                grounded.selection_context.dataset_id,
                _config_updates_from_request(),
            )
        except ValueError as exc:
            return jsonify(api_error("invalid_runtime_config", str(exc))), 400

        refreshed = _grounded_state()
        state = get_chatbox_state(session.chat_store, session.provider)
        instruction = session.provider.store.get(refreshed.selection_context.dataset_id)
        runtime_payload = _runtime_payload(refreshed, session, state, instruction)
        return jsonify(api_success(runtime_payload, diagnostics=runtime_payload["runtime_diagnostics"]))

    @blueprint.get("/api/provider-health")
    def provider_health_api():
        grounded = _grounded_state()
        session = _runtime.session_for(
            grounded.selection_context.dataset_id,
            _config_updates_from_request(),
        )
        diagnostics = _runtime.provider_diagnostics(
            session.dataset_id,
            refresh_health=True,
        )
        return jsonify(api_success(diagnostics, diagnostics=diagnostics))

    @blueprint.post("/api/messages")
    def messages_api():
        body: Mapping = request.get_json(silent=True) or {}
        message = body.get("message", "")
        grounded = _grounded_state()
        session = _runtime.session_for(
            grounded.selection_context.dataset_id,
            _config_updates_from_request(),
        )
        state = get_chatbox_state(session.chat_store, session.provider)
        instruction = session.provider.store.get(grounded.selection_context.dataset_id)
        pre_runtime_payload = {
            "state": grounded.state_payload(),
            "chat_state": state.to_dict(),
            "current_instruction": instruction.to_dict(),
        }
        memory_context = build_memory_context(pre_runtime_payload, session)

        try:
            forwarded, response = submit_message(
                session.chat_store,
                session.provider,
                message=message,
                selection_context=grounded.selection_context,
                selection_groups=grounded.selection_groups,
                label_context=grounded.label_context_payload(),
                memory_context=memory_context,
                history_window_size=DEFAULT_HISTORY_WINDOW,
            )
        except ValueError as exc:
            return jsonify(api_error("invalid_chat_message", str(exc))), 400

        _runtime.remember_response(grounded.selection_context.dataset_id, forwarded, response)
        refreshed = _grounded_state()
        state = get_chatbox_state(session.chat_store, session.provider)
        instruction = session.provider.store.get(refreshed.selection_context.dataset_id)
        runtime_payload = _runtime_payload(refreshed, session, state, instruction)
        return jsonify(
            api_success(
                {
                    "forwarded_payload": forwarded.to_dict(),
                    "response": response.to_dict(),
                    "display_response": runtime_payload.get("last_response"),
                    "runtime": runtime_payload,
                },
                diagnostics=runtime_payload["runtime_diagnostics"],
            )
        )

    @blueprint.post("/api/reset")
    def reset_api():
        grounded = _grounded_state()
        try:
            session = _runtime.reset(
                grounded.selection_context.dataset_id,
                _config_updates_from_request(),
            )
        except ValueError as exc:
            return jsonify(api_error("invalid_runtime_config", str(exc))), 400

        refreshed = _grounded_state()
        state = get_chatbox_state(session.chat_store, session.provider)
        instruction = session.provider.store.get(refreshed.selection_context.dataset_id)
        runtime_payload = _runtime_payload(refreshed, session, state, instruction)
        return jsonify(api_success(runtime_payload, diagnostics=runtime_payload["runtime_diagnostics"]))

    @blueprint.post("/api/clear")
    def clear_api():
        grounded = _grounded_state()
        session = _runtime.session_for(
            grounded.selection_context.dataset_id,
            _config_updates_from_request(),
        )
        clear_history(session.chat_store)
        _runtime.clear_chat(session.dataset_id)
        refreshed = _grounded_state()
        state = get_chatbox_state(session.chat_store, session.provider)
        instruction = session.provider.store.get(refreshed.selection_context.dataset_id)
        runtime_payload = _runtime_payload(refreshed, session, state, instruction)
        return jsonify(api_success(runtime_payload, diagnostics=runtime_payload["runtime_diagnostics"]))

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
        session = _runtime.session_for(
            grounded.selection_context.dataset_id,
            _config_updates_from_request(),
        )
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

        _runtime.remember_manual_event(
            session.dataset_id,
            "selection_group_saved",
            {"group": group.to_dict()},
        )
        refreshed = _grounded_state()
        state = get_chatbox_state(session.chat_store, session.provider)
        instruction = session.provider.store.get(refreshed.selection_context.dataset_id)
        runtime_payload = _runtime_payload(refreshed, session, state, instruction)
        return jsonify(
            api_success(
                {
                    "group": group.to_dict(),
                    "runtime": runtime_payload,
                },
                diagnostics=runtime_payload["runtime_diagnostics"],
            )
        )

    @blueprint.post("/api/groups/<group_id>/select")
    def select_group_api(group_id: str):
        grounded = _grounded_state()
        session = _runtime.session_for(
            grounded.selection_context.dataset_id,
            _config_updates_from_request(),
        )
        try:
            result = select_selection_group(grounded.selection_store, group_id)
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        _runtime.remember_manual_event(
            session.dataset_id,
            "selection_group_selected",
            {"selection": result.to_dict()},
        )
        refreshed = _grounded_state()
        state = get_chatbox_state(session.chat_store, session.provider)
        instruction = session.provider.store.get(refreshed.selection_context.dataset_id)
        runtime_payload = _runtime_payload(refreshed, session, state, instruction)
        return jsonify(
            api_success(
                {"selection": result.to_dict(), "runtime": runtime_payload},
                diagnostics=runtime_payload["runtime_diagnostics"],
            )
        )

    @blueprint.delete("/api/groups/<group_id>")
    def delete_group_api(group_id: str):
        grounded = _grounded_state()
        session = _runtime.session_for(
            grounded.selection_context.dataset_id,
            _config_updates_from_request(),
        )
        try:
            group = delete_selection_group(grounded.selection_store, group_id)
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        _runtime.remember_manual_event(
            session.dataset_id,
            "selection_group_deleted",
            {"deleted_group": group.to_dict()},
        )
        refreshed = _grounded_state()
        state = get_chatbox_state(session.chat_store, session.provider)
        instruction = session.provider.store.get(refreshed.selection_context.dataset_id)
        runtime_payload = _runtime_payload(refreshed, session, state, instruction)
        return jsonify(
            api_success(
                {"deleted_group": group.to_dict(), "runtime": runtime_payload},
                diagnostics=runtime_payload["runtime_diagnostics"],
            )
        )

    @blueprint.post("/api/label")
    def label_api():
        payload = request_payload(request)
        grounded = _grounded_state()
        session = _runtime.session_for(
            grounded.selection_context.dataset_id,
            _config_updates_from_request(),
        )
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

        _runtime.remember_manual_event(
            session.dataset_id,
            "label_applied",
            {"annotation": annotation.to_dict()},
        )
        refreshed = _grounded_state()
        state = get_chatbox_state(session.chat_store, session.provider)
        instruction = session.provider.store.get(refreshed.selection_context.dataset_id)
        runtime_payload = _runtime_payload(refreshed, session, state, instruction)
        return jsonify(
            api_success(
                {"annotation": annotation.to_dict(), "runtime": runtime_payload},
                diagnostics=runtime_payload["runtime_diagnostics"],
            )
        )

    @blueprint.post("/api/clear-labels")
    def clear_labels_api():
        grounded = _grounded_state()
        session = _runtime.session_for(
            grounded.selection_context.dataset_id,
            _config_updates_from_request(),
        )
        clear_annotations(get_labeling_store_for_context(grounded.selection_context))
        _runtime.remember_manual_event(session.dataset_id, "labels_cleared", {})
        refreshed = _grounded_state()
        state = get_chatbox_state(session.chat_store, session.provider)
        instruction = session.provider.store.get(refreshed.selection_context.dataset_id)
        runtime_payload = _runtime_payload(refreshed, session, state, instruction)
        return jsonify(api_success(runtime_payload, diagnostics=runtime_payload["runtime_diagnostics"]))

    @blueprint.post("/api/reset-labels")
    def reset_labels_api():
        grounded = _grounded_state()
        session = _runtime.session_for(
            grounded.selection_context.dataset_id,
            _config_updates_from_request(),
        )
        reset_labeling_store_for_context(grounded.selection_context)
        _runtime.remember_manual_event(session.dataset_id, "labels_reset", {})
        refreshed = _grounded_state()
        state = get_chatbox_state(session.chat_store, session.provider)
        instruction = session.provider.store.get(refreshed.selection_context.dataset_id)
        runtime_payload = _runtime_payload(refreshed, session, state, instruction)
        return jsonify(api_success(runtime_payload, diagnostics=runtime_payload["runtime_diagnostics"]))

    return blueprint


def _selection_action_response(action_name: str):
    grounded = _grounded_state()
    session = _runtime.session_for(
        grounded.selection_context.dataset_id,
        _config_updates_from_request(),
    )
    payload = request_payload(request)
    result, error = apply_selection_action_or_error(
        grounded.selection_store,
        action_name,
        payload,
        metadata={"workflow": "intent-runtime-validation"},
    )
    if error is not None:
        return jsonify(api_error("invalid_selection_action", error)), 400

    _runtime.remember_manual_event(
        session.dataset_id,
        "selection_updated",
        {"selection": result.to_dict()},
    )
    refreshed = _grounded_state()
    state = get_chatbox_state(session.chat_store, session.provider)
    instruction = session.provider.store.get(refreshed.selection_context.dataset_id)
    runtime_payload = _runtime_payload(refreshed, session, state, instruction)
    return jsonify(
        api_success(
            {
                "selection": result.to_dict(),
                "runtime": runtime_payload,
            },
            diagnostics=runtime_payload["runtime_diagnostics"],
        )
    )


def _runtime_payload(grounded, session, state, instruction):
    if session.config.response_mode == "raw":
        _runtime.ensure_freeform_replies(session.dataset_id)
    display_state = build_display_chat_state(
        state,
        session.interactions,
        session.config.response_mode,
    )
    provider_diagnostics = _runtime.provider_diagnostics(session.dataset_id)
    runtime_payload = {
        "state": grounded.state_payload(),
        "chat_state": display_state.to_dict(),
        "current_instruction": instruction.to_dict(),
        "runtime_diagnostics": {
            **provider_diagnostics,
            "dependency_mode": DEPENDENCY_MODE,
            "mode": (
                "live_model_runtime"
                if session.config.provider_kind in {"ollama", "deepseek"}
                else "deterministic_mock_runtime"
            ),
        },
        "memory_state": {},
        "evaluation_results": {
            "grounding_checks": {
                "selection_auditable": True,
                "labeling_auditable": True,
                "visible_cluster_count": len(grounded.cluster_ids),
                "visible_outlier_count": len(grounded.outlier_point_ids),
                "saved_group_count": len(grounded.selection_groups),
            },
            "status": "runtime_validation_ready",
        },
    }
    runtime_payload["memory_state"] = build_memory_state(runtime_payload, session)
    runtime_payload["last_response"] = last_display_response(
        session.interactions,
        session.config.response_mode,
    )
    runtime_payload["storage"] = _runtime.persist_runtime_snapshot(session.dataset_id, runtime_payload)
    return runtime_payload


def _allowed_cluster_labels(n_clusters: int):
    return [f"cluster_{index}" for index in range(1, n_clusters + 1)]


def _allowed_labels(n_clusters: int):
    return [*_allowed_cluster_labels(n_clusters), "outlier"]


def _ssdbcodi_score_lookup(grounded) -> Mapping[str, Mapping[str, Any]]:
    return {
        score.point_id: score.to_dict()
        for score in grounded.ssdbcodi_result.point_scores
    }


def _validate_label(action: str, label_value, n_clusters: int) -> None:
    if action == "assign_cluster" and label_value in set(_allowed_cluster_labels(n_clusters)):
        return
    if action == "mark_outlier":
        return
    allowed = ", ".join([*_allowed_cluster_labels(n_clusters), "outlier"])
    raise ValueError(f"label must be one of: {allowed}")


def _grounded_state():
    return build_grounded_chat_context(
        n_clusters=n_clusters_from_request(),
        ssdbcodi_params=_ssdbcodi_params_from_request(),
    )


def _ssdbcodi_params_from_request() -> Mapping[str, Any]:
    payload: Mapping[str, Any] = {}
    if request.is_json:
        payload = request.get_json(silent=True) or {}
    elif request.args:
        payload = request.args.to_dict()
    elif request.form:
        payload = request.form.to_dict()

    params = {
        "min_pts": _int_value(payload.get("min_pts"), DEFAULT_MIN_PTS),
        "alpha": _float_value(payload.get("alpha"), DEFAULT_ALPHA),
        "beta": _float_value(payload.get("beta"), DEFAULT_BETA),
        "contamination": _float_value(payload.get("contamination"), DEFAULT_CONTAMINATION),
        "rscore_weight": _float_value(payload.get("rscore_weight"), DEFAULT_RSCORE_WEIGHT),
    }
    return {
        "min_pts": max(1, params["min_pts"]),
        "alpha": max(0.0, min(1.0, params["alpha"])),
        "beta": max(0.0, min(1.0, params["beta"])),
        "contamination": max(0.01, min(0.49, params["contamination"])),
        "rscore_weight": max(0.0, min(1.0, params["rscore_weight"])),
    }


def _int_value(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _config_updates_from_request() -> Mapping[str, Any]:
    payload: Mapping[str, Any] = {}
    if request.is_json:
        payload = request.get_json(silent=True) or {}
    elif request.args:
        payload = request.args.to_dict()
    elif request.form:
        payload = request.form.to_dict()

    keys = {
        "provider_kind",
        "model_name",
        "base_url",
        "temperature",
        "timeout_seconds",
        "max_output_tokens",
        "allow_mock_fallback",
    }
    return {key: payload[key] for key in keys if key in payload}
