from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from app.modules.chatbox.schemas import ChatMessagePayload
from app.shared.flask_helpers import api_error, api_success

from .fixtures import demo_dataset_context, example_messages, router_category_order
from .router import classify
from .schemas import ALL_INTENT_TYPES
from .state import get_debug_provider


DEFAULT_DATASET_ID = "intent_debug"


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "intent_instruction",
        __name__,
        template_folder="templates",
        url_prefix="/modules/intent-instruction",
    )

    @blueprint.get("/")
    def index():
        provider = get_debug_provider()
        context = demo_dataset_context()
        state = provider.store.get(context.dataset_id)
        return render_template(
            "intent_instruction/index.html",
            provider_label=provider.label,
            llm_label=provider.llm.label,
            dataset_context=context,
            state=state,
            examples=example_messages(),
            intent_types=ALL_INTENT_TYPES,
            router_categories=router_category_order(),
            dependency_mode="deterministic mock compiler backend, real selection/label context via fixtures",
        )

    @blueprint.get("/health")
    def health():
        provider = get_debug_provider()
        return jsonify(
            api_success(
                {"module": "intent-instruction", "status": "working"},
                diagnostics={
                    "llm_provider": provider.llm.label,
                    "provider_label": provider.label,
                },
            )
        )

    @blueprint.post("/api/route")
    def route_api():
        body = request.get_json(silent=True) or {}
        message = body.get("message", "")
        if not isinstance(message, str) or not message.strip():
            return jsonify(api_error("invalid_message", "message must be a non-empty string")), 400

        provider = get_debug_provider()
        context = demo_dataset_context()
        result = classify(provider.llm, message, context, ())
        return jsonify(
            api_success(
                {
                    "router_result": result.to_dict(),
                    "dataset_context": context.to_dict(),
                },
                diagnostics={"llm_provider": provider.llm.label},
            )
        )

    @blueprint.post("/api/compile")
    def compile_api():
        body = request.get_json(silent=True) or {}
        message = body.get("message", "")
        if not isinstance(message, str) or not message.strip():
            return jsonify(api_error("invalid_message", "message must be a non-empty string")), 400

        provider = get_debug_provider()
        context = demo_dataset_context()
        try:
            payload = ChatMessagePayload(
                message=message,
                dataset_id=context.dataset_id,
                selected_point_ids=context.selected_point_ids,
                unselected_point_ids=context.unselected_point_ids,
                selection_groups=tuple({"group_name": name} for name in context.selection_group_names),
                label_context={
                    "analysis_context": {
                        "feature_names": list(context.feature_names),
                        "cluster_ids": list(context.cluster_ids),
                        "outlier_point_ids": list(context.outlier_point_ids),
                    }
                },
                history_window=(),
            )
        except ValueError as exc:
            return jsonify(api_error("invalid_payload", str(exc))), 400

        response = provider.respond(payload)
        state = provider.store.get(context.dataset_id)
        return jsonify(
            api_success(
                {
                    "response": response.to_dict(),
                    "current_instruction": state.to_dict(),
                    "dataset_context": context.to_dict(),
                },
                diagnostics={"llm_provider": provider.llm.label},
            )
        )

    @blueprint.get("/api/state")
    def state_api():
        provider = get_debug_provider()
        context = demo_dataset_context()
        state = provider.store.get(context.dataset_id)
        return jsonify(
            api_success(
                {
                    "module": "intent-instruction",
                    "current_instruction": state.to_dict(),
                    "dataset_context": context.to_dict(),
                    "snapshot": state.snapshot().to_dict(),
                },
                diagnostics={"llm_provider": provider.llm.label},
            )
        )

    @blueprint.post("/api/reset")
    def reset_api():
        provider = get_debug_provider()
        context = demo_dataset_context()
        provider.reset(context.dataset_id)
        state = provider.store.get(context.dataset_id)
        return jsonify(
            api_success(
                {
                    "current_instruction": state.to_dict(),
                },
                diagnostics={"llm_provider": provider.llm.label},
            )
        )

    @blueprint.get("/api/examples")
    def examples_api():
        return jsonify(api_success({"examples": example_messages()}))

    return blueprint
