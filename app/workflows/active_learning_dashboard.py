from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from flask import (
    Blueprint,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from app.modules.active_learning import (
    ActiveLearningService,
    ActiveLearningStore,
    SessionConfig,
)
from app.modules.active_learning.service import ActiveLearningConflict
from app.modules.active_learning.translation import (
    PROMPT_VERSION as TRANSLATION_PROMPT_VERSION,
)
from app.shared.env import env_text, repo_root
from app.shared.flask_helpers import api_error, api_success
from app.shared.wine_dataset import WINE_FEATURE_NAMES, wine_raw_points


WORKFLOW_NAME = "active-learning-dashboard"


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "active_learning_dashboard_workflow",
        __name__,
        template_folder="templates",
    )

    @blueprint.get("/workflows/active-learning-dashboard/")
    def index():
        service = _service()
        return render_template(
            "workflows/active_learning_index.html",
            datasets=service.store.list_dataset_versions(),
            sessions=service.store.list_sessions(),
            error=request.args.get("error", ""),
        )

    @blueprint.post("/workflows/active-learning-dashboard/import")
    def import_and_create():
        try:
            prepared = _import_dataset_from_request(_service())
            session = _service().create_session(
                prepared.version.dataset_version_id,
                _session_config_from_request(),
            )
            return redirect(
                url_for(
                    "active_learning_dashboard_workflow.dashboard",
                    session_id=session.session_id,
                )
            )
        except (ValueError, RuntimeError) as exc:
            return redirect(
                url_for(
                    "active_learning_dashboard_workflow.index",
                    error=str(exc),
                )
            )

    @blueprint.post("/workflows/active-learning-dashboard/wine-fixture")
    def create_wine_fixture():
        try:
            service = _service()
            prepared = _ensure_wine_fixture(service)
            session = service.create_session(
                prepared.version.dataset_version_id,
                _session_config_from_request(),
            )
            return redirect(
                url_for(
                    "active_learning_dashboard_workflow.dashboard",
                    session_id=session.session_id,
                )
            )
        except (ValueError, RuntimeError) as exc:
            return redirect(
                url_for(
                    "active_learning_dashboard_workflow.index",
                    error=str(exc),
                )
            )

    @blueprint.get("/workflows/active-learning-dashboard/<session_id>/")
    def dashboard(session_id: str):
        focus_category = request.args.get("focus_category", "label_priority")
        provider_kind = request.args.get("provider_kind", "mock").strip().lower()
        if provider_kind not in {"mock", "deepseek"}:
            provider_kind = "mock"
        try:
            service = _service()
            state = service.session_state(
                session_id,
                focus_category=focus_category,
            )
            service.store.record_recommendation_shown(
                session_id=session_id,
                round_id=state["round"]["round_id"],
                plan_id=state["recommendation_plan"]["plan_id"],
                point_ids=state["recommendation_plan"].get(
                    "recommended_point_ids",
                    (),
                ),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            interpretation = None
            if request.args.get("show_interpretation") == "1":
                cached_interpretation = service.store.get_interpretation(
                    state["round"]["round_id"],
                    state["recommendation_plan"]["plan_id"],
                    provider_kind,
                )
                if (
                    cached_interpretation is not None
                    and cached_interpretation.get("diagnostics", {}).get(
                        "prompt_template_version"
                    )
                    == TRANSLATION_PROMPT_VERSION
                ):
                    interpretation = cached_interpretation
            guidance = (
                interpretation["guidance"]
                if interpretation is not None
                else state["guidance"]
            )
            return render_template(
                "workflows/active_learning_dashboard.html",
                state=state,
                guidance=guidance,
                interpretation=interpretation,
                provider_kind=provider_kind,
            )
        except ValueError as exc:
            return render_template(
                "workflows/active_learning_index.html",
                datasets=_service().store.list_dataset_versions(),
                sessions=_service().store.list_sessions(),
                error=str(exc),
            ), 404

    @blueprint.get("/api/datasets")
    def datasets_api():
        versions = [
            item.to_dict() for item in _service().store.list_dataset_versions()
        ]
        return jsonify(api_success({"dataset_versions": versions}))

    @blueprint.post("/api/datasets")
    def create_dataset_api():
        try:
            prepared = _import_dataset_from_request(_service())
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            return jsonify(api_error("invalid_dataset", str(exc))), 400
        return jsonify(api_success(prepared.version.to_dict())), 201

    @blueprint.post("/api/active-learning/sessions")
    def create_session_api():
        payload = request.get_json(silent=True) or {}
        try:
            session = _service().create_session(
                str(payload.get("dataset_version_id", "")),
                SessionConfig.from_dict(payload.get("config")),
            )
            state = _service().session_state(session.session_id)
        except ValueError as exc:
            return jsonify(api_error("invalid_session", str(exc))), 400
        return jsonify(
            api_success(
                {
                    "session": session.to_dict(),
                    "state": state,
                }
            )
        ), 201

    @blueprint.get("/api/active-learning/sessions/<session_id>/state")
    def session_state_api(session_id: str):
        try:
            state = _service().session_state(
                session_id,
                focus_category=request.args.get(
                    "focus_category", "label_priority"
                ),
            )
        except ValueError as exc:
            return jsonify(api_error("unknown_session", str(exc))), 404
        return jsonify(api_success(state))

    @blueprint.post(
        "/api/active-learning/sessions/<session_id>/rounds/<round_id>/labels"
    )
    def labels_api(session_id: str, round_id: str):
        payload = request.get_json(silent=True) or {}
        try:
            if not payload.get("expected_round_id"):
                raise ValueError("expected_round_id is required")
            result = _service().commit_labels(
                session_id,
                round_id=round_id,
                expected_round_id=payload.get("expected_round_id"),
                expected_label_revision=int(
                    payload.get("expected_label_revision", -1)
                ),
                plan_id=str(payload.get("plan_id", "")),
                category=str(payload.get("category", "label_priority")),
                labels=payload.get("labels", ()),
            )
        except ActiveLearningConflict as exc:
            return jsonify(api_error("stale_round", str(exc))), 409
        except (TypeError, ValueError) as exc:
            return jsonify(api_error("invalid_labels", str(exc))), 400
        return jsonify(api_success(result))

    @blueprint.post(
        "/api/active-learning/sessions/<session_id>/rounds/<round_id>/revert"
    )
    def revert_api(session_id: str, round_id: str):
        try:
            result = _service().revert_to_round(session_id, round_id)
        except ValueError as exc:
            return jsonify(api_error("invalid_round", str(exc))), 400
        return jsonify(api_success(result))

    @blueprint.post(
        "/api/active-learning/sessions/<session_id>/rounds/<round_id>"
        "/categories/<category>/interpret"
    )
    def interpret_api(session_id: str, round_id: str, category: str):
        payload = request.get_json(silent=True) or {}
        try:
            result = _service().interpret_category(
                session_id,
                round_id=round_id,
                category=category,
                provider_kind=str(payload.get("provider_kind", "deepseek")),
            )
        except ValueError as exc:
            return jsonify(api_error("invalid_interpretation", str(exc))), 400
        return jsonify(api_success(result))

    @blueprint.get("/api/active-learning/sessions/<session_id>/history")
    def history_api(session_id: str):
        try:
            session = _service().store.get_session(session_id)
            rounds = _service().store.list_rounds(session_id)
            labels = _service().store.all_label_events(session_id)
        except ValueError as exc:
            return jsonify(api_error("unknown_session", str(exc))), 404
        return jsonify(
            api_success(
                {
                    "session": session.to_dict(),
                    "rounds": [item.to_dict() for item in rounds],
                    "label_events": [item.to_dict() for item in labels],
                }
            )
        )

    return blueprint


def _service() -> ActiveLearningService:
    configured = current_app.config.get("ACTIVE_LEARNING_DB_PATH")
    default_path = (
        repo_root()
        / "runtime_data"
        / "active_learning"
        / "active_learning.sqlite3"
    )
    db_path = Path(
        configured
        or env_text(
            "METRIC_DASHBOARD_ACTIVE_LEARNING_DB_PATH",
            str(default_path),
        )
    )
    if not db_path.is_absolute():
        db_path = repo_root() / db_path
    return ActiveLearningService(ActiveLearningStore(db_path))


def _import_dataset_from_request(service: ActiveLearningService):
    options = _dataset_options()
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        records = payload.get("records")
        if records is None:
            raise ValueError("JSON request must include records")
        return service.import_records(
            records,
            dataset_id=payload.get("dataset_id"),
            entity_name=payload.get("entity_name", "record"),
            source_format=payload.get("source_format", "json"),
            point_id_column=payload.get("point_id_column"),
            feature_columns=payload.get("feature_columns"),
            metadata_columns=payload.get("metadata_columns", ()),
            ground_truth_columns=payload.get("ground_truth_columns", ()),
            preprocessing_config=payload.get("preprocessing_config"),
        )
    upload = request.files.get("dataset_file")
    if upload is None or not upload.filename:
        raise ValueError("choose a CSV, JSON, or MAT dataset file")
    source_format = (
        request.form.get("source_format")
        or Path(upload.filename).suffix.lstrip(".")
    ).lower()
    if source_format == "mat":
        return service.import_file(
            upload.read(),
            "mat",
            dataset_id=options["dataset_id"],
            entity_name=options["entity_name"],
            matrix_key=request.form.get("matrix_key", "X"),
            label_key=request.form.get("label_key", "y"),
            feature_names=_split_list(request.form.get("feature_columns")),
            preprocessing_config=options["preprocessing_config"],
        )
    return service.import_file(
        upload.read(),
        source_format,
        dataset_id=options["dataset_id"],
        entity_name=options["entity_name"],
        point_id_column=options["point_id_column"],
        feature_columns=options["feature_columns"] or None,
        metadata_columns=options["metadata_columns"],
        ground_truth_columns=options["ground_truth_columns"],
        preprocessing_config=options["preprocessing_config"],
    )


def _dataset_options() -> Mapping[str, Any]:
    return {
        "dataset_id": request.form.get("dataset_id") or None,
        "entity_name": request.form.get("entity_name") or "record",
        "point_id_column": request.form.get("point_id_column") or None,
        "feature_columns": _split_list(request.form.get("feature_columns")),
        "metadata_columns": _split_list(request.form.get("metadata_columns")),
        "ground_truth_columns": _split_list(
            request.form.get("ground_truth_columns")
        ),
        "preprocessing_config": _json_mapping(
            request.form.get("preprocessing_config")
        ),
    }


def _session_config_from_request() -> SessionConfig:
    values = request.form if request.form else (request.get_json(silent=True) or {})
    return SessionConfig.from_dict(
        {
            "n_clusters": values.get("n_clusters", 3),
            "max_depth": values.get("max_depth", 3),
            "min_samples_leaf": values.get("min_samples_leaf", 1),
            "batch_size": values.get("batch_size", 4),
            "label_budget": values.get("label_budget") or None,
        }
    )


def _split_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return tuple(
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    )


def _json_mapping(value: Any) -> Mapping[str, Any] | None:
    if value in (None, ""):
        return None
    if isinstance(value, Mapping):
        return dict(value)
    payload = json.loads(str(value))
    if not isinstance(payload, Mapping):
        raise ValueError("preprocessing_config must be a JSON object")
    return dict(payload)


def _ensure_wine_fixture(service: ActiveLearningService):
    records = []
    for point in wine_raw_points():
        record = {
            name: point["features"][index]
            for index, name in enumerate(WINE_FEATURE_NAMES)
        }
        record["point_id"] = point["point_id"]
        record["source_file"] = point["metadata"]["source_file"]
        record["ground_truth"] = point["metadata"]["class_label"]
        records.append(record)
    return service.import_records(
        records,
        dataset_id="wine_mat",
        entity_name="wine",
        source_format="mat_fixture",
        point_id_column="point_id",
        feature_columns=WINE_FEATURE_NAMES,
        metadata_columns=("source_file",),
        ground_truth_columns=("ground_truth",),
    )
