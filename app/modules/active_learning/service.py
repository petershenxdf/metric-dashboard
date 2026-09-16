from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Sequence
from uuid import uuid4

from app.modules.algorithm_adapters.service import run_default_analysis
from app.modules.labeling.schemas import LabelingState, ManualAnnotation
from app.modules.projection.service import (
    project_feature_matrix,
    scaled_projection_points,
)
from app.modules.rule_panel.schemas import (
    RuleCard,
    RuleCondition,
    RuleSet,
    TreeConfig,
)
from app.modules.rule_panel.service import generate_rule_set
from app.shared.schemas import (
    AnalysisResult,
    ClusterAssignment,
    ClusterResult,
    OutlierResult,
    OutlierScore,
    ProjectionCoordinate,
    ProjectionResult,
)

from .data import (
    PreparedDataset,
    display_condition,
    import_dataset_bytes,
    prepare_records,
)
from .schemas import (
    ActiveLearningRound,
    ActiveLearningSession,
    LabelEvent,
    SessionConfig,
)
from .store import ActiveLearningStore
from .review import CATEGORY_IDS, build_review, member_sets
from app.modules.ssdbcodi.service import SsdbcodiProvider


EXACT_SSDBCODI_MAX_POINTS = 2000
PLOT_WIDTH = 860
PLOT_HEIGHT = 520


class ActiveLearningService:
    def __init__(self, store: ActiveLearningStore) -> None:
        self.store = store

    def import_records(
        self,
        records: Sequence[Mapping[str, Any]],
        **options,
    ) -> PreparedDataset:
        return self.store.save_prepared_dataset(prepare_records(records, **options))

    def import_file(
        self,
        content: bytes,
        source_format: str,
        **options,
    ) -> PreparedDataset:
        prepared = import_dataset_bytes(content, source_format, **options)
        return self.store.save_prepared_dataset(prepared)

    def create_session(
        self,
        dataset_version_id: str,
        config: SessionConfig | Mapping[str, Any] | None = None,
    ) -> ActiveLearningSession:
        prepared = self.store.load_prepared_dataset(dataset_version_id)
        session_config = (
            config
            if isinstance(config, SessionConfig)
            else SessionConfig.from_dict(config)
        )
        if prepared.version.point_count > EXACT_SSDBCODI_MAX_POINTS:
            raise ValueError(
                f"dataset has {prepared.version.point_count} points; "
                f"the current exact SSDBCODI provider supports at most "
                f"{EXACT_SSDBCODI_MAX_POINTS}"
            )
        if prepared.version.point_count > session_config.max_points:
            raise ValueError(
                f"dataset has {prepared.version.point_count} points; "
                f"this session is configured for at most {session_config.max_points}"
            )
        if session_config.n_clusters > prepared.version.point_count:
            raise ValueError("n_clusters must not exceed the dataset point count")
        if session_config.min_pts >= prepared.version.point_count:
            raise ValueError("MinPts must be less than the dataset point count")
        now = _now_iso()
        session = ActiveLearningSession(
            session_id=f"als_{uuid4().hex[:12]}",
            dataset_version_id=dataset_version_id,
            status="active",
            config=session_config,
            label_vocabulary={},
            current_round_id=None,
            created_at=now,
            updated_at=now,
        )
        self.store.save_session(session)
        round_state = self._compute_round(session, parent_round=None, label_revision=0)
        return self.store.update_session_current(
            session,
            current_round_id=round_state.round_id,
            status="stopped" if round_state.status == "stopped" else "active",
            updated_at=_now_iso(),
        )

    def commit_labels(
        self,
        session_id: str,
        *,
        round_id: str,
        expected_round_id: str | None = None,
        expected_label_revision: int,
        category: str = "model_instability",
        labels: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        session = self.store.get_session(session_id)
        if session.status != "active":
            raise ActiveLearningConflict(
                f"session is {session.status}; refresh before submitting labels"
            )
        if expected_round_id is not None and expected_round_id != round_id:
            raise ActiveLearningConflict(
                "expected_round_id does not match the request path"
            )
        if session.current_round_id != round_id:
            raise ActiveLearningConflict(
                "this page is based on an older active-learning round"
            )
        current_round = self.store.get_round(round_id)
        if current_round.label_revision != expected_label_revision:
            raise ActiveLearningConflict(
                "label revision changed; refresh before submitting labels"
            )
        if category not in CATEGORY_IDS:
            raise ValueError("unknown review category")
        if not isinstance(labels, (list, tuple)):
            raise ValueError("labels must be a list")
        if not labels:
            raise ValueError("labels must contain at least one label")

        prepared = self.store.load_prepared_dataset(session.dataset_version_id)
        known_points = set(prepared.version.point_ids)
        vocabulary = dict(session.label_vocabulary)
        now = _now_iso()
        events = []
        submission_keys = set()
        for index, payload in enumerate(labels):
            if not isinstance(payload, Mapping):
                raise ValueError("each label must be an object")
            point_id = str(payload.get("point_id", "")).strip()
            if point_id not in known_points:
                raise ValueError(f"unknown point id: {point_id}")
            dimension = str(payload.get("label_dimension", "")).strip()
            if dimension not in {"semantic_class", "outlier_status", "uncertain"}:
                raise ValueError(
                    "label_dimension must be semantic_class, outlier_status, or uncertain"
                )
            submission_key = (point_id, dimension)
            if submission_key in submission_keys:
                raise ValueError(
                    "a label batch must not repeat the same point and label dimension"
                )
            submission_keys.add(submission_key)
            value = payload.get("label_value")
            if dimension == "semantic_class":
                label_id, vocabulary = _semantic_label(value, vocabulary)
                value = label_id
            elif dimension == "outlier_status":
                value = _coerce_bool(value, "outlier_status label_value")
            else:
                value = True
            events.append(
                LabelEvent(
                    event_id=f"label_{uuid4().hex[:14]}",
                    session_id=session_id,
                    round_id=round_id,
                    point_id=point_id,
                    label_dimension=dimension,
                    label_value=value,
                    status="active",
                    supersedes_event_id=None,
                    provenance={
                        "review_version": current_round.review.get("version"),
                        "category": category,
                        "submission_index": index,
                    },
                    created_at=now,
                )
            )

        existing_active_events = self.store.active_label_events(session_id)
        projected_effective_labels = {
            (event.point_id, event.label_dimension)
            for event in existing_active_events
            if event.label_dimension != "uncertain"
        }
        projected_effective_labels.update(
            (event.point_id, event.label_dimension)
            for event in events
            if event.label_dimension != "uncertain"
        )
        if (
            session.config.label_budget is not None
            and len(projected_effective_labels) > session.config.label_budget
        ):
            raise ValueError("label submission exceeds the session label budget")

        replaced_keys = {(event.point_id, event.label_dimension) for event in events}
        next_active_events = tuple(
            event
            for event in existing_active_events
            if (event.point_id, event.label_dimension) not in replaced_keys
        ) + tuple(events)
        refreshed_session = replace(
            session,
            label_vocabulary=vocabulary,
        )
        try:
            next_round = self._compute_round(
                refreshed_session,
                parent_round=current_round,
                label_revision=current_round.label_revision + 1,
                effective_events=next_active_events,
                persist=False,
            )
            committed, updated_session = self.store.commit_round_transition(
                session,
                current_round,
                events,
                next_round,
                updated_at=_now_iso(),
                label_vocabulary=vocabulary,
            )
        except ValueError as exc:
            if "session changed" in str(exc) or "not ready" in str(exc):
                raise ActiveLearningConflict(str(exc)) from exc
            raise
        return {
            "events": [event.to_dict() for event in committed],
            "session": updated_session.to_dict(),
            "round": next_round.to_dict(),
        }

    def revert_to_round(self, session_id: str, round_id: str) -> Mapping[str, Any]:
        session = self.store.get_session(session_id)
        target = self.store.get_round(round_id)
        if target.session_id != session_id:
            raise ValueError("round does not belong to this session")
        try:
            self.store.revert_events_to_round(
                session_id,
                target.round_id,
                expected_round_id=session.current_round_id,
                updated_at=_now_iso(),
            )
        except ValueError as exc:
            if "session changed" in str(exc):
                raise ActiveLearningConflict(str(exc)) from exc
            raise
        updated = self.store.get_session(session_id)
        return {"session": updated.to_dict(), "round": target.to_dict()}

    def session_state(self, session_id: str, *, focus_category="model_instability"):
        session = self.store.get_session(session_id)
        if session.current_round_id is None:
            raise ValueError("active-learning session has no current round")
        round_state = self.store.get_round(session.current_round_id)
        prepared = self.store.load_prepared_dataset(session.dataset_version_id)
        assignments = {
            item["point_id"]: item["cluster_id"]
            for item in round_state.analysis["cluster_result"]["assignments"]
        }
        outliers = set(round_state.analysis["outlier_result"]["outlier_point_ids"])
        raw_by_id = {item["point_id"]: item for item in prepared.raw_records}
        plot_points = [
            {
                **point,
                "cluster_id": assignments.get(point["point_id"], "unassigned"),
                "is_outlier": point["point_id"] in outliers,
                "raw_features": raw_by_id[point["point_id"]].get("raw_features", {}),
                "metadata": raw_by_id[point["point_id"]].get("metadata", {}),
            }
            for point in scaled_projection_points(
                _projection_from_dict(round_state.projection), assignments
            )
        ]
        return {
            "workflow": "active-learning-dashboard",
            "session": session.to_dict(),
            "round": round_state.to_dict(),
            "dataset_version": prepared.version.to_dict(),
            "review": round_state.review,
            "legacy_round": not bool(round_state.review),
            "focus_category": focus_category
            if focus_category in CATEGORY_IDS
            else "model_instability",
            "analysis": round_state.analysis,
            "rule_set": round_state.display_rule_set,
            "plot_points": plot_points,
            "plot_width": PLOT_WIDTH,
            "plot_height": PLOT_HEIGHT,
            "active_labels": [
                event.to_dict() for event in self.store.active_label_events(session_id)
            ],
            "history": [
                {
                    "round_id": item.round_id,
                    "round_index": item.round_index,
                    "label_revision": item.label_revision,
                    "status": item.status,
                    "created_at": item.created_at,
                    "delta": item.delta,
                    "is_current": item.round_id == session.current_round_id,
                }
                for item in self.store.list_rounds(session_id)
            ],
        }

    def upgrade_round(self, session_id, expected_round_id):
        """Explicitly recompute a legacy snapshot; GET never mutates history."""
        session = self.store.get_session(session_id)
        if session.current_round_id != expected_round_id:
            raise ActiveLearningConflict("this is an older round; refresh first")
        current = self.store.get_round(expected_round_id)
        if current.review:
            raise ValueError("this round already uses the six-score review system")
        next_round = self._compute_round(
            session,
            parent_round=current,
            label_revision=current.label_revision,
            persist=False,
        )
        try:
            _, updated = self.store.commit_round_transition(
                session,
                current,
                (),
                next_round,
                updated_at=_now_iso(),
                label_vocabulary=session.label_vocabulary,
                allow_legacy_upgrade=True,
            )
        except ValueError as exc:
            if "session changed" in str(exc) or "not ready" in str(exc):
                raise ActiveLearningConflict(str(exc)) from exc
            raise
        return {"session": updated.to_dict(), "round": next_round.to_dict()}

    def _review_snapshot(
        self, session, prepared, analysis, active_events, parent_round
    ):
        config = session.config
        runs = {config.min_pts: analysis}
        errors = {}
        labeling_state = _labeling_state(prepared, active_events)
        for setting in (config.min_pts - 1, config.min_pts + 1):
            if 1 <= setting < prepared.version.point_count:
                try:
                    runs[setting] = run_default_analysis(
                        prepared.feature_matrix,
                        n_clusters=config.n_clusters,
                        provider=SsdbcodiProvider(
                            labeling_state=labeling_state, min_pts=setting
                        ),
                    )
                except (ValueError, RuntimeError) as exc:
                    errors[str(setting)] = str(exc)
        confirmed = {
            e.point_id for e in active_events if e.label_dimension != "uncertain"
        }
        statuses = {pid: "labeled" for pid in confirmed}
        reasons = {}
        current_members = member_sets(analysis, prepared.version.point_ids)
        previous_members = (
            member_sets(
                analysis_from_dict(parent_round.analysis), prepared.version.point_ids
            )
            if parent_round
            else None
        )
        just_reviewed = {
            e.point_id
            for e in active_events
            if parent_round and e.round_id == parent_round.round_id
        }
        if current_members is not None and previous_members is not None:
            for pid in confirmed - just_reviewed:
                if current_members[pid] != previous_members[pid]:
                    statuses[pid] = "re_review"
                    reasons[pid] = (
                        "Model status or normal-group membership changed since the previous round."
                    )
        for event in active_events:
            if (
                event.label_dimension == "uncertain"
                and event.point_id not in confirmed
                and parent_round
                and event.round_id == parent_round.round_id
            ):
                statuses[event.point_id] = "deferred"
        budget_reached = (
            config.label_budget is not None
            and len(
                {
                    (e.point_id, e.label_dimension)
                    for e in active_events
                    if e.label_dimension != "uncertain"
                }
            )
            >= config.label_budget
        )
        return build_review(
            prepared.feature_matrix,
            analysis,
            active_events,
            dict(sorted(runs.items())),
            min_pts=config.min_pts,
            statuses=statuses,
            rereview_reasons=reasons,
            batch_size=config.batch_size,
            threshold=config.review_percentile_threshold,
            budget_reached=budget_reached,
            run_errors=errors,
        )

    def _compute_round(
        self,
        session: ActiveLearningSession,
        *,
        parent_round: ActiveLearningRound | None,
        label_revision: int,
        effective_events: Sequence[LabelEvent] | None = None,
        persist: bool = True,
    ) -> ActiveLearningRound:
        prepared = self.store.load_prepared_dataset(session.dataset_version_id)
        active_events = tuple(
            effective_events
            if effective_events is not None
            else self.store.active_label_events(session.session_id)
        )
        labeling_state = _labeling_state(prepared, active_events)
        raw_analysis = run_default_analysis(
            prepared.feature_matrix,
            n_clusters=session.config.n_clusters,
            provider=SsdbcodiProvider(
                labeling_state=labeling_state, min_pts=session.config.min_pts
            ),
        )
        aligned_analysis, lineage = _align_cluster_lineage(raw_analysis, parent_round)
        projection = (
            _projection_from_dict(parent_round.projection)
            if parent_round is not None
            else project_feature_matrix(prepared.feature_matrix)
        )
        tree_config = TreeConfig(
            max_depth=session.config.max_depth,
            min_samples_leaf=session.config.min_samples_leaf,
        )
        rule_set = generate_rule_set(
            prepared.feature_matrix,
            aligned_analysis,
            dataset_id=prepared.version.dataset_id,
            config=tree_config,
        )
        display_rules = _display_rule_set(rule_set, prepared)
        round_index = 0 if parent_round is None else parent_round.round_index + 1
        round_id = _round_id(
            session.session_id,
            round_index,
            label_revision,
            active_events,
            aligned_analysis.analysis_run_id,
        )
        analysis_delta = _analysis_delta(parent_round, aligned_analysis, rule_set)
        review = self._review_snapshot(
            session, prepared, aligned_analysis, active_events, parent_round
        )
        delta = dict(analysis_delta)
        budget_reached = (
            session.config.label_budget is not None
            and len(
                {
                    (e.point_id, e.label_dimension)
                    for e in active_events
                    if e.label_dimension != "uncertain"
                }
            )
            >= session.config.label_budget
        )
        round_status = "stopped" if budget_reached else "ready_for_labeling"
        delta["stop_advice"] = {
            "should_stop": budget_reached,
            "message": "The label budget is complete."
            if budget_reached
            else (
                "No useful recommendation in this round; no points are forced into the queue."
                if not any(c["recommended_point_ids"] for c in review["categories"])
                else ""
            ),
        }
        round_state = ActiveLearningRound(
            round_id=round_id,
            session_id=session.session_id,
            round_index=round_index,
            parent_round_id=parent_round.round_id if parent_round else None,
            label_revision=label_revision,
            status=round_status,
            analysis=aligned_analysis.to_dict(),
            rule_set=rule_set.to_dict(),
            display_rule_set=display_rules,
            projection=projection.to_dict(),
            review=review,
            delta=delta,
            cluster_lineage=lineage,
            created_at=_now_iso(),
        )
        if persist:
            self.store.save_round(round_state)
        return round_state


class ActiveLearningConflict(ValueError):
    pass


def _analysis_delta(
    parent_round: ActiveLearningRound | None,
    analysis: AnalysisResult,
    rule_set: RuleSet,
) -> Dict[str, Any]:
    if parent_round is None:
        current_fingerprints = {
            rule.rule_id: _rule_content_fingerprint(rule.to_dict())
            for rule in rule_set.rules
        }
        return {
            "baseline": True,
            "changed_cluster_point_ids": [],
            "outlier_added_point_ids": [],
            "outlier_removed_point_ids": [],
            "rule_ids_added": [rule.rule_id for rule in rule_set.rules],
            "rule_ids_removed": [],
            "rule_ids_changed": [],
            "rule_fingerprints": current_fingerprints,
            "summary": (
                "This is the first round, before any human labels have changed "
                "the current groups."
            ),
        }
    previous = parent_round.analysis
    previous_assignments = {
        item["point_id"]: item["cluster_id"]
        for item in previous["cluster_result"]["assignments"]
    }
    current_assignments = {
        item.point_id: item.cluster_id for item in analysis.cluster_result.assignments
    }
    changed = sorted(
        point_id
        for point_id in set(previous_assignments) | set(current_assignments)
        if previous_assignments.get(point_id) != current_assignments.get(point_id)
    )
    previous_outliers = set(previous["outlier_result"].get("outlier_point_ids", ()))
    current_outliers = set(analysis.outlier_result.outlier_point_ids)
    previous_rule_payloads = {
        item["rule_id"]: item for item in parent_round.rule_set.get("rules", ())
    }
    current_rule_payloads = {rule.rule_id: rule.to_dict() for rule in rule_set.rules}
    previous_rules = set(previous_rule_payloads)
    current_rules = set(current_rule_payloads)
    previous_fingerprints = {
        rule_id: _rule_content_fingerprint(payload)
        for rule_id, payload in previous_rule_payloads.items()
    }
    current_fingerprints = {
        rule_id: _rule_content_fingerprint(payload)
        for rule_id, payload in current_rule_payloads.items()
    }
    changed_rule_ids = sorted(
        rule_id
        for rule_id in previous_rules & current_rules
        if previous_fingerprints[rule_id] != current_fingerprints[rule_id]
    )
    outlier_added = sorted(current_outliers - previous_outliers)
    outlier_removed = sorted(previous_outliers - current_outliers)
    return {
        "baseline": False,
        "changed_cluster_point_ids": changed,
        "outlier_added_point_ids": outlier_added,
        "outlier_removed_point_ids": outlier_removed,
        "rule_ids_added": sorted(current_rules - previous_rules),
        "rule_ids_removed": sorted(previous_rules - current_rules),
        "rule_ids_changed": changed_rule_ids,
        "rule_fingerprints": current_fingerprints,
        "summary": _analysis_delta_summary(
            changed_group=bool(changed),
            changed_unusual=bool(outlier_added or outlier_removed),
            changed_rules=bool(previous_rules != current_rules or changed_rule_ids),
        ),
    }


def _analysis_delta_summary(
    *,
    changed_group: bool,
    changed_unusual: bool,
    changed_rules: bool,
) -> str:
    changes = []
    if changed_group:
        changes.append("some records moved to a different group")
    if changed_unusual:
        changes.append("the set of records marked unusual changed")
    if changed_rules:
        changes.append("the current group descriptions changed")
    if not changes:
        return (
            "The latest labels did not change the main results, so the remaining "
            "questions still need checking."
        )
    if len(changes) == 1:
        detail = changes[0]
    else:
        detail = ", ".join(changes[:-1]) + f", and {changes[-1]}"
    return f"After the latest labels, {detail}."


def _align_cluster_lineage(
    analysis: AnalysisResult,
    parent_round: ActiveLearningRound | None,
) -> tuple[AnalysisResult, Dict[str, str]]:
    current_groups: Dict[str, set[str]] = {}
    for assignment in analysis.cluster_result.assignments:
        current_groups.setdefault(assignment.cluster_id, set()).add(assignment.point_id)
    previous_groups: Dict[str, set[str]] = {}
    if parent_round is not None:
        for assignment in parent_round.analysis["cluster_result"]["assignments"]:
            previous_groups.setdefault(assignment["cluster_id"], set()).add(
                assignment["point_id"]
            )
    mapping: Dict[str, str] = {}
    used_previous = set()
    for current_id in sorted(current_groups):
        if current_id.startswith("class:"):
            mapping[current_id] = current_id
            used_previous.add(current_id)
    overlaps = []
    for current_id, current_points in current_groups.items():
        if current_id in mapping:
            continue
        for previous_id, previous_points in previous_groups.items():
            if previous_id.startswith("class:"):
                continue
            intersection = len(current_points & previous_points)
            union = len(current_points | previous_points)
            overlaps.append(
                (
                    -intersection,
                    -(intersection / union if union else 0.0),
                    current_id,
                    previous_id,
                )
            )
    for negative_intersection, _, current_id, previous_id in sorted(overlaps):
        if negative_intersection == 0:
            continue
        if current_id in mapping or previous_id in used_previous:
            continue
        mapping[current_id] = previous_id
        used_previous.add(previous_id)
    next_index = 1
    existing = set(previous_groups) | set(mapping.values())
    for current_id in sorted(current_groups):
        if current_id in mapping:
            continue
        while f"group_{next_index:03d}" in existing:
            next_index += 1
        mapping[current_id] = f"group_{next_index:03d}"
        existing.add(mapping[current_id])
        next_index += 1
    assignments = tuple(
        ClusterAssignment(
            point_id=item.point_id,
            cluster_id=mapping[item.cluster_id],
        )
        for item in analysis.cluster_result.assignments
    )
    cluster_payload = {
        "source_cluster_run_id": analysis.cluster_result.cluster_run_id,
        "mapping": mapping,
        "assignments": [item.to_dict() for item in assignments],
    }
    cluster_result = ClusterResult(
        cluster_run_id=f"lineage_{_hash(cluster_payload)[:12]}",
        algorithm=analysis.cluster_result.algorithm,
        n_clusters=max(1, len({item.cluster_id for item in assignments})),
        assignments=assignments,
        excluded_outlier_point_ids=analysis.cluster_result.excluded_outlier_point_ids,
        diagnostics={
            **dict(analysis.cluster_result.diagnostics),
            "cluster_lineage_mapping": mapping,
        },
    )
    aligned = AnalysisResult(
        analysis_run_id=f"analysis_lineage_{_hash({'source': analysis.analysis_run_id, 'mapping': mapping})[:12]}",
        outlier_result=analysis.outlier_result,
        cluster_result=cluster_result,
        diagnostics={
            **dict(analysis.diagnostics),
            "cluster_lineage_mapping": mapping,
        },
    )
    return aligned, mapping


def _display_rule_set(rule_set: RuleSet, prepared: PreparedDataset) -> Dict[str, Any]:
    payload = rule_set.to_dict()
    for rule in payload["rules"]:
        for condition in rule["conditions"]:
            condition.update(
                {
                    "model_feature": condition["feature"],
                    **display_condition(
                        condition["feature"],
                        condition["operator"],
                        condition["threshold"],
                        prepared.version.transformation_map,
                    ),
                }
            )
    payload["diagnostics"] = {
        **dict(payload.get("diagnostics", {})),
        "raw_feature_names": [item.name for item in prepared.version.feature_specs],
        "preprocessing_version": prepared.version.preprocessing_version,
    }
    return payload


def _labeling_state(
    prepared: PreparedDataset,
    events: Iterable[LabelEvent],
) -> LabelingState:
    annotations = []
    for event in events:
        if event.label_dimension == "uncertain":
            continue
        if event.label_dimension == "semantic_class":
            label_type = "class"
            label_value = str(event.label_value)
        else:
            label_type = "outlier"
            label_value = bool(event.label_value)
        annotations.append(
            ManualAnnotation(
                annotation_id=event.event_id,
                dataset_id=prepared.dataset.dataset_id,
                source="active_learning",
                scope="point",
                point_ids=(event.point_id,),
                label_type=label_type,
                label_value=label_value,
                metadata={
                    "round_id": event.round_id,
                    "provenance": dict(event.provenance),
                },
            )
        )
    return LabelingState(
        dataset_id=prepared.dataset.dataset_id,
        annotations=tuple(annotations),
    )


def _semantic_label(
    value: Any,
    vocabulary: Mapping[str, str],
) -> tuple[str, Dict[str, str]]:
    display_name = str(value or "").strip()
    if not display_name:
        raise ValueError("semantic_class label_value must not be empty")
    for label_id, known_name in vocabulary.items():
        if display_name in {label_id, known_name}:
            return label_id, dict(vocabulary)
    base = (
        "".join(char.lower() if char.isalnum() else "_" for char in display_name).strip(
            "_"
        )
        or "class"
    )
    label_id = f"label_{base}"
    if label_id in vocabulary and vocabulary[label_id] != display_name:
        label_id = f"{label_id}_{_hash(display_name)[:6]}"
    updated = dict(vocabulary)
    updated[label_id] = display_name
    return label_id, updated


def _coerce_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "outlier"}:
            return True
        if normalized in {"false", "0", "no", "normal"}:
            return False
    raise ValueError(f"{field_name} must be a boolean")


def _round_id(
    session_id: str,
    round_index: int,
    label_revision: int,
    events: Sequence[LabelEvent],
    analysis_run_id: str,
) -> str:
    payload = {
        "session_id": session_id,
        "round_index": round_index,
        "label_revision": label_revision,
        "events": [event.event_id for event in events],
        "analysis_run_id": analysis_run_id,
    }
    return f"alround_{round_index:04d}_{_hash(payload)[:12]}"


def _projection_from_dict(payload: Mapping[str, Any]) -> ProjectionResult:
    return ProjectionResult(
        projection_id=payload["projection_id"],
        method=payload["method"],
        coordinates=tuple(
            ProjectionCoordinate(
                point_id=item["point_id"],
                x=item["x"],
                y=item["y"],
            )
            for item in payload.get("coordinates", ())
        ),
    )


def rule_set_from_dict(payload: Mapping[str, Any]) -> RuleSet:
    return RuleSet(
        rule_set_id=payload["rule_set_id"],
        dataset_id=payload["dataset_id"],
        source_analysis_run_id=payload["source_analysis_run_id"],
        model=payload.get("model", {}),
        rules=tuple(
            RuleCard(
                rule_id=item["rule_id"],
                target_kind=item["target_kind"],
                target_id=item["target_id"],
                conditions=tuple(
                    RuleCondition(
                        feature=condition["feature"],
                        operator=condition["operator"],
                        threshold=condition["threshold"],
                    )
                    for condition in item.get("conditions", ())
                ),
                support_count=item["support_count"],
                coverage=item["coverage"],
                purity=item["purity"],
                matched_point_ids=tuple(item.get("matched_point_ids", ())),
                exception_point_ids=tuple(item.get("exception_point_ids", ())),
                diagnostics=item.get("diagnostics", {}),
            )
            for item in payload.get("rules", ())
        ),
        diagnostics=payload.get("diagnostics", {}),
    )


def analysis_from_dict(payload: Mapping[str, Any]) -> AnalysisResult:
    cluster = payload["cluster_result"]
    outlier = payload["outlier_result"]
    return AnalysisResult(
        analysis_run_id=payload["analysis_run_id"],
        cluster_result=ClusterResult(
            cluster_run_id=cluster["cluster_run_id"],
            algorithm=cluster["algorithm"],
            n_clusters=cluster["n_clusters"],
            assignments=tuple(
                ClusterAssignment(
                    point_id=item["point_id"],
                    cluster_id=item["cluster_id"],
                )
                for item in cluster.get("assignments", ())
            ),
            excluded_outlier_point_ids=tuple(
                cluster.get("excluded_outlier_point_ids", ())
            ),
            diagnostics=cluster.get("diagnostics", {}),
        ),
        outlier_result=OutlierResult(
            outlier_run_id=outlier["outlier_run_id"],
            algorithm=outlier["algorithm"],
            scores=tuple(
                OutlierScore(
                    point_id=item["point_id"],
                    score=item["score"],
                    is_outlier=item["is_outlier"],
                )
                for item in outlier.get("scores", ())
            ),
            diagnostics=outlier.get("diagnostics", {}),
        ),
        diagnostics=payload.get("diagnostics", {}),
    )


def _rule_content_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = {
        "target_kind": payload.get("target_kind"),
        "target_id": payload.get("target_id"),
        "conditions": [
            {
                "feature": item.get("feature"),
                "operator": item.get("operator"),
                "threshold": round(float(item.get("threshold", 0.0)), 9),
            }
            for item in payload.get("conditions", ())
        ],
        "matched_point_ids": sorted(payload.get("matched_point_ids", ())),
        "exception_point_ids": sorted(payload.get("exception_point_ids", ())),
    }
    return _hash(canonical)[:16]


def _hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
