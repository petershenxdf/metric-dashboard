from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple
from uuid import uuid4

import numpy as np

from app.modules.algorithm_adapters.service import run_default_analysis
from app.modules.labeling.schemas import LabelingState, ManualAnnotation
from app.modules.projection.service import project_feature_matrix, scaled_projection_points
from app.modules.rule_panel.recommendation import (
    RECOMMENDATION_CATEGORIES,
    build_recommendation_plan,
)
from app.modules.rule_panel.schemas import (
    RuleCard,
    RuleCondition,
    RuleSet,
    TreeConfig,
)
from app.modules.rule_panel.service import generate_rule_set
from app.shared.deepseek import DEEPSEEK_PRO_MODEL
from app.shared.plot_guidance import recommendation_badges
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
from .evidence import (
    CATEGORY_EXPLANATIONS,
    EVIDENCE_POLICY_VERSION,
    build_category_evidence_cards,
    build_evidence_cards_for_plans,
)
from .schemas import (
    ActiveLearningRound,
    ActiveLearningSession,
    LabelEvent,
    RecommendationPlanV2,
    SessionConfig,
)
from .store import ActiveLearningStore
from .translation import (
    PROMPT_VERSION as TRANSLATION_PROMPT_VERSION,
    build_translation_packet,
    translate_plan,
)


ACTIVE_PLAN_VERSION = "active_recommendation_v3"
EXACT_SSDBCODI_MAX_POINTS = 2000
PLOT_WIDTH = 860
PLOT_HEIGHT = 520
_CATEGORY_PRIORITY = {
    "exception_relabel_review": 8,
    "overlap_merge_signal": 7,
    "boundary_review": 6,
    "anomaly_label_review": 5,
    "split_or_new_cluster_signal": 4,
    "rule_confidence_audit": 3,
    "feature_label_strategy": 2,
}
_PLAIN_CATEGORY_DESCRIPTIONS = CATEGORY_EXPLANATIONS


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
        session_config = config if isinstance(config, SessionConfig) else SessionConfig.from_dict(config)
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
        plan_id: str,
        category: str,
        labels: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        session = self.store.get_session(session_id)
        if session.status != "active":
            raise ActiveLearningConflict(
                f"session is {session.status}; refresh before submitting labels"
            )
        if expected_round_id is not None and expected_round_id != round_id:
            raise ActiveLearningConflict("expected_round_id does not match the request path")
        if session.current_round_id != round_id:
            raise ActiveLearningConflict("this page is based on an older active-learning round")
        current_round = self.store.get_round(round_id)
        if current_round.label_revision != expected_label_revision:
            raise ActiveLearningConflict("label revision changed; refresh before submitting labels")
        known_plan_ids = {
            plan.get("plan_id")
            for plan in current_round.recommendation_plans.values()
            if isinstance(plan, Mapping)
        }
        if plan_id not in known_plan_ids:
            raise ActiveLearningConflict("recommendation plan changed; refresh before submitting labels")
        if category not in current_round.recommendation_plans:
            raise ValueError("unknown recommendation category")
        if not labels:
            raise ValueError("labels must contain at least one label")

        prepared = self.store.load_prepared_dataset(session.dataset_version_id)
        known_points = set(prepared.version.point_ids)
        vocabulary = dict(session.label_vocabulary)
        now = _now_iso()
        events = []
        submission_keys = set()
        for index, payload in enumerate(labels):
            point_id = str(payload.get("point_id", "")).strip()
            if point_id not in known_points:
                raise ValueError(f"unknown point id: {point_id}")
            dimension = str(payload.get("label_dimension", "")).strip()
            if dimension not in {"semantic_class", "outlier_status", "uncertain"}:
                raise ValueError("label_dimension must be semantic_class, outlier_status, or uncertain")
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
                        "plan_id": plan_id,
                        "category": category,
                        "recommended": point_id
                        in set(
                            current_round.recommendation_plans[category].get(
                                "recommended_point_ids", ()
                            )
                        ),
                        "suggested_label": payload.get("suggested_label"),
                        "user_modified_suggestion": bool(
                            payload.get("user_modified_suggestion", False)
                        ),
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

        replaced_keys = {
            (event.point_id, event.label_dimension) for event in events
        }
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
        self.store.revert_events_to_round(session_id, target.round_id)
        updated = self.store.update_session_current(
            session,
            current_round_id=target.round_id,
            status="active",
            updated_at=_now_iso(),
        )
        return {"session": updated.to_dict(), "round": target.to_dict()}

    def session_state(
        self,
        session_id: str,
        *,
        focus_category: str = "label_priority",
    ) -> Mapping[str, Any]:
        session = self.store.get_session(session_id)
        if session.current_round_id is None:
            raise ValueError("active-learning session has no current round")
        round_state = self.store.get_round(session.current_round_id)
        prepared = self.store.load_prepared_dataset(session.dataset_version_id)
        plans = self._plans_with_current_evidence(
            session,
            round_state,
            prepared,
        )
        if focus_category not in plans:
            focus_category = "label_priority"
        plan = self._plan_with_current_evidence(
            session,
            round_state,
            prepared,
            plans[focus_category],
        )
        analysis = round_state.analysis
        assignments = {
            item["point_id"]: item["cluster_id"]
            for item in analysis["cluster_result"]["assignments"]
        }
        outlier_ids = set(analysis["outlier_result"].get("outlier_point_ids", ()))
        projection = _projection_from_dict(round_state.projection)
        plot_points = []
        raw_by_id = {
            item["point_id"]: item for item in prepared.raw_records
        }
        recommended = tuple(plan.get("recommended_point_ids", ()))
        for point in scaled_projection_points(projection, assignments):
            raw = raw_by_id[point["point_id"]]
            plot_points.append(
                {
                    **point,
                    "cluster_id": assignments.get(point["point_id"], ""),
                    "is_outlier": point["point_id"] in outlier_ids,
                    "raw_features": dict(raw.get("raw_features", {})),
                    "metadata": dict(raw.get("metadata", {})),
                    "recommended": point["point_id"] in recommended,
                }
            )
        plot_legend = []
        for point in plot_points:
            group_id = str(point.get("cluster_id", ""))
            if not group_id or any(
                item["group_id"] == group_id for item in plot_legend
            ):
                continue
            plot_legend.append(
                {
                    "group_id": group_id,
                    "color": point["color"],
                }
            )
        plot_legend.sort(key=lambda item: item["group_id"])
        badges = recommendation_badges(
            recommended,
            plot_points,
            plot_width=PLOT_WIDTH,
            plot_height=PLOT_HEIGHT,
        )
        category_cards = tuple(
            {
                "category": category,
                "label": category.replace("_", " ").title(),
                "description": _PLAIN_CATEGORY_DESCRIPTIONS[category],
                "is_active": category == focus_category,
                "has_typical_case": bool(plans[category].get("has_typical_case")),
                "candidate_pool_count": int(plans[category].get("candidate_pool_count", 0)),
                "recommended_point_count": int(
                    plans[category].get("recommended_point_count", 0)
                ),
                "stop_reason": plans[category].get("stop_reason", ""),
            }
            for category in RECOMMENDATION_CATEGORIES
        )
        active_events = self.store.active_label_events(session_id)
        history = self.store.list_rounds(session_id)
        guidance = deterministic_point_guidance(
            plan,
            entity_name=prepared.version.entity_name,
            delta=round_state.delta,
            vocabulary=session.label_vocabulary,
        )
        point_category_coverage = {
            item.get("point_id"): [
                category
                for category in item.get("covered_categories", ())
                if category != focus_category
            ]
            for item in plan.get("point_profiles", ())
        }
        round_payload = round_state.to_dict()
        round_payload["recommendation_plans"] = {
            category: dict(item) for category, item in plans.items()
        }
        return {
            "workflow": "active-learning-dashboard",
            "session": session.to_dict(),
            "round": round_payload,
            "dataset_version": prepared.version.to_dict(),
            "analysis": analysis,
            "rule_set": round_state.display_rule_set,
            "model_rule_set": round_state.rule_set,
            "recommendation_plan": plan,
            "focus_category": focus_category,
            "category_cards": list(category_cards),
            "guidance": guidance,
            "point_category_coverage": point_category_coverage,
            "plot_points": plot_points,
            "plot_legend": plot_legend,
            "guidance_point_badges": badges,
            "plot_width": PLOT_WIDTH,
            "plot_height": PLOT_HEIGHT,
            "active_labels": [event.to_dict() for event in active_events],
            "history": [
                {
                    "round_id": item.round_id,
                    "round_index": item.round_index,
                    "label_revision": item.label_revision,
                    "status": item.status,
                    "delta": dict(item.delta),
                    "created_at": item.created_at,
                    "is_current": item.round_id == session.current_round_id,
                }
                for item in history
            ],
        }

    def interpret_category(
        self,
        session_id: str,
        *,
        round_id: str,
        category: str,
        provider_kind: str,
    ) -> Mapping[str, Any]:
        session = self.store.get_session(session_id)
        round_state = self.store.get_round(round_id)
        if round_state.session_id != session_id:
            raise ValueError("round does not belong to this session")
        if category not in round_state.recommendation_plans:
            raise ValueError("unknown recommendation category")
        prepared = self.store.load_prepared_dataset(session.dataset_version_id)
        plan = self._plan_with_current_evidence(
            session,
            round_state,
            prepared,
            round_state.recommendation_plans[category],
        )
        cached = self.store.get_interpretation(
            round_id,
            str(plan.get("plan_id")),
            provider_kind,
        )
        if (
            cached is not None
            and cached.get("diagnostics", {}).get("prompt_template_version")
            == TRANSLATION_PROMPT_VERSION
        ):
            return cached
        deterministic = deterministic_point_guidance(
            plan,
            entity_name=prepared.version.entity_name,
            delta=round_state.delta,
            vocabulary=session.label_vocabulary,
        )
        if provider_kind == "deepseek" and not plan.get("has_typical_case"):
            result = {
                "guidance": deterministic,
                "diagnostics": {
                    "provider_kind": "deepseek",
                    "provider_label": f"deepseek:{DEEPSEEK_PRO_MODEL}:skipped",
                    "requested_model_name": DEEPSEEK_PRO_MODEL,
                    "model_name": DEEPSEEK_PRO_MODEL,
                    "using_deepseek_v4_pro": False,
                    "used_fallback": False,
                    "deepseek_skipped": True,
                    "skip_reason": plan.get(
                        "stop_reason",
                        "no_typical_case_for_category",
                    ),
                    "prompt_template_version": TRANSLATION_PROMPT_VERSION,
                },
            }
            self.store.save_interpretation(
                session_id=session_id,
                round_id=round_id,
                plan_id=str(plan.get("plan_id")),
                provider_kind=provider_kind,
                payload=result["guidance"],
                diagnostics=result["diagnostics"],
                updated_at=_now_iso(),
            )
            return {
                **result,
                "translation_packet": None,
                "cache_hit": False,
            }
        previous_events = [
            event.to_dict()
            for event in self.store.all_label_events(session_id)
            if event.provenance.get("result_round_id") == round_state.round_id
        ]
        packet = build_translation_packet(
            plan=plan,
            display_rule_set=round_state.display_rule_set,
            round_delta=round_state.delta,
            previous_label_events=previous_events,
            entity_name=prepared.version.entity_name,
        )
        result = translate_plan(
            packet,
            provider_kind=provider_kind,
            deterministic_fallback=deterministic,
        )
        self.store.save_interpretation(
            session_id=session_id,
            round_id=round_id,
            plan_id=str(plan.get("plan_id")),
            provider_kind=provider_kind,
            payload=result["guidance"],
            diagnostics=result["diagnostics"],
            updated_at=_now_iso(),
        )
        return {
            **result,
            "translation_packet": packet.to_dict(),
            "cache_hit": False,
        }

    def _plan_with_current_evidence(
        self,
        session: ActiveLearningSession,
        round_state: ActiveLearningRound,
        prepared: PreparedDataset,
        source_plan: Mapping[str, Any],
    ) -> Dict[str, Any]:
        plan = dict(source_plan)
        cards = tuple(plan.get("category_evidence_cards", ()))
        has_current_contract = (
            plan.get("evidence_policy_version") == EVIDENCE_POLICY_VERSION
            and all(
                all(
                    bullet.get("point_connection")
                    and bullet.get("labeling_value")
                    for bullet in card.get("evidence_bullets", ())
                )
                for card in cards
            )
        )
        if has_current_contract:
            return plan

        parent_round = (
            self.store.get_round(round_state.parent_round_id)
            if round_state.parent_round_id
            else None
        )
        refreshed_cards = build_category_evidence_cards(
            plan,
            prepared=prepared,
            analysis=analysis_from_dict(round_state.analysis),
            rule_set=rule_set_from_dict(round_state.rule_set),
            projection=_projection_from_dict(round_state.projection),
            active_events=self.store.active_label_events(session.session_id),
            parent_round=parent_round,
            label_vocabulary=session.label_vocabulary,
        )
        plan["category_explanation"] = CATEGORY_EXPLANATIONS[
            plan["focus_category"]
        ]
        plan["evidence_policy_version"] = EVIDENCE_POLICY_VERSION
        plan["category_evidence_cards"] = list(refreshed_cards)
        RecommendationPlanV2.from_dict(plan)
        return plan

    def _plans_with_current_evidence(
        self,
        session: ActiveLearningSession,
        round_state: ActiveLearningRound,
        prepared: PreparedDataset,
    ) -> Dict[str, Mapping[str, Any]]:
        plans = {
            category: dict(plan)
            for category, plan in round_state.recommendation_plans.items()
        }
        if all(
            plan.get("evidence_policy_version") == EVIDENCE_POLICY_VERSION
            and all(
                all(
                    bullet.get("point_connection")
                    and bullet.get("labeling_value")
                    for bullet in card.get("evidence_bullets", ())
                )
                for card in plan.get("category_evidence_cards", ())
            )
            for plan in plans.values()
        ):
            return plans

        parent_round = (
            self.store.get_round(round_state.parent_round_id)
            if round_state.parent_round_id
            else None
        )
        cards_by_category = build_evidence_cards_for_plans(
            plans,
            prepared=prepared,
            analysis=analysis_from_dict(round_state.analysis),
            rule_set=rule_set_from_dict(round_state.rule_set),
            projection=_projection_from_dict(round_state.projection),
            active_events=self.store.active_label_events(session.session_id),
            parent_round=parent_round,
            label_vocabulary=session.label_vocabulary,
        )
        refreshed = {}
        for category, source_plan in plans.items():
            plan = dict(source_plan)
            plan["category_explanation"] = CATEGORY_EXPLANATIONS[category]
            plan["evidence_policy_version"] = EVIDENCE_POLICY_VERSION
            plan["category_evidence_cards"] = list(
                cards_by_category[category]
            )
            RecommendationPlanV2.from_dict(plan)
            refreshed[category] = plan
        return refreshed

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
            labeling_state=labeling_state if labeling_state.annotations else None,
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
        plans = self._recommendation_plans(
            session,
            round_id,
            label_revision,
            prepared,
            aligned_analysis,
            rule_set,
            projection,
            parent_round,
            active_events,
            analysis_delta,
        )
        delta = _finish_delta(analysis_delta, parent_round, plans)
        round_status, stop_advice = _round_lifecycle(plans, delta)
        delta["stop_advice"] = stop_advice
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
            recommendation_plans=plans,
            delta=delta,
            cluster_lineage=lineage,
            created_at=_now_iso(),
        )
        if persist:
            self.store.save_round(round_state)
        return round_state

    def _recommendation_plans(
        self,
        session: ActiveLearningSession,
        round_id: str,
        label_revision: int,
        prepared: PreparedDataset,
        analysis: AnalysisResult,
        rule_set: RuleSet,
        projection: ProjectionResult,
        parent_round: ActiveLearningRound | None,
        active_events: Tuple[LabelEvent, ...],
        analysis_delta: Mapping[str, Any],
    ) -> Dict[str, Mapping[str, Any]]:
        effective_label_count = len(
            [
                event
                for event in active_events
                if event.label_dimension != "uncertain"
            ]
        )
        if (
            session.config.label_budget is not None
            and effective_label_count >= session.config.label_budget
        ):
            return _stopped_recommendation_plans(
                session=session,
                round_id=round_id,
                label_revision=label_revision,
                prepared=prepared,
                parent_round=parent_round,
                stop_reason="label_budget_reached",
            )
        history_rounds = (
            self.store.round_ancestry(
                session.session_id,
                parent_round.round_id,
            )
            if parent_round is not None
            else ()
        )
        history_round_ids = {item.round_id for item in history_rounds}
        history_labels = tuple(
            event
            for event in self.store.all_label_events(session.session_id)
            if (
                event.provenance.get("result_round_id") in history_round_ids
                or (
                    not event.provenance.get("result_round_id")
                    and event.round_id in history_round_ids
                )
            )
        )
        recommendation_history = _recommendation_history_summary(
            history_rounds,
            shown_events=self.store.recommendation_events(
                session.session_id,
                history_round_ids,
            ),
            label_events=history_labels,
        )
        history_counts = recommendation_history["shown"]
        plans: Dict[str, Mapping[str, Any]] = {}
        for category in RECOMMENDATION_CATEGORIES:
            if category == "label_priority":
                continue
            base = build_recommendation_plan(
                rule_set,
                analysis_result=analysis,
                feature_matrix=prepared.feature_matrix,
                focus_category=category,
                candidate_pool_limit=prepared.version.point_count,
            )
            plans[category] = _history_aware_plan(
                base,
                session=session,
                round_id=round_id,
                label_revision=label_revision,
                prepared=prepared,
                analysis=analysis,
                rule_set=rule_set,
                parent_round=parent_round,
                active_events=active_events,
                analysis_delta=analysis_delta,
                recommendation_history_counts=history_counts,
                recommendation_history=recommendation_history,
            ).to_dict()
        plans = _annotate_cross_category_coverage(plans)
        plans["label_priority"] = _meta_priority_plan(
            plans,
            session=session,
            round_id=round_id,
            label_revision=label_revision,
            prepared=prepared,
            parent_round=parent_round,
        ).to_dict()
        cards_by_category = build_evidence_cards_for_plans(
            plans,
            prepared=prepared,
            analysis=analysis,
            rule_set=rule_set,
            projection=projection,
            active_events=active_events,
            parent_round=parent_round,
            label_vocabulary=session.label_vocabulary,
        )
        evidence_plans: Dict[str, Mapping[str, Any]] = {}
        for category in RECOMMENDATION_CATEGORIES:
            payload = dict(plans[category])
            payload["category_explanation"] = CATEGORY_EXPLANATIONS[category]
            payload["evidence_policy_version"] = EVIDENCE_POLICY_VERSION
            payload["category_evidence_cards"] = list(
                cards_by_category[category]
            )
            RecommendationPlanV2.from_dict(payload)
            evidence_plans[category] = payload
        return evidence_plans


class ActiveLearningConflict(ValueError):
    pass


def deterministic_point_guidance(
    plan: Mapping[str, Any],
    *,
    entity_name: str,
    delta: Mapping[str, Any],
    vocabulary: Mapping[str, str],
) -> Dict[str, Any]:
    category = str(plan.get("focus_category", "label_priority"))
    delegated_category = str(
        plan.get("history_context", {}).get("delegated_category", "")
    )
    evidence_category = (
        delegated_category
        if category == "label_priority" and delegated_category
        else category
    )
    recommended = tuple(plan.get("recommended_point_ids", ()))
    profiles = {
        item.get("point_id"): item
        for item in plan.get("point_profiles", ())
        if isinstance(item, Mapping)
    }
    ranking = {
        item.get("point_id"): item
        for item in plan.get("candidate_rankings", ())
        if isinstance(item, Mapping)
    }
    evidence_cards = {
        item.get("point_id"): item
        for item in plan.get("category_evidence_cards", ())
        if isinstance(item, Mapping)
    }
    point_guidance = []
    for point_id in recommended:
        row = ranking.get(point_id, {})
        profile = profiles.get(point_id, {})
        evidence_card = dict(evidence_cards.get(point_id, {}))
        if evidence_card:
            (
                why_selected,
                how_to_label,
                possible_outcomes,
            ) = _plain_card_guidance(
                evidence_category,
                entity_name=entity_name,
                evidence_card=evidence_card,
            )
            evidence_bullets = [
                {
                    "dimension_id": item.get("dimension_id"),
                    "question": item.get("question"),
                    "status": item.get("status"),
                    "headline": item.get("headline"),
                    "explanation": str(
                        item.get("plain_fact", "")
                    ).strip(),
                    "why_this_point": _plain_point_connection(item),
                    "evidence_fact_ids": list(
                        item.get("evidence_fact_ids", ())
                    ),
                    "technical_details": dict(
                        item.get("technical_details", {})
                    ),
                }
                for item in evidence_card.get("evidence_bullets", ())
            ]
            comparison_targets = [
                dict(item)
                for item in evidence_card.get("comparison_targets", ())
            ]
        else:
            why_selected, how_to_label, possible_outcomes = _plain_point_guidance(
                evidence_category,
                entity_name=entity_name,
                profile=profile,
            )
            evidence_bullets = []
            comparison_targets = []
        point_guidance.append(
            {
                "point_id": point_id,
                "why_selected": why_selected,
                "what_changed_since_last_round": _plain_round_change(
                    point_id,
                    plan=plan,
                    profile=profile,
                    ranking=row,
                    delta=delta,
                ),
                "how_to_label": how_to_label,
                "possible_outcomes": possible_outcomes,
                "evidence_bullets": evidence_bullets,
                "comparison_targets": comparison_targets,
            }
        )
    return {
        "provider_kind": "deterministic",
        "plan_id": plan.get("plan_id"),
        "category": category,
        "recommended_point_ids": list(recommended),
        "target_rule_ids": list(plan.get("target_rule_ids", ())),
        "label_options": list(plan.get("label_options", ())),
        "category_explanation": str(
            plan.get("category_explanation")
            or _PLAIN_CATEGORY_DESCRIPTIONS.get(category, "")
        ),
        "summary": _plain_guidance_summary(
            category,
            evidence_category=evidence_category,
            recommended_count=len(recommended),
            entity_name=entity_name,
        ),
        "point_guidance": point_guidance,
        "label_vocabulary": dict(vocabulary),
        "warnings": [plan.get("stop_reason")] if plan.get("stop_reason") else [],
    }


def _plain_point_connection(item: Mapping[str, Any]) -> str:
    plain_fact = str(item.get("plain_fact", "")).strip()
    point_connection = str(
        item.get("point_connection", plain_fact)
    ).strip()
    labeling_value = str(
        item.get("labeling_value")
        or item.get("why_it_matters", "")
    ).strip()
    parts = []
    if point_connection and point_connection != plain_fact:
        parts.append(point_connection)
    if labeling_value and labeling_value not in parts:
        parts.append(labeling_value)
    return " ".join(parts).strip()


def _plain_card_guidance(
    category: str,
    *,
    entity_name: str,
    evidence_card: Mapping[str, Any],
) -> tuple[str, str, list[str]]:
    bullets = [
        dict(item) for item in evidence_card.get("evidence_bullets", ())
    ]
    useful = [
        item
        for item in bullets
        if item.get("status") in {"yes", "partly"}
    ]
    summary_source = useful or bullets
    why_selected = " ".join(
        str(item.get("headline", "")).strip()
        for item in summary_source[:2]
        if str(item.get("headline", "")).strip()
    )
    if not why_selected:
        why_selected = (
            f"This {entity_name} is the clearest available example for the "
            "current labeling question."
        )

    targets = [
        dict(item) for item in evidence_card.get("comparison_targets", ())
    ]
    target_names = [
        str(item.get("point_id"))
        for item in targets
        if item.get("point_id")
    ]
    features = []
    for target in targets:
        for feature in target.get("features_to_compare", ()):
            feature_text = _plain_feature_name(feature)
            if feature_text and feature_text not in features:
                features.append(feature_text)
    target_text = (
        f"Compare it with {', '.join(target_names)}"
        if target_names
        else "Compare it with the clearest reviewed examples nearby"
    )
    feature_text = (
        f", paying particular attention to {', '.join(features[:2])}"
        if features
        else ", using the original fields and real-world meaning"
    )
    decision = {
        "boundary_review": (
            "Decide whether the records represent the same real-world type or "
            "whether the difference between the two sides matters."
        ),
        "overlap_merge_signal": (
            "Decide whether the shared area has one real-world meaning or "
            "contains genuinely different types."
        ),
        "split_or_new_cluster_signal": (
            "Reuse an existing type when it fits; use a new type only when the "
            "difference is meaningful in the domain."
        ),
        "anomaly_label_review": (
            "Check the source record first, then decide whether it is truly "
            "unusual, rare but valid, or normal."
        ),
        "exception_relabel_review": (
            "Choose the real-world type that best fits the full record, then "
            "treat the rule mismatch as a separate warning."
        ),
        "feature_label_strategy": (
            "Label from the complete record first, then check whether the named "
            "field would have led to the same decision."
        ),
        "rule_confidence_audit": (
            "Label from domain knowledge without treating the current group "
            "description as the answer."
        ),
    }.get(
        category,
        "Choose the real-world type that best matches the complete record.",
    )
    how_to_label = f"{target_text}{feature_text}. {decision}"
    outcomes = {
        "boundary_review": [
            "If both sides receive the same human type, the next round should question whether the dividing line is meaningful.",
            "If the sides receive different human types, the current distinction gains human support.",
        ],
        "overlap_merge_signal": [
            "Matching human types would support reviewing whether the two descriptions are unnecessarily separate.",
            "Different human types would show that the shared area still contains a meaningful distinction.",
        ],
        "split_or_new_cluster_signal": [
            "An existing human type would connect this area to what is already known.",
            "A consistently different human type would justify checking this area separately in the next round.",
        ],
        "anomaly_label_review": [
            "A confirmed unusual label would help the next round recognize similar rare records.",
            "A normal label would show that rarity alone should not make this record unusual.",
        ],
        "exception_relabel_review": [
            "Agreement with the current human type would show that the simple rule needs room for valid exceptions.",
            "A different human type would tell the next round to question the current placement or description.",
        ],
        "feature_label_strategy": [
            "Agreement would support the field as a useful clue for similar records.",
            "Disagreement would show that future guidance should rely less on this field alone.",
        ],
        "rule_confidence_audit": [
            "Agreement would make the description more credible for similar records.",
            "Disagreement would identify where the description needs more human checks.",
        ],
    }.get(
        category,
        [
            "A clear label will give the next round stronger evidence for similar records.",
            "An uncertain answer will keep the point visible without forcing a weak conclusion.",
        ],
    )
    return (
        " ".join(why_selected.split()),
        " ".join(how_to_label.split()),
        [" ".join(item.split()) for item in outcomes],
    )


def _plain_guidance_summary(
    category: str,
    *,
    evidence_category: str,
    recommended_count: int,
    entity_name: str,
) -> str:
    if not recommended_count:
        return "There is no clear example for this question in the current round."
    noun = entity_name if recommended_count == 1 else f"{entity_name}s"
    question = {
        "boundary_review": "checking the current dividing line",
        "overlap_merge_signal": "checking an area described by more than one group",
        "split_or_new_cluster_signal": "checking a separated or poorly described area",
        "anomaly_label_review": "checking records that currently look unusual",
        "exception_relabel_review": "checking records that do not fit their group description",
        "feature_label_strategy": "checking whether a feature-based rule matches human meaning",
        "rule_confidence_audit": (
            "checking whether the current group description agrees with "
            "human judgment"
        ),
    }.get(evidence_category, "answering the most important open labeling question")
    prefix = "Start with" if category == "label_priority" else "Check"
    return (
        f"{prefix} these {recommended_count} {noun}. They are the clearest "
        f"available examples for {question}."
    )


def _plain_point_guidance(
    category: str,
    *,
    entity_name: str,
    profile: Mapping[str, Any],
) -> tuple[str, str, list[str]]:
    evidence = dict(profile.get("plain_language_evidence", {}))
    boundary = dict(evidence.get("closest_boundary", {}))
    feature = _plain_feature_name(
        boundary.get("feature_name", "a feature used by the current rule")
    )
    proximity = str(boundary.get("proximity", "near"))
    relation = str(boundary.get("relation", "near"))
    affected_scope = str(evidence.get("affected_scope", "several records"))
    unusual_level = str(evidence.get("unusual_level", "moderate"))

    boundary_sentence = (
        f"This {entity_name} sits {relation} the current dividing line based on "
        f"{feature}."
        if boundary
        else (
            f"This {entity_name} lies in an area where the current grouping is "
            "not easy to judge."
        )
    )
    if proximity == "not especially close":
        boundary_sentence = (
            f"This {entity_name} is linked to a dividing rule based on {feature}, "
            "but it is not especially close to that line."
        )
    unusual_sentence = {
        "high": "The system currently sees it as strongly unusual.",
        "moderate": "The system currently sees it as somewhat unusual.",
        "low": "The system does not currently see it as especially unusual.",
    }.get(unusual_level, "")
    exception_sentence = (
        "It also does not fit the usual rule for its current group."
        if profile.get("is_rule_exception")
        else ""
    )

    if category == "boundary_review":
        why = (
            f"{boundary_sentence} A human label here can show whether the line "
            f"separates meaningful types, and the same line affects {affected_scope}. "
            f"{unusual_sentence} {exception_sentence}"
        )
        how = (
            f"Compare {feature} and the rest of the record with a few clearly "
            "understood records from both nearby groups. Choose the existing "
            "real-world type it resembles most. Use a new type only when neither "
            "group fits, and choose uncertain when the evidence is mixed."
        )
        outcomes = [
            "If it matches the group on its current side, the present dividing line gains support.",
            "If it matches the other side or neither side, the next round can recheck this line and nearby records.",
        ]
    elif category == "overlap_merge_signal":
        why = (
            f"This {entity_name} is described by more than one current group. "
            "Its human label can show whether those descriptions refer to the "
            f"same real-world type or only overlap by accident. {boundary_sentence}"
        )
        how = (
            f"Ignore the group numbers at first. Compare {feature} and the rest "
            "of the record with clear examples from each group, then choose the "
            "one real-world type that fits best. Choose uncertain if both remain plausible."
        )
        outcomes = [
            "The same human type on both sides suggests the current group descriptions may be unnecessarily separate.",
            "Different human types show that the overlap is real but should remain distinguishable.",
        ]
    elif category == "split_or_new_cluster_signal":
        why = (
            f"This {entity_name} comes from a separated or poorly described part "
            "of its current group. Its label can show whether that area still belongs "
            f"to a known type or represents something meaningfully different. "
            f"{boundary_sentence} {unusual_sentence} {exception_sentence}"
        )
        how = (
            f"Compare {feature} and the rest of the record with the clearest "
            "examples of each existing real-world type. Reuse a type when the "
            "important characteristics agree. Create a new type only when the "
            "difference is meaningful in the user's domain."
        )
        outcomes = [
            "An existing type keeps this area connected to what is already known.",
            "A genuinely different type gives the next round evidence to examine this area separately.",
        ]
    elif category == "anomaly_label_review":
        why = (
            f"{unusual_sentence} {boundary_sentence} A person needs to decide "
            "whether it is a valid rare case, an ordinary member, or a data problem."
        )
        how = (
            f"Check {feature} and the rest of the original record for an obvious "
            "error first. If it is valid, compare it with normal examples of the "
            "same real-world type. Mark it unusual only when the difference is "
            "meaningful, not merely uncommon."
        )
        outcomes = [
            "A confirmed unusual case helps the next round recognize similar rare records.",
            "A normal label tells the next round that rarity alone should not make this record unusual.",
        ]
    elif category == "exception_relabel_review":
        why = (
            f"This {entity_name} does not fit the usual description of its current group. "
            "It may be mislabeled, sit near a dividing line, or simply be a valid "
            f"exception that the current rule cannot describe. {boundary_sentence} "
            f"{unusual_sentence}"
        )
        how = (
            f"Check its original meaning, especially {feature}, before looking at "
            "the system's group. Choose the real-world type it best represents, "
            "then decide whether the mismatch is a data error, a boundary case, "
            "or a valid exception."
        )
        outcomes = [
            "Agreement with its current group shows that the rule needs room for valid exceptions.",
            "A different human type tells the next round to question this assignment or the nearby boundary.",
        ]
    elif category == "feature_label_strategy":
        why = (
            f"{boundary_sentence} The open question is whether this feature-based "
            "difference also matches a difference that matters to a person."
        )
        how = (
            f"Look at {feature} together with the record's overall real-world meaning. "
            "Choose the type based on the full record, then note whether this one "
            "feature gives the same answer or a misleading shortcut."
        )
        outcomes = [
            "Agreement means this feature is a useful human-facing clue for similar records.",
            "Disagreement means future guidance should rely less on this feature alone.",
        ]
    elif category == "rule_confidence_audit":
        why = (
            f"This {entity_name} is an informative test of a rule used for "
            f"{affected_scope}. Checking it helps avoid trusting a rule simply "
            f"because it fits the current computer-generated groups. {boundary_sentence} "
            f"{exception_sentence}"
        )
        how = (
            f"Label the record from domain knowledge first, without treating the "
            f"rule as the answer. Then compare the human label with {feature}, "
            "the current group, and the description supplied by the rule."
        )
        outcomes = [
            "Agreement makes the description more credible for similar records.",
            "Disagreement shows where the description needs more human checks.",
        ]
    else:
        why = (
            f"This {entity_name} is one of the clearest unresolved examples in "
            "the current round. Its label can answer a question that affects other records."
        )
        how = (
            "Use the record's real-world meaning, not its plot position or group "
            "number, to choose an existing type. Create a new type only when "
            "needed, or choose uncertain when there is not enough evidence."
        )
        outcomes = [
            "A clear human label gives the next round stronger evidence for similar records.",
            "An uncertain answer keeps the record visible without forcing a weak conclusion.",
        ]
    return (
        " ".join(why.split()),
        " ".join(how.split()),
        [" ".join(outcome.split()) for outcome in outcomes],
    )


def _plain_feature_name(value: Any) -> str:
    text = str(value or "a feature used by the current rule")
    return " ".join(text.replace("_", " ").split())


def _plain_round_change(
    point_id: str,
    *,
    plan: Mapping[str, Any],
    profile: Mapping[str, Any],
    ranking: Mapping[str, Any],
    delta: Mapping[str, Any],
) -> str:
    if bool(delta.get("baseline")):
        return "This is the first round, so there is no earlier result to compare."
    recheck_reason = str(ranking.get("recheck_reason", ""))
    if recheck_reason == "cluster_changed_after_label" or profile.get("cluster_changed"):
        return (
            "After the last labels, the current analysis moved this record to a "
            "different group. It is being checked again because its meaning may "
            "not agree with that move."
        )
    if recheck_reason == "outlier_status_changed_after_label" or profile.get(
        "outlier_changed"
    ):
        return (
            "After the last labels, the current analysis changed whether it treats "
            "this record as unusual. A human check can confirm that change."
        )
    if recheck_reason == "current_rule_conflicts_with_existing_label":
        return (
            "This record already has a human label, but the current group rule "
            "does not agree with it. It is being shown again to resolve that conflict."
        )
    diff = dict(plan.get("previous_plan_diff", {}))
    if point_id in set(diff.get("added_point_ids", ())):
        return (
            "This record is new in the current batch because the latest round made "
            "it a clearer example of the remaining question."
        )
    if point_id in set(diff.get("retained_point_ids", ())):
        return (
            "This record remains useful because the previous round did not settle "
            "the question around it."
        )
    return (
        "The last round did not change this record directly, but the question it "
        "helps answer is still open."
    )


def _history_aware_plan(
    base: Mapping[str, Any],
    *,
    session: ActiveLearningSession,
    round_id: str,
    label_revision: int,
    prepared: PreparedDataset,
    analysis: AnalysisResult,
    rule_set: RuleSet,
    parent_round: ActiveLearningRound | None,
    active_events: Tuple[LabelEvent, ...],
    analysis_delta: Mapping[str, Any],
    recommendation_history_counts: Mapping[str, int],
    recommendation_history: Mapping[str, Mapping[str, int]],
) -> RecommendationPlanV2:
    category = str(base["focus_category"])
    base_candidates = tuple(base.get("candidate_pool_point_ids", ()))
    labeled = {
        event.point_id
        for event in active_events
        if event.label_dimension != "uncertain"
    }
    previous_plan = (
        parent_round.recommendation_plans.get(category, {})
        if parent_round is not None
        else {}
    )
    previous_recommended = set(previous_plan.get("recommended_point_ids", ()))
    changed_cluster = set(analysis_delta.get("changed_cluster_point_ids", ()))
    changed_outlier = set(analysis_delta.get("outlier_added_point_ids", ())) | set(
        analysis_delta.get("outlier_removed_point_ids", ())
    )
    point_profiles = _point_profiles(prepared, analysis, rule_set, base_candidates)
    profiles_by_id = {item["point_id"]: dict(item) for item in point_profiles}
    maximum_impact = max(
        (profile.get("affected_point_count", 0) for profile in point_profiles),
        default=1,
    )
    excluded = []
    candidates = []
    for base_rank, point_id in enumerate(base_candidates, start=1):
        profile = dict(profiles_by_id.get(point_id, {"point_id": point_id}))
        profile["cluster_changed"] = point_id in changed_cluster
        profile["outlier_changed"] = point_id in changed_outlier
        profiles_by_id[point_id] = profile
        is_exception = bool(profile.get("is_rule_exception"))
        recheck_reason = ""
        if point_id in labeled:
            if point_id in changed_cluster:
                recheck_reason = "cluster_changed_after_label"
            elif point_id in changed_outlier:
                recheck_reason = "outlier_status_changed_after_label"
            elif is_exception:
                recheck_reason = "current_rule_conflicts_with_existing_label"
            else:
                excluded.append(
                    {
                        "point_id": point_id,
                        "reason": "already_has_active_human_label",
                    }
                )
                continue
        category_score = 1.0 - ((base_rank - 1) / max(len(base_candidates), 1))
        impact_score = float(profile.get("affected_point_count", 0)) / max(
            float(maximum_impact), 1.0
        )
        candidates.append(
            {
                "point_id": point_id,
                "base_rank": base_rank,
                "category_score": round(category_score, 6),
                "impact_score": round(impact_score, 6),
                "recently_recommended": (
                    point_id in previous_recommended and not recheck_reason
                ),
                "recommendation_history_count": int(
                    recommendation_history_counts.get(point_id, 0)
                ),
                "already_labeled": point_id in labeled,
                "recheck_reason": recheck_reason,
                "profile": profile,
            }
        )

    preordered = sorted(
        candidates,
        key=lambda item: (
            -round(
                0.72 * float(item["category_score"])
                + 0.23 * float(item["impact_score"])
                + (0.10 if item["recheck_reason"] else 0.0)
                - (0.25 if item["recently_recommended"] else 0.0)
                - min(
                    0.20,
                    0.05
                    * int(item.get("recommendation_history_count", 0)),
                ),
                9,
            ),
            int(item["base_rank"]),
            item["point_id"],
        ),
    )
    diversity_window = max(
        session.config.candidate_pool_size * 4,
        session.config.candidate_pool_size,
    )
    ordered = _greedy_candidate_order(
        preordered[:diversity_window],
        prepared,
    )
    recommendation_limit = (
        6
        if category in {"overlap_merge_signal", "exception_relabel_review"}
        else session.config.batch_size
    )
    fresh = [item for item in ordered if not item["recently_recommended"]]
    recent = [item for item in ordered if item["recently_recommended"]]
    candidate_rows = (fresh + recent)[: session.config.candidate_pool_size]
    candidate_pool = tuple(item["point_id"] for item in candidate_rows)
    selection_pool = candidate_rows
    recommended = tuple(item["point_id"] for item in selection_pool[:recommendation_limit])
    rankings = []
    for rank, item in enumerate(candidate_rows, start=1):
        selected = item["point_id"] in set(recommended)
        rankings.append(
            {
                "rank": rank,
                "candidate_rank": rank,
                "point_id": item["point_id"],
                "category": category,
                "ranking_score": item["ranking_score"],
                "ranking_score_components": dict(item["ranking_score_components"]),
                "selection_reason": _active_selection_reason(
                    category,
                    item,
                    selected=selected,
                ),
                "selected_now": selected,
                "deferred_reason": (
                    ""
                    if selected
                    else "lower deterministic rank in the current labeling batch"
                ),
                "recheck_reason": item["recheck_reason"],
                "recommendation_history_count": item[
                    "recommendation_history_count"
                ],
            }
        )
    candidate_profiles = tuple(
        dict(profiles_by_id[point_id])
        for point_id in candidate_pool
    )
    selected_profiles = tuple(
        dict(profiles_by_id[point_id])
        for point_id in recommended
    )
    deferred_points = _deferred_rows(candidate_pool, recommended, rankings)
    previous_plan_diff = _plan_diff(
        previous_plan.get("recommended_point_ids", ()),
        recommended,
    )
    previous_plan_diff = _explain_plan_diff(
        previous_plan_diff,
        rankings=rankings,
        excluded=excluded,
        deferred=deferred_points,
    )
    stop_reason = ""
    if not base.get("has_typical_case"):
        stop_reason = "no_typical_case_for_category"
    elif not recommended:
        stop_reason = "no_eligible_unlabeled_candidates"
    plan_basis = {
        "session_id": session.session_id,
        "round_id": round_id,
        "category": category,
        "base_plan_id": base.get("plan_id"),
        "recommended": recommended,
        "candidate_pool": candidate_pool,
        "excluded": excluded,
        "label_revision": label_revision,
        "version": ACTIVE_PLAN_VERSION,
        "eligible_candidate_count": len(candidates),
    }
    plan_id = f"alplan_{_hash(plan_basis)[:14]}"
    return RecommendationPlanV2(
        plan_id=plan_id,
        plan_version=ACTIVE_PLAN_VERSION,
        session_id=session.session_id,
        round_id=round_id,
        dataset_version_id=prepared.version.dataset_version_id,
        preprocessing_version=prepared.version.preprocessing_version,
        label_revision=label_revision,
        focus_category=category,
        has_typical_case=bool(base.get("has_typical_case")) and bool(recommended),
        candidate_pool_point_ids=candidate_pool,
        recommended_point_ids=recommended,
        highlighted_point_ids=recommended,
        target_rule_ids=tuple(base.get("target_rule_ids", ())),
        candidate_rankings=tuple(rankings),
        candidate_point_profiles=candidate_profiles,
        point_profiles=selected_profiles,
        excluded_points=tuple(excluded),
        deferred_points=deferred_points,
        previous_plan_id=previous_plan.get("plan_id"),
        previous_plan_diff=previous_plan_diff,
        history_context={
            "previous_recommended_point_ids": list(previous_recommended),
            "active_labeled_point_count": len(labeled),
            "cluster_changed_point_count": len(changed_cluster),
            "outlier_changed_point_count": len(changed_outlier),
            "recommendation_history_counts": {
                point_id: int(recommendation_history_counts.get(point_id, 0))
                for point_id in candidate_pool
            },
            "recommendation_history": {
                event_kind: {
                    point_id: int(counts.get(point_id, 0))
                    for point_id in candidate_pool
                    if int(counts.get(point_id, 0)) > 0
                }
                for event_kind, counts in recommendation_history.items()
            },
            "eligible_candidate_count_before_pool_limit": len(candidates),
            "diversity_window_count": len(ordered),
        },
        label_options=tuple(_label_options(session.label_vocabulary)),
        expected_label_outcomes=tuple(base.get("expected_label_outcomes", ())),
        stop_reason=stop_reason,
        source_plan=dict(base),
    )


def _meta_priority_plan(
    plans: Mapping[str, Mapping[str, Any]],
    *,
    session: ActiveLearningSession,
    round_id: str,
    label_revision: int,
    prepared: PreparedDataset,
    parent_round: ActiveLearningRound | None,
) -> RecommendationPlanV2:
    available = [
        (category, plan)
        for category, plan in plans.items()
        if plan.get("recommended_point_ids")
    ]
    if available:
        scored_plans = [
            (
                category,
                plan,
                _meta_plan_score(
                    category,
                    plan,
                    point_count=prepared.version.point_count,
                    target_pool_size=session.config.candidate_pool_size,
                ),
            )
            for category, plan in available
        ]
        category, source, meta_score = sorted(
            scored_plans,
            key=lambda item: (
                -item[2]["total"],
                -_CATEGORY_PRIORITY.get(item[0], 0),
                item[0],
            ),
        )[0]
        recommended = tuple(source.get("recommended_point_ids", ()))[: session.config.batch_size]
        previous_plan = (
            parent_round.recommendation_plans.get("label_priority", {})
            if parent_round is not None
            else {}
        )
        meta_plan_basis = {
            "round": round_id,
            "category": "label_priority",
            "source": source.get("plan_id"),
            "score": meta_score,
        }
        plan_id = f"alplan_{_hash(meta_plan_basis)[:14]}"
        candidate_profiles_by_id = {
            item.get("point_id"): item
            for item in source.get("candidate_point_profiles", ())
        }
        rankings = tuple(
            {
                **dict(item),
                "category": "label_priority",
                "source_category": category,
                "selected_now": item.get("point_id") in set(recommended),
                "deferred_reason": (
                    ""
                    if item.get("point_id") in set(recommended)
                    else "lower deterministic rank in the current meta-ranked batch"
                ),
                "selection_reason": (
                    f"{item.get('point_id')} comes from the highest-priority unresolved "
                    f"{category.replace('_', ' ')} question. {item.get('selection_reason', '')}"
                ),
            }
            for item in source.get("candidate_rankings", ())
        )
        return RecommendationPlanV2(
            plan_id=plan_id,
            plan_version=ACTIVE_PLAN_VERSION,
            session_id=session.session_id,
            round_id=round_id,
            dataset_version_id=prepared.version.dataset_version_id,
            preprocessing_version=prepared.version.preprocessing_version,
            label_revision=label_revision,
            focus_category="label_priority",
            has_typical_case=True,
            candidate_pool_point_ids=tuple(source.get("candidate_pool_point_ids", ())),
            recommended_point_ids=recommended,
            highlighted_point_ids=recommended,
            target_rule_ids=tuple(source.get("target_rule_ids", ())),
            candidate_rankings=rankings,
            candidate_point_profiles=tuple(
                source.get("candidate_point_profiles", ())
            ),
            point_profiles=tuple(
                dict(candidate_profiles_by_id[point_id])
                for point_id in recommended
            ),
            excluded_points=tuple(source.get("excluded_points", ())),
            deferred_points=_deferred_rows(
                tuple(source.get("candidate_pool_point_ids", ())),
                recommended,
                rankings,
            ),
            previous_plan_id=previous_plan.get("plan_id"),
            previous_plan_diff=_explain_plan_diff(
                _plan_diff(
                    previous_plan.get("recommended_point_ids", ()),
                    recommended,
                ),
                rankings=rankings,
                excluded=source.get("excluded_points", ()),
                deferred=_deferred_rows(
                    tuple(source.get("candidate_pool_point_ids", ())),
                    recommended,
                    rankings,
                ),
            ),
            history_context={
                **dict(source.get("history_context", {})),
                "delegated_category": category,
                "meta_score_components": meta_score,
                "category_scores": {
                    item_category: item_score
                    for item_category, _, item_score in scored_plans
                },
            },
            label_options=tuple(source.get("label_options", ())),
            expected_label_outcomes=tuple(source.get("expected_label_outcomes", ())),
            source_plan=dict(source),
        )
    plan_id = f"alplan_{_hash({'round': round_id, 'category': 'label_priority', 'empty': True})[:14]}"
    return RecommendationPlanV2(
        plan_id=plan_id,
        plan_version=ACTIVE_PLAN_VERSION,
        session_id=session.session_id,
        round_id=round_id,
        dataset_version_id=prepared.version.dataset_version_id,
        preprocessing_version=prepared.version.preprocessing_version,
        label_revision=label_revision,
        focus_category="label_priority",
        has_typical_case=False,
        candidate_pool_point_ids=(),
        recommended_point_ids=(),
        highlighted_point_ids=(),
        target_rule_ids=(),
        candidate_rankings=(),
        candidate_point_profiles=(),
        point_profiles=(),
        excluded_points=(),
        deferred_points=(),
        previous_plan_id=(
            parent_round.recommendation_plans.get("label_priority", {}).get("plan_id")
            if parent_round is not None
            else None
        ),
        previous_plan_diff=_explain_plan_diff(
            _plan_diff(
                (
                    parent_round.recommendation_plans.get("label_priority", {}).get(
                        "recommended_point_ids",
                        (),
                    )
                    if parent_round is not None
                    else ()
                ),
                (),
            ),
            rankings=(),
            excluded=(),
            deferred=(),
            default_removed_reason="no eligible category remains",
        ),
        history_context={},
        label_options=tuple(_label_options(session.label_vocabulary)),
        expected_label_outcomes=(),
        stop_reason="no_eligible_candidates_across_categories",
    )


def _deferred_rows(
    candidate_pool: Sequence[str],
    recommended: Sequence[str],
    rankings: Sequence[Mapping[str, Any]],
) -> Tuple[Mapping[str, Any], ...]:
    recommended_set = set(recommended)
    ranking_by_id = {item.get("point_id"): item for item in rankings}
    return tuple(
        {
            "point_id": point_id,
            "candidate_rank": ranking_by_id.get(point_id, {}).get("candidate_rank"),
            "ranking_score": ranking_by_id.get(point_id, {}).get("ranking_score"),
            "reason": ranking_by_id.get(point_id, {}).get(
                "deferred_reason",
                "not selected in the current deterministic batch",
            ),
        }
        for point_id in candidate_pool
        if point_id not in recommended_set
    )


def _plan_diff(
    previous_ids: Sequence[str],
    current_ids: Sequence[str],
) -> Dict[str, Any]:
    previous = tuple(previous_ids)
    current = tuple(current_ids)
    previous_set = set(previous)
    current_set = set(current)
    return {
        "added_point_ids": [
            point_id for point_id in current if point_id not in previous_set
        ],
        "removed_point_ids": [
            point_id for point_id in previous if point_id not in current_set
        ],
        "retained_point_ids": [
            point_id for point_id in current if point_id in previous_set
        ],
        "order_changed": previous != current,
    }


def _explain_plan_diff(
    diff: Mapping[str, Any],
    *,
    rankings: Sequence[Mapping[str, Any]],
    excluded: Sequence[Mapping[str, Any]],
    deferred: Sequence[Mapping[str, Any]],
    default_removed_reason: str = "no longer selected by current category evidence",
) -> Dict[str, Any]:
    result = dict(diff)
    ranking_by_id = {item.get("point_id"): item for item in rankings}
    excluded_by_id = {item.get("point_id"): item for item in excluded}
    deferred_by_id = {item.get("point_id"): item for item in deferred}
    result["added_reasons"] = {
        point_id: ranking_by_id.get(point_id, {}).get(
            "selection_reason",
            "newly selected by the current deterministic ranking",
        )
        for point_id in result.get("added_point_ids", ())
    }
    result["removed_reasons"] = {}
    for point_id in result.get("removed_point_ids", ()):
        if point_id in excluded_by_id:
            reason = excluded_by_id[point_id].get("reason")
        elif point_id in deferred_by_id:
            reason = deferred_by_id[point_id].get("reason")
        else:
            reason = default_removed_reason
        result["removed_reasons"][point_id] = reason
    return result


def _meta_plan_score(
    category: str,
    plan: Mapping[str, Any],
    *,
    point_count: int,
    target_pool_size: int,
) -> Dict[str, float]:
    selected = [
        item
        for item in plan.get("candidate_rankings", ())
        if item.get("selected_now")
    ]
    conflicts = sum(1 for item in selected if item.get("recheck_reason"))
    unresolved_conflicts = min(conflicts / max(len(selected), 1), 1.0)
    affected = max(
        (
            int(item.get("affected_point_count", 0))
            for item in plan.get("candidate_point_profiles", ())
        ),
        default=0,
    )
    affected_region = min(affected / max(point_count, 1), 1.0)
    candidate_count = int(plan.get("candidate_pool_count", 0))
    candidate_availability = min(
        candidate_count / max(target_pool_size, 1),
        1.0,
    )
    history_counts = plan.get("history_context", {}).get(
        "recommendation_history_counts",
        {},
    )
    previously_seen = sum(
        1
        for point_id in plan.get("candidate_pool_point_ids", ())
        if int(history_counts.get(point_id, 0)) > 0
    )
    history_coverage = (
        1.0 - (previously_seen / candidate_count)
        if candidate_count
        else 0.0
    )
    total = (
        0.38 * unresolved_conflicts
        + 0.32 * affected_region
        + 0.20 * candidate_availability
        + 0.10 * history_coverage
    )
    return {
        "total": round(total, 6),
        "unresolved_conflicts": round(unresolved_conflicts, 6),
        "affected_region": round(affected_region, 6),
        "candidate_availability": round(candidate_availability, 6),
        "history_coverage": round(history_coverage, 6),
        "category_tie_break_priority": float(_CATEGORY_PRIORITY.get(category, 0)),
    }


def _recommendation_history_summary(
    rounds: Sequence[ActiveLearningRound],
    *,
    shown_events: Sequence[Mapping[str, Any]],
    label_events: Sequence[LabelEvent],
) -> Dict[str, Dict[str, int]]:
    computed: Dict[str, int] = {}
    for round_state in rounds:
        seen_this_round = {
            point_id
            for plan in round_state.recommendation_plans.values()
            if isinstance(plan, Mapping)
            for point_id in plan.get("recommended_point_ids", ())
        }
        for point_id in seen_this_round:
            computed[point_id] = computed.get(point_id, 0) + 1

    shown: Dict[str, int] = {}
    shown_by_round: Dict[str, set[str]] = {}
    for event in shown_events:
        shown_by_round.setdefault(str(event.get("round_id")), set()).add(
            str(event.get("point_id"))
        )
    for point_ids in shown_by_round.values():
        for point_id in point_ids:
            shown[point_id] = shown.get(point_id, 0) + 1

    selected: Dict[str, int] = {}
    labeled: Dict[str, int] = {}
    selected_by_round: Dict[str, set[str]] = {}
    labeled_by_round: Dict[str, set[str]] = {}
    for event in label_events:
        selected_by_round.setdefault(event.round_id, set()).add(event.point_id)
        if event.label_dimension != "uncertain":
            labeled_by_round.setdefault(event.round_id, set()).add(event.point_id)
    for point_ids in selected_by_round.values():
        for point_id in point_ids:
            selected[point_id] = selected.get(point_id, 0) + 1
    for point_ids in labeled_by_round.values():
        for point_id in point_ids:
            labeled[point_id] = labeled.get(point_id, 0) + 1
    return {
        "computed": computed,
        "shown": shown,
        "selected": selected,
        "labeled": labeled,
    }


def _annotate_cross_category_coverage(
    plans: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    coverage: Dict[str, list[str]] = {}
    for category, plan in plans.items():
        for point_id in plan.get("recommended_point_ids", ()):
            coverage.setdefault(point_id, []).append(category)

    annotated: Dict[str, Mapping[str, Any]] = {}
    for category, plan in plans.items():
        payload = dict(plan)
        payload["candidate_rankings"] = [
            {
                **dict(item),
                "covered_categories": list(
                    coverage.get(item.get("point_id"), (category,))
                ),
            }
            for item in plan.get("candidate_rankings", ())
        ]
        for profile_field in ("candidate_point_profiles", "point_profiles"):
            payload[profile_field] = [
                {
                    **dict(item),
                    "covered_categories": list(
                        coverage.get(item.get("point_id"), (category,))
                    ),
                }
                for item in plan.get(profile_field, ())
            ]
        annotated[category] = payload
    return annotated


def _stopped_recommendation_plans(
    *,
    session: ActiveLearningSession,
    round_id: str,
    label_revision: int,
    prepared: PreparedDataset,
    parent_round: ActiveLearningRound | None,
    stop_reason: str,
) -> Dict[str, Mapping[str, Any]]:
    plans = {}
    for category in RECOMMENDATION_CATEGORIES:
        previous = (
            parent_round.recommendation_plans.get(category, {})
            if parent_round is not None
            else {}
        )
        plan_basis = {
            'round': round_id,
            'category': category,
            'stop_reason': stop_reason,
            'version': ACTIVE_PLAN_VERSION,
        }
        plan_id = f"alplan_{_hash(plan_basis)[:14]}"
        plans[category] = RecommendationPlanV2(
            plan_id=plan_id,
            plan_version=ACTIVE_PLAN_VERSION,
            session_id=session.session_id,
            round_id=round_id,
            dataset_version_id=prepared.version.dataset_version_id,
            preprocessing_version=prepared.version.preprocessing_version,
            label_revision=label_revision,
            focus_category=category,
            has_typical_case=False,
            candidate_pool_point_ids=(),
            recommended_point_ids=(),
            highlighted_point_ids=(),
            target_rule_ids=(),
            candidate_rankings=(),
            candidate_point_profiles=(),
            point_profiles=(),
            excluded_points=(),
            deferred_points=(),
            previous_plan_id=previous.get("plan_id"),
            previous_plan_diff=_explain_plan_diff(
                _plan_diff(
                    previous.get("recommended_point_ids", ()),
                    (),
                ),
                rankings=(),
                excluded=(),
                deferred=(),
                default_removed_reason=stop_reason.replace("_", " "),
            ),
            history_context={"automatic_stop": True},
            label_options=tuple(_label_options(session.label_vocabulary)),
            expected_label_outcomes=(),
            stop_reason=stop_reason,
        ).to_dict()
    return plans


def _round_lifecycle(
    plans: Mapping[str, Mapping[str, Any]],
    delta: Mapping[str, Any],
) -> tuple[str, Dict[str, Any]]:
    stop_reason = str(plans["label_priority"].get("stop_reason", ""))
    if stop_reason in {
        "label_budget_reached",
        "no_eligible_candidates_across_categories",
    }:
        message = {
            "label_budget_reached": (
                "The label budget is complete. Review the round history before "
                "starting a larger-budget session."
            ),
            "no_eligible_candidates_across_categories": (
                "No eligible unlabeled or evidence-backed recheck candidates remain."
            ),
        }[stop_reason]
        return "stopped", {
            "should_stop": True,
            "automatic": True,
            "reason": stop_reason,
            "message": message,
        }
    stable = (
        not delta.get("baseline")
        and not delta.get("changed_cluster_point_ids")
        and not delta.get("outlier_added_point_ids")
        and not delta.get("outlier_removed_point_ids")
        and not delta.get("rule_ids_added")
        and not delta.get("rule_ids_removed")
    )
    if stable:
        return "ready_for_labeling", {
            "should_stop": False,
            "automatic": False,
            "reason": "analysis_stable",
            "message": (
                "The latest labels did not change groups, outliers, or rules. "
                "Consider stopping after reviewing the remaining recommendation."
            ),
        }
    return "ready_for_labeling", {
        "should_stop": False,
        "automatic": False,
        "reason": "",
        "message": "",
    }


def _point_profiles(
    prepared: PreparedDataset,
    analysis: AnalysisResult,
    rule_set: RuleSet,
    point_ids: Sequence[str],
) -> Tuple[Mapping[str, Any], ...]:
    index_by_id = {
        point_id: index for index, point_id in enumerate(prepared.feature_matrix.point_ids)
    }
    raw_by_id = {item["point_id"]: item for item in prepared.raw_records}
    assignments = {
        item.point_id: item.cluster_id for item in analysis.cluster_result.assignments
    }
    outliers = {
        item.point_id: item for item in analysis.outlier_result.scores
    }
    ordered_outlier_scores = sorted(
        float(item.score) for item in analysis.outlier_result.scores
    )
    profiles = []
    for point_id in point_ids:
        index = index_by_id.get(point_id)
        if index is None:
            continue
        related = [
            rule
            for rule in rule_set.rules
            if point_id in rule.matched_point_ids or point_id in rule.exception_point_ids
        ]
        margins = []
        row = prepared.feature_matrix.values[index]
        model_index = {
            name: feature_index
            for feature_index, name in enumerate(prepared.feature_matrix.feature_names)
        }
        for rule in related:
            for condition in rule.conditions:
                feature_index = model_index.get(condition.feature)
                if feature_index is None:
                    continue
                value = float(row[feature_index])
                margins.append(
                    {
                        "rule_id": rule.rule_id,
                        "model_feature": condition.feature,
                        "operator": condition.operator,
                        "threshold": condition.threshold,
                        "point_value": value,
                        "absolute_margin": abs(value - condition.threshold),
                        **display_condition(
                            condition.feature,
                            condition.operator,
                            condition.threshold,
                            prepared.version.transformation_map,
                        ),
                    }
                )
        ordered_margins = sorted(
            margins,
            key=lambda item: (
                item["absolute_margin"],
                item["rule_id"],
                item["model_feature"],
            ),
        )[:8]
        outlier = outliers.get(point_id)
        affected_point_count = len(
            {
                related_point
                for rule in related
                for related_point in rule.matched_point_ids
            }
        )
        profiles.append(
            {
                "point_id": point_id,
                "current_cluster_id": assignments.get(point_id),
                "outlier_score": outlier.score if outlier else None,
                "is_outlier": outlier.is_outlier if outlier else False,
                "raw_features": dict(raw_by_id[point_id].get("raw_features", {})),
                "metadata": dict(raw_by_id[point_id].get("metadata", {})),
                "related_rule_ids": [rule.rule_id for rule in related],
                "is_rule_exception": any(
                    point_id in rule.exception_point_ids for rule in related
                ),
                "affected_point_count": affected_point_count,
                "threshold_margins": ordered_margins,
                "plain_language_evidence": _plain_language_evidence(
                    closest_margin=ordered_margins[0] if ordered_margins else None,
                    outlier_score=float(outlier.score) if outlier else None,
                    ordered_outlier_scores=ordered_outlier_scores,
                    affected_point_count=affected_point_count,
                    point_count=prepared.version.point_count,
                ),
            }
        )
    return tuple(profiles)


def _plain_language_evidence(
    *,
    closest_margin: Mapping[str, Any] | None,
    outlier_score: float | None,
    ordered_outlier_scores: Sequence[float],
    affected_point_count: int,
    point_count: int,
) -> Dict[str, Any]:
    if affected_point_count >= max(10, int(round(point_count * 0.25))):
        affected_scope = "many nearby records"
    elif affected_point_count >= max(4, int(round(point_count * 0.08))):
        affected_scope = "several nearby records"
    else:
        affected_scope = "a few nearby records"

    unusual_level = "low"
    if outlier_score is not None and ordered_outlier_scores:
        rank = sum(value <= outlier_score for value in ordered_outlier_scores)
        percentile = rank / len(ordered_outlier_scores)
        if percentile >= 0.8:
            unusual_level = "high"
        elif percentile >= 0.4:
            unusual_level = "moderate"

    boundary: Dict[str, Any] = {}
    if closest_margin:
        margin = float(closest_margin.get("absolute_margin", 0.0))
        if margin <= 0.15:
            proximity = "very close"
        elif margin <= 0.5:
            proximity = "close"
        else:
            proximity = "not especially close"
        value = float(closest_margin.get("point_value", 0.0))
        threshold = float(closest_margin.get("threshold", 0.0))
        side = "above" if value > threshold else "below"
        if margin <= 0.15:
            relation = f"very close to and slightly {side}"
        elif margin <= 0.5:
            relation = f"just {side}"
        else:
            relation = side
        boundary = {
            "feature_name": closest_margin.get("source_feature")
            or closest_margin.get("model_feature"),
            "proximity": proximity,
            "relation": relation,
            "rule_id": closest_margin.get("rule_id"),
            "condition": closest_margin.get("display_text"),
        }
    return {
        "closest_boundary": boundary,
        "affected_scope": affected_scope,
        "unusual_level": unusual_level,
    }


def _greedy_candidate_order(
    candidates: Sequence[Mapping[str, Any]],
    prepared: PreparedDataset,
) -> list[Dict[str, Any]]:
    remaining = [dict(item) for item in candidates]
    selected = []
    vector_by_id = {
        point_id: np.asarray(prepared.feature_matrix.values[index], dtype=float)
        for index, point_id in enumerate(prepared.feature_matrix.point_ids)
    }
    while remaining:
        scored = []
        for item in remaining:
            point_id = item["point_id"]
            if selected:
                distances = [
                    float(np.linalg.norm(vector_by_id[point_id] - vector_by_id[chosen["point_id"]]))
                    for chosen in selected
                ]
                diversity = min(distances)
                diversity = diversity / (1.0 + diversity)
            else:
                diversity = 1.0
            recent_penalty = 0.25 if item["recently_recommended"] else 0.0
            history_penalty = min(
                0.20,
                0.05 * int(item.get("recommendation_history_count", 0)),
            )
            recheck_bonus = 0.1 if item["recheck_reason"] else 0.0
            score = (
                0.65 * float(item["category_score"])
                + 0.20 * float(item["impact_score"])
                + 0.15 * diversity
                + recheck_bonus
                - recent_penalty
                - history_penalty
            )
            scored.append(
                (
                    -round(score, 9),
                    int(item["base_rank"]),
                    point_id,
                    item,
                    diversity,
                    score,
                )
            )
        _, _, _, chosen, diversity, score = sorted(scored)[0]
        chosen = dict(chosen)
        chosen_history_penalty = min(
            0.20,
            0.05 * int(chosen.get("recommendation_history_count", 0)),
        )
        chosen["ranking_score"] = round(score, 6)
        chosen["ranking_score_components"] = {
            "category_evidence": chosen["category_score"],
            "affected_region": chosen["impact_score"],
            "batch_diversity": round(diversity, 6),
                "recent_recommendation_penalty": (
                    0.25 if chosen["recently_recommended"] else 0.0
                ),
                "history_recommendation_penalty": round(
                    chosen_history_penalty,
                    6,
                ),
                "recheck_bonus": 0.1 if chosen["recheck_reason"] else 0.0,
        }
        selected.append(chosen)
        remaining = [item for item in remaining if item["point_id"] != chosen["point_id"]]
    return selected


def _active_selection_reason(
    category: str,
    item: Mapping[str, Any],
    *,
    selected: bool,
) -> str:
    question = {
        "boundary_review": "checking the current dividing line",
        "overlap_merge_signal": "checking an area described by more than one group",
        "split_or_new_cluster_signal": "checking a separated group area",
        "anomaly_label_review": "checking a record that looks unusual",
        "exception_relabel_review": "checking a record that does not fit its group rule",
        "feature_label_strategy": "checking whether a feature rule matches human meaning",
        "rule_confidence_audit": (
            "checking whether the current group description agrees with "
            "human judgment"
        ),
    }.get(category, "answering the current labeling question")
    parts = [f"This record is useful for {question}"]
    if item["recently_recommended"]:
        parts.append("the previous round did not fully settle the question")
    if item["recheck_reason"]:
        parts.append("new evidence makes it worth checking again")
    parts.append(
        "it adds a useful example to this batch"
        if selected
        else "it remains available for a later batch"
    )
    return "; ".join(parts) + "."


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
        item["rule_id"]: item
        for item in parent_round.rule_set.get("rules", ())
    }
    current_rule_payloads = {
        rule.rule_id: rule.to_dict() for rule in rule_set.rules
    }
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
            changed_rules=bool(
                previous_rules != current_rules or changed_rule_ids
            ),
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


def _finish_delta(
    delta: Mapping[str, Any],
    parent_round: ActiveLearningRound | None,
    plans: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    result = dict(delta)
    recommendation_changes = {}
    for category, plan in plans.items():
        plan_diff = dict(plan.get("previous_plan_diff", {}))
        recommendation_changes[category] = {
            "added_point_ids": list(plan_diff.get("added_point_ids", ())),
            "removed_point_ids": list(plan_diff.get("removed_point_ids", ())),
            "unchanged_point_ids": list(plan_diff.get("retained_point_ids", ())),
            "added_reasons": dict(plan_diff.get("added_reasons", {})),
            "removed_reasons": dict(plan_diff.get("removed_reasons", {})),
            "order_changed": bool(plan_diff.get("order_changed")),
        }
    result["recommendation_changes"] = recommendation_changes
    result["remaining_categories"] = [
        category for category, plan in plans.items() if plan.get("recommended_point_ids")
    ]
    result["resolved_categories"] = [
        category for category, plan in plans.items() if not plan.get("recommended_point_ids")
    ]
    return result


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
        "raw_feature_names": [
            item.name for item in prepared.version.feature_specs
        ],
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


def _label_options(vocabulary: Mapping[str, str]) -> list[str]:
    return [
        *[f"semantic:{label_id}:{name}" for label_id, name in sorted(vocabulary.items())],
        "semantic:new",
        "outlier:true",
        "outlier:false",
        "uncertain",
    ]


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
    base = "".join(
        char.lower() if char.isalnum() else "_"
        for char in display_name
    ).strip("_") or "class"
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
        "exception_point_ids": sorted(
            payload.get("exception_point_ids", ())
        ),
    }
    return _hash(canonical)[:16]


def _hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
