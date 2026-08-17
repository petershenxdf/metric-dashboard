from __future__ import annotations

import gzip
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np

from .data import PreparedDataset, reconstruct_prepared_dataset
from .schemas import (
    ActiveLearningRound,
    ActiveLearningSession,
    DatasetVersion,
    LabelEvent,
    SessionConfig,
)


class ActiveLearningStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root = self.db_path.parent / "artifacts"
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    dataset_version_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS active_learning_sessions (
                    session_id TEXT PRIMARY KEY,
                    dataset_version_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    label_vocabulary_json TEXT NOT NULL,
                    current_round_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(dataset_version_id)
                        REFERENCES dataset_versions(dataset_version_id)
                );

                CREATE TABLE IF NOT EXISTS active_learning_rounds (
                    round_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    round_index INTEGER NOT NULL,
                    parent_round_id TEXT,
                    label_revision INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id)
                        REFERENCES active_learning_sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS active_learning_label_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    round_id TEXT NOT NULL,
                    point_id TEXT NOT NULL,
                    label_dimension TEXT NOT NULL,
                    label_value_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    supersedes_event_id TEXT,
                    provenance_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id)
                        REFERENCES active_learning_sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_active_labels
                    ON active_learning_label_events(session_id, point_id, label_dimension, status);

                CREATE TABLE IF NOT EXISTS active_learning_interpretations (
                    session_id TEXT NOT NULL,
                    round_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    provider_kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    diagnostics_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(round_id, plan_id, provider_kind)
                );

                CREATE TABLE IF NOT EXISTS active_learning_recommendation_events (
                    session_id TEXT NOT NULL,
                    round_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    point_id TEXT NOT NULL,
                    event_kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(round_id, plan_id, point_id, event_kind),
                    FOREIGN KEY(session_id)
                        REFERENCES active_learning_sessions(session_id)
                );
                """
            )

    def save_prepared_dataset(self, prepared: PreparedDataset) -> PreparedDataset:
        target = self.artifact_root / "datasets" / prepared.version.dataset_version_id
        target.mkdir(parents=True, exist_ok=True)
        raw_path = target / "raw_records.json.gz"
        matrix_path = target / "feature_matrix.npz"
        with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
            json.dump(
                {"records": [dict(item) for item in prepared.raw_records]},
                handle,
                sort_keys=True,
                ensure_ascii=True,
            )
        np.savez_compressed(
            matrix_path,
            values=np.asarray(prepared.feature_matrix.values, dtype=float),
        )
        stored = prepared.with_artifacts(str(raw_path), str(matrix_path))
        payload = stored.version.to_dict()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO dataset_versions (
                    dataset_version_id, dataset_id, fingerprint, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(dataset_version_id) DO UPDATE SET
                    payload_json=excluded.payload_json
                """,
                (
                    stored.version.dataset_version_id,
                    stored.version.dataset_id,
                    stored.version.fingerprint,
                    _dump(payload),
                    stored.version.created_at,
                ),
            )
        return stored

    def get_dataset_version(self, dataset_version_id: str) -> DatasetVersion:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM dataset_versions WHERE dataset_version_id = ?",
                (dataset_version_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"unknown dataset_version_id: {dataset_version_id}")
        return DatasetVersion.from_dict(_load(row["payload_json"]))

    def load_prepared_dataset(self, dataset_version_id: str) -> PreparedDataset:
        version = self.get_dataset_version(dataset_version_id)
        with gzip.open(version.raw_artifact_path, "rt", encoding="utf-8") as handle:
            raw_records = json.load(handle)["records"]
        with np.load(version.matrix_artifact_path) as payload:
            values = payload["values"].tolist()
        return reconstruct_prepared_dataset(version, raw_records, values)

    def list_dataset_versions(self) -> Tuple[DatasetVersion, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM dataset_versions ORDER BY created_at DESC"
            ).fetchall()
        return tuple(DatasetVersion.from_dict(_load(row["payload_json"])) for row in rows)

    def save_session(self, session: ActiveLearningSession) -> ActiveLearningSession:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO active_learning_sessions (
                    session_id, dataset_version_id, status, config_json,
                    label_vocabulary_json, current_round_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    status=excluded.status,
                    config_json=excluded.config_json,
                    label_vocabulary_json=excluded.label_vocabulary_json,
                    current_round_id=excluded.current_round_id,
                    updated_at=excluded.updated_at
                """,
                (
                    session.session_id,
                    session.dataset_version_id,
                    session.status,
                    _dump(session.config.to_dict()),
                    _dump(dict(session.label_vocabulary)),
                    session.current_round_id,
                    session.created_at,
                    session.updated_at,
                ),
            )
        return session

    def get_session(self, session_id: str) -> ActiveLearningSession:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM active_learning_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"unknown active-learning session: {session_id}")
        return _session_from_row(row)

    def list_sessions(self) -> Tuple[ActiveLearningSession, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM active_learning_sessions ORDER BY updated_at DESC"
            ).fetchall()
        return tuple(_session_from_row(row) for row in rows)

    def save_round(self, round_state: ActiveLearningRound) -> ActiveLearningRound:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO active_learning_rounds (
                    round_id, session_id, round_index, parent_round_id,
                    label_revision, status, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(round_id) DO UPDATE SET
                    status=excluded.status,
                    payload_json=excluded.payload_json
                """,
                (
                    round_state.round_id,
                    round_state.session_id,
                    round_state.round_index,
                    round_state.parent_round_id,
                    round_state.label_revision,
                    round_state.status,
                    _dump(round_state.to_dict()),
                    round_state.created_at,
                ),
            )
        return round_state

    def get_round(self, round_id: str) -> ActiveLearningRound:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM active_learning_rounds WHERE round_id = ?",
                (round_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"unknown active-learning round: {round_id}")
        return _round_from_payload(_load(row["payload_json"]))

    def list_rounds(self, session_id: str) -> Tuple[ActiveLearningRound, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM active_learning_rounds
                WHERE session_id = ?
                ORDER BY round_index, created_at
                """,
                (session_id,),
            ).fetchall()
        return tuple(_round_from_payload(_load(row["payload_json"])) for row in rows)

    def update_session_current(
        self,
        session: ActiveLearningSession,
        *,
        current_round_id: str,
        status: str,
        updated_at: str,
        label_vocabulary: Mapping[str, str] | None = None,
    ) -> ActiveLearningSession:
        updated = replace(
            session,
            current_round_id=current_round_id,
            status=status,
            updated_at=updated_at,
            label_vocabulary=dict(label_vocabulary or session.label_vocabulary),
        )
        return self.save_session(updated)

    def commit_round_transition(
        self,
        session: ActiveLearningSession,
        current_round: ActiveLearningRound,
        events: Iterable[LabelEvent],
        next_round: ActiveLearningRound,
        *,
        updated_at: str,
        label_vocabulary: Mapping[str, str],
    ) -> Tuple[Tuple[LabelEvent, ...], ActiveLearningSession]:
        """Commit the complete label-to-next-round transition atomically."""

        materialized = tuple(events)
        next_session_status = (
            "stopped" if next_round.status == "stopped" else "active"
        )
        updated_session = replace(
            session,
            current_round_id=next_round.round_id,
            status=next_session_status,
            updated_at=updated_at,
            label_vocabulary=dict(label_vocabulary),
        )
        committed_current = replace(current_round, status="labels_committed")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                """
                SELECT current_round_id, status
                FROM active_learning_sessions
                WHERE session_id = ?
                """,
                (session.session_id,),
            ).fetchone()
            if current is None or current["current_round_id"] != current_round.round_id:
                raise ValueError(
                    "active-learning session changed before the round transition"
                )
            if current["status"] != "active":
                raise ValueError(
                    "active-learning session is not ready for a round transition"
                )

            committed_events = []
            for event in materialized:
                previous = connection.execute(
                    """
                    SELECT event_id
                    FROM active_learning_label_events
                    WHERE session_id = ? AND point_id = ? AND label_dimension = ?
                      AND status = 'active'
                    ORDER BY created_at DESC, event_id DESC
                    LIMIT 1
                    """,
                    (event.session_id, event.point_id, event.label_dimension),
                ).fetchone()
                supersedes = previous["event_id"] if previous is not None else None
                if supersedes:
                    connection.execute(
                        """
                        UPDATE active_learning_label_events
                        SET status = 'superseded'
                        WHERE event_id = ?
                        """,
                        (supersedes,),
                    )
                stored_event = replace(
                    event,
                    supersedes_event_id=supersedes,
                    provenance={
                        **dict(event.provenance),
                        "result_round_id": next_round.round_id,
                    },
                )
                connection.execute(
                    """
                    INSERT INTO active_learning_label_events (
                        event_id, session_id, round_id, point_id, label_dimension,
                        label_value_json, status, supersedes_event_id,
                        provenance_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stored_event.event_id,
                        stored_event.session_id,
                        stored_event.round_id,
                        stored_event.point_id,
                        stored_event.label_dimension,
                        _dump(stored_event.label_value),
                        stored_event.status,
                        stored_event.supersedes_event_id,
                        _dump(dict(stored_event.provenance)),
                        stored_event.created_at,
                    ),
                )
                committed_events.append(stored_event)

            connection.execute(
                """
                UPDATE active_learning_rounds
                SET status = ?, payload_json = ?
                WHERE round_id = ? AND session_id = ?
                """,
                (
                    committed_current.status,
                    _dump(committed_current.to_dict()),
                    committed_current.round_id,
                    committed_current.session_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO active_learning_rounds (
                    round_id, session_id, round_index, parent_round_id,
                    label_revision, status, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(round_id) DO UPDATE SET
                    status=excluded.status,
                    payload_json=excluded.payload_json
                """,
                (
                    next_round.round_id,
                    next_round.session_id,
                    next_round.round_index,
                    next_round.parent_round_id,
                    next_round.label_revision,
                    next_round.status,
                    _dump(next_round.to_dict()),
                    next_round.created_at,
                ),
            )
            connection.execute(
                """
                UPDATE active_learning_sessions
                SET status = ?, label_vocabulary_json = ?,
                    current_round_id = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    updated_session.status,
                    _dump(dict(updated_session.label_vocabulary)),
                    updated_session.current_round_id,
                    updated_session.updated_at,
                    updated_session.session_id,
                ),
            )
        return tuple(committed_events), updated_session

    def active_label_events(self, session_id: str) -> Tuple[LabelEvent, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM active_learning_label_events
                WHERE session_id = ? AND status = 'active'
                ORDER BY created_at, event_id
                """,
                (session_id,),
            ).fetchall()
        return tuple(_label_event_from_row(row) for row in rows)

    def all_label_events(self, session_id: str) -> Tuple[LabelEvent, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM active_learning_label_events
                WHERE session_id = ?
                ORDER BY created_at, event_id
                """,
                (session_id,),
            ).fetchall()
        return tuple(_label_event_from_row(row) for row in rows)

    def round_ancestry(
        self,
        session_id: str,
        round_id: str,
        *,
        include_target: bool = True,
    ) -> Tuple[ActiveLearningRound, ...]:
        rounds = {
            item.round_id: item for item in self.list_rounds(session_id)
        }
        if round_id not in rounds:
            raise ValueError("round does not belong to this session")
        ancestry = []
        cursor = rounds[round_id] if include_target else (
            rounds.get(rounds[round_id].parent_round_id)
            if rounds[round_id].parent_round_id
            else None
        )
        seen = set()
        while cursor is not None:
            if cursor.round_id in seen:
                raise ValueError("active-learning round ancestry contains a cycle")
            seen.add(cursor.round_id)
            ancestry.append(cursor)
            cursor = (
                rounds.get(cursor.parent_round_id)
                if cursor.parent_round_id
                else None
            )
        return tuple(reversed(ancestry))

    def revert_events_to_round(self, session_id: str, round_id: str) -> None:
        ancestry = self.round_ancestry(
            session_id,
            round_id,
            include_target=True,
        )
        allowed_result_round_ids = {
            item.round_id
            for item in ancestry
        }
        legacy_source_round_ids = {
            item.round_id for item in ancestry[:-1]
        }
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT event_id, round_id, provenance_json
                FROM active_learning_label_events
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchall()
            for row in rows:
                provenance = _load(row["provenance_json"])
                result_round_id = provenance.get("result_round_id")
                belongs_to_ancestry = (
                    result_round_id in allowed_result_round_ids
                    if result_round_id
                    else row["round_id"] in legacy_source_round_ids
                )
                connection.execute(
                    """
                    UPDATE active_learning_label_events
                    SET status = ?
                    WHERE event_id = ?
                    """,
                    (
                        (
                            "superseded"
                            if belongs_to_ancestry
                            else "retracted"
                        ),
                        row["event_id"],
                    ),
                )
            dimensions = connection.execute(
                """
                SELECT DISTINCT point_id, label_dimension
                FROM active_learning_label_events
                WHERE session_id = ? AND status = 'superseded'
                """,
                (session_id,),
            ).fetchall()
            for dimension in dimensions:
                candidates = connection.execute(
                    """
                    SELECT event_id
                    FROM active_learning_label_events
                    WHERE session_id = ? AND point_id = ? AND label_dimension = ?
                      AND status = 'superseded'
                    ORDER BY created_at DESC, event_id DESC
                    """,
                    (
                        session_id,
                        dimension["point_id"],
                        dimension["label_dimension"],
                    ),
                ).fetchall()
                for index, candidate in enumerate(candidates):
                    connection.execute(
                        """
                        UPDATE active_learning_label_events
                        SET status = ?
                        WHERE event_id = ?
                        """,
                        ("active" if index == 0 else "superseded", candidate["event_id"]),
                    )

    def record_recommendation_shown(
        self,
        *,
        session_id: str,
        round_id: str,
        plan_id: str,
        point_ids: Iterable[str],
        created_at: str,
    ) -> None:
        materialized = tuple(dict.fromkeys(str(item) for item in point_ids))
        if not materialized:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO active_learning_recommendation_events (
                    session_id, round_id, plan_id, point_id,
                    event_kind, created_at
                ) VALUES (?, ?, ?, ?, 'shown', ?)
                """,
                [
                    (
                        session_id,
                        round_id,
                        plan_id,
                        point_id,
                        created_at,
                    )
                    for point_id in materialized
                ],
            )

    def recommendation_events(
        self,
        session_id: str,
        round_ids: Iterable[str],
    ) -> Tuple[Mapping[str, Any], ...]:
        materialized = tuple(dict.fromkeys(str(item) for item in round_ids))
        if not materialized:
            return ()
        placeholders = ",".join("?" for _ in materialized)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT round_id, plan_id, point_id, event_kind, created_at
                FROM active_learning_recommendation_events
                WHERE session_id = ? AND round_id IN ({placeholders})
                ORDER BY created_at, point_id
                """,
                (session_id, *materialized),
            ).fetchall()
        return tuple(dict(row) for row in rows)

    def save_interpretation(
        self,
        *,
        session_id: str,
        round_id: str,
        plan_id: str,
        provider_kind: str,
        payload: Mapping[str, Any],
        diagnostics: Mapping[str, Any],
        updated_at: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO active_learning_interpretations (
                    session_id, round_id, plan_id, provider_kind,
                    payload_json, diagnostics_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(round_id, plan_id, provider_kind) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    diagnostics_json=excluded.diagnostics_json,
                    updated_at=excluded.updated_at
                """,
                (
                    session_id,
                    round_id,
                    plan_id,
                    provider_kind,
                    _dump(dict(payload)),
                    _dump(dict(diagnostics)),
                    updated_at,
                ),
            )

    def get_interpretation(
        self,
        round_id: str,
        plan_id: str,
        provider_kind: str,
    ) -> Mapping[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json, diagnostics_json
                FROM active_learning_interpretations
                WHERE round_id = ? AND plan_id = ? AND provider_kind = ?
                """,
                (round_id, plan_id, provider_kind),
            ).fetchone()
        if row is None:
            return None
        return {
            "guidance": _load(row["payload_json"]),
            "diagnostics": _load(row["diagnostics_json"]),
            "cache_hit": True,
        }

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _session_from_row(row: sqlite3.Row) -> ActiveLearningSession:
    return ActiveLearningSession(
        session_id=row["session_id"],
        dataset_version_id=row["dataset_version_id"],
        status=row["status"],
        config=SessionConfig.from_dict(_load(row["config_json"])),
        label_vocabulary=_load(row["label_vocabulary_json"]),
        current_round_id=row["current_round_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _round_from_payload(payload: Mapping[str, Any]) -> ActiveLearningRound:
    return ActiveLearningRound(
        round_id=payload["round_id"],
        session_id=payload["session_id"],
        round_index=int(payload["round_index"]),
        parent_round_id=payload.get("parent_round_id"),
        label_revision=int(payload["label_revision"]),
        status=payload["status"],
        analysis=payload.get("analysis", {}),
        rule_set=payload.get("rule_set", {}),
        display_rule_set=payload.get("display_rule_set", payload.get("rule_set", {})),
        projection=payload.get("projection", {}),
        recommendation_plans=payload.get("recommendation_plans", {}),
        delta=payload.get("delta", {}),
        cluster_lineage=payload.get("cluster_lineage", {}),
        created_at=payload["created_at"],
    )


def _label_event_from_row(row: sqlite3.Row) -> LabelEvent:
    return LabelEvent(
        event_id=row["event_id"],
        session_id=row["session_id"],
        round_id=row["round_id"],
        point_id=row["point_id"],
        label_dimension=row["label_dimension"],
        label_value=_load(row["label_value_json"]),
        status=row["status"],
        supersedes_event_id=row["supersedes_event_id"],
        provenance=_load(row["provenance_json"]),
        created_at=row["created_at"],
    )


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _load(value: str) -> Any:
    return json.loads(value)
