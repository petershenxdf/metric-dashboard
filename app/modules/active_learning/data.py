from __future__ import annotations

from abc import ABC, abstractmethod
import csv
import hashlib
import io
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from app.shared.schemas import Dataset, FeatureMatrix, Point

from .schemas import DatasetVersion, FeatureSpec


PREPROCESSING_VERSION = "mixed_tabular_v1"
MISSING_CATEGORY = "__missing__"
DEFAULT_PREPROCESSING_CONFIG = {
    "numeric_imputation": "median",
    "numeric_scaling": "robust_iqr",
    "categorical_imputation": "explicit_missing_token",
    "categorical_encoding": "one_hot",
    "missing_category_token": MISSING_CATEGORY,
}


@dataclass(frozen=True)
class PreparedDataset:
    version: DatasetVersion
    dataset: Dataset
    feature_matrix: FeatureMatrix
    raw_records: Tuple[Mapping[str, Any], ...]

    def with_artifacts(self, raw_path: str, matrix_path: str) -> "PreparedDataset":
        return replace(
            self,
            version=replace(
                self.version,
                raw_artifact_path=raw_path,
                matrix_artifact_path=matrix_path,
            ),
        )


class DatasetAdapter(ABC):
    source_format: str

    @abstractmethod
    def import_bytes(self, content: bytes, **options: Any) -> PreparedDataset:
        """Convert one source payload into the shared prepared-data contract."""


class CsvDatasetAdapter(DatasetAdapter):
    source_format = "csv"

    def import_bytes(self, content: bytes, **options: Any) -> PreparedDataset:
        return import_csv_bytes(content, **options)


class JsonDatasetAdapter(DatasetAdapter):
    source_format = "json"

    def import_bytes(self, content: bytes, **options: Any) -> PreparedDataset:
        return import_json_bytes(content, **options)


class MatDatasetAdapter(DatasetAdapter):
    source_format = "mat"

    def import_bytes(self, content: bytes, **options: Any) -> PreparedDataset:
        return import_mat_bytes(content, **options)


_DATASET_ADAPTERS = {
    adapter.source_format: adapter
    for adapter in (
        CsvDatasetAdapter(),
        JsonDatasetAdapter(),
        MatDatasetAdapter(),
    )
}


def dataset_adapter(source_format: str) -> DatasetAdapter:
    normalized = str(source_format or "").strip().lower()
    try:
        return _DATASET_ADAPTERS[normalized]
    except KeyError as exc:
        raise ValueError("source_format must be csv, json, or mat") from exc


def import_dataset_bytes(
    content: bytes,
    source_format: str,
    **options: Any,
) -> PreparedDataset:
    return dataset_adapter(source_format).import_bytes(content, **options)


def import_csv_bytes(
    content: bytes,
    *,
    dataset_id: str | None = None,
    entity_name: str = "record",
    point_id_column: str | None = None,
    feature_columns: Sequence[str] | None = None,
    metadata_columns: Sequence[str] = (),
    ground_truth_columns: Sequence[str] = (),
    preprocessing_config: Mapping[str, Any] | None = None,
) -> PreparedDataset:
    text = content.decode("utf-8-sig")
    records = tuple(dict(row) for row in csv.DictReader(io.StringIO(text)))
    return prepare_records(
        records,
        dataset_id=dataset_id,
        entity_name=entity_name,
        source_format="csv",
        point_id_column=point_id_column,
        feature_columns=feature_columns,
        metadata_columns=metadata_columns,
        ground_truth_columns=ground_truth_columns,
        preprocessing_config=preprocessing_config,
    )


def import_json_bytes(
    content: bytes,
    *,
    dataset_id: str | None = None,
    entity_name: str = "record",
    point_id_column: str | None = None,
    feature_columns: Sequence[str] | None = None,
    metadata_columns: Sequence[str] = (),
    ground_truth_columns: Sequence[str] = (),
    preprocessing_config: Mapping[str, Any] | None = None,
) -> PreparedDataset:
    payload = json.loads(content.decode("utf-8"))
    records = payload.get("records") if isinstance(payload, Mapping) else payload
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("JSON dataset must be a list of records or an object with records")
    return prepare_records(
        tuple(dict(row) for row in records),
        dataset_id=dataset_id,
        entity_name=entity_name,
        source_format="json",
        point_id_column=point_id_column,
        feature_columns=feature_columns,
        metadata_columns=metadata_columns,
        ground_truth_columns=ground_truth_columns,
        preprocessing_config=preprocessing_config,
    )


def import_mat_bytes(
    content: bytes,
    *,
    dataset_id: str | None = None,
    entity_name: str = "record",
    matrix_key: str = "X",
    label_key: str = "y",
    feature_names: Sequence[str] | None = None,
    preprocessing_config: Mapping[str, Any] | None = None,
) -> PreparedDataset:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError("scipy is required to import MAT datasets") from exc

    payload = loadmat(io.BytesIO(content))
    if matrix_key not in payload:
        raise ValueError(f"MAT dataset must contain matrix key {matrix_key}")
    matrix = np.asarray(payload[matrix_key])
    if matrix.ndim != 2:
        raise ValueError("MAT feature matrix must be two-dimensional")
    names = tuple(feature_names or (f"feature_{index + 1}" for index in range(matrix.shape[1])))
    if len(names) != matrix.shape[1]:
        raise ValueError("feature_names length must match MAT matrix width")
    labels = None
    if label_key in payload:
        labels = np.asarray(payload[label_key]).reshape(-1)
        if len(labels) != matrix.shape[0]:
            raise ValueError("MAT label vector length must match matrix rows")
    records = []
    for row_index, row in enumerate(matrix):
        record = {name: float(row[column]) for column, name in enumerate(names)}
        record["point_id"] = f"{entity_name}_{row_index + 1:03d}"
        if labels is not None:
            record["ground_truth"] = _json_scalar(labels[row_index])
        records.append(record)
    return prepare_records(
        records,
        dataset_id=dataset_id,
        entity_name=entity_name,
        source_format="mat",
        point_id_column="point_id",
        feature_columns=names,
        ground_truth_columns=("ground_truth",) if labels is not None else (),
        preprocessing_config=preprocessing_config,
    )


def prepare_records(
    records: Sequence[Mapping[str, Any]],
    *,
    dataset_id: str | None = None,
    entity_name: str = "record",
    source_format: str = "json",
    point_id_column: str | None = None,
    feature_columns: Sequence[str] | None = None,
    metadata_columns: Sequence[str] = (),
    ground_truth_columns: Sequence[str] = (),
    preprocessing_config: Mapping[str, Any] | None = None,
) -> PreparedDataset:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
        raise ValueError("records must be a non-empty sequence")
    normalized = tuple(_normalize_record(row) for row in records)
    preprocessing = _normalize_preprocessing_config(preprocessing_config)
    columns = _ordered_columns(normalized)
    metadata_columns = tuple(metadata_columns)
    ground_truth_columns = tuple(ground_truth_columns)
    excluded = set(metadata_columns) | set(ground_truth_columns)
    if point_id_column:
        excluded.add(point_id_column)
    selected_features = tuple(feature_columns or (column for column in columns if column not in excluded))
    if not selected_features:
        raise ValueError("at least one feature column is required")
    unknown = set(selected_features) - set(columns)
    if unknown:
        raise ValueError(f"unknown feature column(s): {', '.join(sorted(unknown))}")

    content_fingerprint = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    fingerprint_payload = {
        "content_fingerprint": content_fingerprint,
        "dataset_id": str(dataset_id or ""),
        "entity_name": str(entity_name or "record"),
        "source_format": str(source_format or "json").strip().lower(),
        "point_id_column": point_id_column,
        "feature_columns": selected_features,
        "metadata_columns": metadata_columns,
        "ground_truth_columns": ground_truth_columns,
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing_config": preprocessing,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    clean_dataset_id = _safe_id(dataset_id or f"dataset_{fingerprint[:10]}")
    point_ids = _point_ids(normalized, point_id_column, content_fingerprint)
    missing_category = preprocessing["missing_category_token"]
    feature_specs = tuple(
        _feature_spec(
            feature,
            tuple(row.get(feature) for row in normalized),
            missing_category=missing_category,
        )
        for feature in selected_features
    )
    model_names, transformed, transformation_map = _transform(
        normalized,
        feature_specs,
        preprocessing=preprocessing,
    )
    created_at = _now_iso()
    dataset_version_id = f"dsv_{fingerprint[:16]}"

    raw_records = []
    points = []
    for index, point_id in enumerate(point_ids):
        raw_features = {
            feature: normalized[index].get(feature)
            for feature in selected_features
        }
        metadata = {
            column: normalized[index].get(column)
            for column in metadata_columns
        }
        ground_truth = {
            column: normalized[index].get(column)
            for column in ground_truth_columns
        }
        raw_record = {
            "point_id": point_id,
            "raw_features": raw_features,
            "metadata": metadata,
            "ground_truth": ground_truth,
        }
        raw_records.append(raw_record)
        points.append(
            Point(
                point_id=point_id,
                features=tuple(float(value) for value in transformed[index]),
                metadata=raw_record,
            )
        )

    dataset = Dataset(
        dataset_id=clean_dataset_id,
        points=tuple(points),
        feature_names=tuple(model_names),
        created_at=created_at,
    )
    feature_matrix = FeatureMatrix(
        point_ids=point_ids,
        feature_names=tuple(model_names),
        values=tuple(tuple(float(value) for value in row) for row in transformed),
    )
    version = DatasetVersion(
        dataset_version_id=dataset_version_id,
        dataset_id=clean_dataset_id,
        fingerprint=fingerprint,
        content_fingerprint=content_fingerprint,
        entity_name=entity_name or "record",
        source_format=str(source_format or "json").strip().lower(),
        point_ids=point_ids,
        feature_specs=feature_specs,
        metadata_columns=metadata_columns,
        ground_truth_columns=ground_truth_columns,
        model_feature_names=tuple(model_names),
        transformation_map=transformation_map,
        preprocessing_version=PREPROCESSING_VERSION,
        preprocessing_config=preprocessing,
        created_at=created_at,
    )
    return PreparedDataset(
        version=version,
        dataset=dataset,
        feature_matrix=feature_matrix,
        raw_records=tuple(raw_records),
    )


def reconstruct_prepared_dataset(
    version: DatasetVersion,
    raw_records: Sequence[Mapping[str, Any]],
    values: Sequence[Sequence[float]],
) -> PreparedDataset:
    rows = tuple(tuple(float(value) for value in row) for row in values)
    points = tuple(
        Point(
            point_id=point_id,
            features=rows[index],
            metadata=dict(raw_records[index]),
        )
        for index, point_id in enumerate(version.point_ids)
    )
    dataset = Dataset(
        dataset_id=version.dataset_id,
        points=points,
        feature_names=version.model_feature_names,
        created_at=version.created_at,
    )
    matrix = FeatureMatrix(
        point_ids=version.point_ids,
        feature_names=version.model_feature_names,
        values=rows,
    )
    return PreparedDataset(
        version=version,
        dataset=dataset,
        feature_matrix=matrix,
        raw_records=tuple(dict(item) for item in raw_records),
    )


def display_condition(
    model_feature: str,
    operator: str,
    threshold: float,
    transformation_map: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    transform = next(
        (item for item in transformation_map if item.get("model_feature") == model_feature),
        None,
    )
    if transform is None:
        return {
            "source_feature": model_feature,
            "display_operator": operator,
            "display_value": threshold,
            "display_text": f"{model_feature} {operator} {threshold:.3f}",
        }
    source = str(transform["source_feature"])
    if transform.get("kind") == "numeric":
        raw_threshold = float(threshold) * float(transform["scale"]) + float(transform["median"])
        return {
            "source_feature": source,
            "display_operator": operator,
            "display_value": round(raw_threshold, 6),
            "display_text": f"{source} {operator} {raw_threshold:.3f}",
        }
    category = str(transform["category"])
    positive = operator == ">" if threshold >= 0.0 else operator == "<="
    display_operator = "=" if positive else "!="
    return {
        "source_feature": source,
        "display_operator": display_operator,
        "display_value": category,
        "display_text": f"{source} {display_operator} {category}",
    }


def _feature_spec(
    name: str,
    values: Tuple[Any, ...],
    *,
    missing_category: str,
) -> FeatureSpec:
    missing_count = sum(1 for value in values if _is_missing(value))
    present = tuple(value for value in values if not _is_missing(value))
    if present and all(_as_float(value) is not None for value in present):
        return FeatureSpec(name=name, kind="numeric", missing_count=missing_count)
    categories = tuple(
        sorted(
            {str(value) for value in present}
            | ({missing_category} if missing_count else set())
        )
    )
    return FeatureSpec(
        name=name,
        kind="categorical",
        missing_count=missing_count,
        categories=categories,
    )


def _transform(
    records: Tuple[Mapping[str, Any], ...],
    specs: Tuple[FeatureSpec, ...],
    *,
    preprocessing: Mapping[str, Any],
) -> tuple[Tuple[str, ...], np.ndarray, Tuple[Mapping[str, Any], ...]]:
    columns = []
    names = []
    mapping = []
    for spec in specs:
        values = tuple(record.get(spec.name) for record in records)
        if spec.kind == "numeric":
            present = np.asarray(
                [float(_as_float(value)) for value in values if not _is_missing(value)],
                dtype=float,
            )
            median = float(np.median(present)) if len(present) else 0.0
            q1 = float(np.percentile(present, 25)) if len(present) else median
            q3 = float(np.percentile(present, 75)) if len(present) else median
            scale = q3 - q1
            if abs(scale) < 1e-12:
                scale = 1.0
            column = np.asarray(
                [
                    0.0
                    if _is_missing(value)
                    else float(_as_float(value)) - median
                    for value in values
                ],
                dtype=float,
            ) / scale
            columns.append(column)
            names.append(spec.name)
            mapping.append(
                {
                    "model_feature": spec.name,
                    "source_feature": spec.name,
                    "kind": "numeric",
                    "imputation": "median",
                    "median": median,
                    "scale": scale,
                    "q1": q1,
                    "q3": q3,
                }
            )
            continue
        missing_category = preprocessing["missing_category_token"]
        for category in spec.categories:
            model_name = f"encoded::{spec.name}=={category}"
            column = np.asarray(
                [
                    1.0
                    if (missing_category if _is_missing(value) else str(value)) == category
                    else 0.0
                    for value in values
                ],
                dtype=float,
            )
            columns.append(column)
            names.append(model_name)
            mapping.append(
                {
                    "model_feature": model_name,
                    "source_feature": spec.name,
                    "kind": "categorical",
                    "category": category,
                    "imputation": missing_category,
                }
            )
    matrix = np.column_stack(columns)
    return tuple(names), matrix, tuple(mapping)


def _point_ids(
    records: Tuple[Mapping[str, Any], ...],
    point_id_column: str | None,
    fingerprint: str,
) -> Tuple[str, ...]:
    if point_id_column:
        point_ids = tuple(str(row.get(point_id_column, "")).strip() for row in records)
        if any(not point_id for point_id in point_ids):
            raise ValueError("point ID column contains empty values")
    else:
        point_ids = tuple(
            f"p_{fingerprint[:8]}_{index + 1:05d}"
            for index in range(len(records))
        )
    if len(set(point_ids)) != len(point_ids):
        raise ValueError("point IDs must be unique")
    return point_ids


def _ordered_columns(records: Tuple[Mapping[str, Any], ...]) -> Tuple[str, ...]:
    columns = []
    for row in records:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    return tuple(columns)


def _normalize_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError("each record must be an object")
    return {str(key): _json_scalar(value) for key, value in record.items()}


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _safe_id(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "_-" else "_" for char in str(value))
    return cleaned.strip("_") or "dataset"


def _normalize_preprocessing_config(
    value: Mapping[str, Any] | None,
) -> Dict[str, str]:
    config = {
        **DEFAULT_PREPROCESSING_CONFIG,
        **dict(value or {}),
    }
    expected = {
        "numeric_imputation": "median",
        "numeric_scaling": "robust_iqr",
        "categorical_imputation": "explicit_missing_token",
        "categorical_encoding": "one_hot",
    }
    for field_name, supported in expected.items():
        if config.get(field_name) != supported:
            raise ValueError(
                f"{field_name} currently supports only {supported}"
            )
    missing_token = str(config.get("missing_category_token", "")).strip()
    if not missing_token:
        raise ValueError("missing_category_token must not be empty")
    config["missing_category_token"] = missing_token
    return {key: str(config[key]) for key in DEFAULT_PREPROCESSING_CONFIG}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
