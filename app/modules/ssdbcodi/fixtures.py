from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List

DEFAULT_FIXTURE_DATASET_ID = "ssdbcodi_demo_fixture"
MOONS_FIXTURE_DATASET_ID = "ssdbcodi_moons_fixture"
CIRCLES_FIXTURE_DATASET_ID = "ssdbcodi_circles_fixture"


@dataclass(frozen=True)
class SsdbcodiDatasetOption:
    dataset_id: str
    display_name: str
    description: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "dataset_id": self.dataset_id,
            "display_name": self.display_name,
            "description": self.description,
        }


DATASET_OPTIONS = (
    SsdbcodiDatasetOption(
        dataset_id=DEFAULT_FIXTURE_DATASET_ID,
        display_name="Separated clusters + outliers",
        description="Three compact groups with far outliers; useful for checking bootstrap anchors.",
    ),
    SsdbcodiDatasetOption(
        dataset_id=MOONS_FIXTURE_DATASET_ID,
        display_name="Two moons + outliers",
        description="Curved, non-convex groups for checking density expansion behavior.",
    ),
    SsdbcodiDatasetOption(
        dataset_id=CIRCLES_FIXTURE_DATASET_ID,
        display_name="Concentric circles + outliers",
        description="Inner and outer ring structure for testing non-linear cluster shape.",
    ),
)


def ssdbcodi_feature_names() -> List[str]:
    return ["x", "y"]


def ssdbcodi_raw_points(dataset_id: str | None = None) -> List[Dict[str, Any]]:
    dataset_id = normalize_ssdbcodi_dataset_id(dataset_id)
    if dataset_id == MOONS_FIXTURE_DATASET_ID:
        return _moons_raw_points()
    if dataset_id == CIRCLES_FIXTURE_DATASET_ID:
        return _circles_raw_points()
    return _separated_raw_points()


def ssdbcodi_dataset_id(dataset_id: str | None = None) -> str:
    return normalize_ssdbcodi_dataset_id(dataset_id)


def ssdbcodi_dataset_options() -> List[Dict[str, str]]:
    return [option.to_dict() for option in DATASET_OPTIONS]


def normalize_ssdbcodi_dataset_id(dataset_id: str | None) -> str:
    if dataset_id is None or dataset_id == "":
        return DEFAULT_FIXTURE_DATASET_ID
    if not isinstance(dataset_id, str):
        raise ValueError("dataset_id must be a string")
    dataset_id = dataset_id.strip()
    if not dataset_id:
        return DEFAULT_FIXTURE_DATASET_ID
    known_ids = {option.dataset_id for option in DATASET_OPTIONS}
    if dataset_id not in known_ids:
        raise ValueError(f"unknown SSDBCODI dataset_id: {dataset_id}")
    return dataset_id


def _separated_raw_points() -> List[Dict[str, Any]]:
    return [
        {"point_id": "ring_a_01", "features": [0.0, 0.0], "metadata": {"hint": "cluster_a"}},
        {"point_id": "ring_a_02", "features": [0.3, 0.1], "metadata": {"hint": "cluster_a"}},
        {"point_id": "ring_a_03", "features": [-0.2, 0.2], "metadata": {"hint": "cluster_a"}},
        {"point_id": "ring_a_04", "features": [0.1, -0.3], "metadata": {"hint": "cluster_a"}},
        {"point_id": "ring_a_05", "features": [-0.3, -0.1], "metadata": {"hint": "cluster_a"}},
        {"point_id": "ring_a_06", "features": [0.4, 0.4], "metadata": {"hint": "cluster_a"}},
        {"point_id": "ring_b_01", "features": [5.0, 5.0], "metadata": {"hint": "cluster_b"}},
        {"point_id": "ring_b_02", "features": [5.3, 4.7], "metadata": {"hint": "cluster_b"}},
        {"point_id": "ring_b_03", "features": [4.7, 5.2], "metadata": {"hint": "cluster_b"}},
        {"point_id": "ring_b_04", "features": [5.1, 5.4], "metadata": {"hint": "cluster_b"}},
        {"point_id": "ring_b_05", "features": [4.8, 4.8], "metadata": {"hint": "cluster_b"}},
        {"point_id": "ring_b_06", "features": [5.4, 5.1], "metadata": {"hint": "cluster_b"}},
        {"point_id": "ring_c_01", "features": [10.0, 0.0], "metadata": {"hint": "cluster_c"}},
        {"point_id": "ring_c_02", "features": [10.3, 0.3], "metadata": {"hint": "cluster_c"}},
        {"point_id": "ring_c_03", "features": [9.7, -0.2], "metadata": {"hint": "cluster_c"}},
        {"point_id": "ring_c_04", "features": [10.4, -0.4], "metadata": {"hint": "cluster_c"}},
        {"point_id": "ring_c_05", "features": [9.6, 0.1], "metadata": {"hint": "cluster_c"}},
        {"point_id": "ring_c_06", "features": [10.1, 0.4], "metadata": {"hint": "cluster_c"}},
        {"point_id": "outlier_far_01", "features": [20.0, 20.0], "metadata": {"hint": "outlier"}},
        {"point_id": "outlier_far_02", "features": [-12.0, 14.0], "metadata": {"hint": "outlier"}},
        {"point_id": "outlier_far_03", "features": [18.0, -14.0], "metadata": {"hint": "outlier"}},
    ]


def _moons_raw_points() -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for index in range(14):
        angle = math.pi * index / 13
        points.append(
            {
                "point_id": f"moon_upper_{index + 1:02d}",
                "features": [round(math.cos(angle), 4), round(math.sin(angle), 4)],
                "metadata": {"hint": "upper_moon"},
            }
        )
        points.append(
            {
                "point_id": f"moon_lower_{index + 1:02d}",
                "features": [
                    round(1.05 - math.cos(angle), 4),
                    round(-math.sin(angle) - 0.45, 4),
                ],
                "metadata": {"hint": "lower_moon"},
            }
        )
    points.extend(
        [
            {"point_id": "moon_outlier_01", "features": [-1.8, 1.55], "metadata": {"hint": "outlier"}},
            {"point_id": "moon_outlier_02", "features": [2.9, -1.75], "metadata": {"hint": "outlier"}},
            {"point_id": "moon_outlier_03", "features": [2.65, 1.15], "metadata": {"hint": "outlier"}},
        ]
    )
    return points


def _circles_raw_points() -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for index in range(12):
        angle = 2 * math.pi * index / 12
        points.append(
            {
                "point_id": f"circle_inner_{index + 1:02d}",
                "features": [round(math.cos(angle), 4), round(math.sin(angle), 4)],
                "metadata": {"hint": "inner_circle"},
            }
        )
    for index in range(18):
        angle = 2 * math.pi * index / 18
        points.append(
            {
                "point_id": f"circle_outer_{index + 1:02d}",
                "features": [round(3 * math.cos(angle), 4), round(3 * math.sin(angle), 4)],
                "metadata": {"hint": "outer_circle"},
            }
        )
    points.extend(
        [
            {"point_id": "circle_outlier_01", "features": [6.6, 6.0], "metadata": {"hint": "outlier"}},
            {"point_id": "circle_outlier_02", "features": [-6.8, 5.8], "metadata": {"hint": "outlier"}},
            {"point_id": "circle_outlier_03", "features": [0.2, -8.8], "metadata": {"hint": "outlier"}},
        ]
    )
    return points
