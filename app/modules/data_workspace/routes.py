from __future__ import annotations

from flask import Blueprint, jsonify, render_template

from app.shared.flask_helpers import api_success

from .fixtures import iris_feature_names, iris_raw_points
from .service import create_dataset, create_feature_matrix


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "data_workspace",
        __name__,
        template_folder="templates",
        url_prefix="/modules/data-workspace",
    )

    @blueprint.get("/")
    def index():
        dataset = _fixture_dataset()
        feature_matrix = create_feature_matrix(dataset)
        return render_template(
            "data_workspace/index.html",
            dataset=dataset,
            feature_matrix=feature_matrix,
            dependency_mode="fixture",
        )

    @blueprint.get("/health")
    def health():
        return jsonify(
            api_success(
                {"module": "data-workspace", "status": "working"},
                diagnostics={"dependency_mode": "fixture"},
            )
        )

    @blueprint.get("/api/dataset")
    def dataset_api():
        return jsonify(
            api_success(
                _fixture_dataset().to_dict(),
                diagnostics={"dependency_mode": "fixture"},
            )
        )

    @blueprint.get("/api/matrix")
    def matrix_api():
        matrix = create_feature_matrix(_fixture_dataset())
        return jsonify(
            api_success(
                matrix.to_dict(),
                diagnostics={"dependency_mode": "fixture"},
            )
        )

    @blueprint.get("/api/state")
    def state_api():
        dataset = _fixture_dataset()
        matrix = create_feature_matrix(dataset)
        return jsonify(
            api_success(
                {
                    "module": "data-workspace",
                    "status": "working",
                    "dataset_id": dataset.dataset_id,
                    "point_count": len(dataset.points),
                    "feature_count": len(dataset.feature_names),
                    "matrix_shape": [len(matrix.values), len(matrix.feature_names)],
                },
                diagnostics={"dependency_mode": "fixture"},
            )
        )

    return blueprint


def _fixture_dataset():
    return create_dataset(
        iris_raw_points(),
        dataset_id="iris_debug_sample",
        feature_names=iris_feature_names(),
    )
