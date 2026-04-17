from __future__ import annotations

from flask import Blueprint, jsonify, render_template

from app.shared.flask_helpers import api_success

from .fixtures import fixture_projection, labels_by_point_id
from .service import scaled_projection_points


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "projection",
        __name__,
        template_folder="templates",
        url_prefix="/modules/projection",
    )

    @blueprint.get("/")
    def index():
        dataset, projection = fixture_projection()
        labels = labels_by_point_id(dataset)
        return render_template(
            "projection/index.html",
            dataset=dataset,
            projection=projection,
            scaled_points=scaled_projection_points(projection, labels),
            dependency_mode="real data-workspace fixture",
        )

    @blueprint.get("/health")
    def health():
        return jsonify(
            api_success(
                {"module": "projection", "status": "working"},
                diagnostics={"dependency_mode": "real data-workspace fixture"},
            )
        )

    @blueprint.get("/api/projection")
    def projection_api():
        dataset, projection = fixture_projection()
        return jsonify(
            api_success(
                projection.to_dict(),
                diagnostics={
                    "dependency_mode": "real data-workspace fixture",
                    "dataset_id": dataset.dataset_id,
                },
            )
        )

    @blueprint.get("/api/state")
    def state_api():
        dataset, projection = fixture_projection()
        return jsonify(
            api_success(
                {
                    "module": "projection",
                    "status": "working",
                    "dataset_id": dataset.dataset_id,
                    "projection_id": projection.projection_id,
                    "method": projection.method,
                    "coordinate_count": len(projection.coordinates),
                },
                diagnostics={"dependency_mode": "real data-workspace fixture"},
            )
        )

    return blueprint
