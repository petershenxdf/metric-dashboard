from __future__ import annotations

from .schemas import SelectionAction


def request_payload(request_obj):
    if request_obj.is_json:
        return request_obj.get_json(silent=True) or {}

    payload = dict(request_obj.form)
    payload.update(request_obj.args)
    return payload


def point_ids_from_payload(payload):
    point_ids = payload.get("point_ids", [])
    if isinstance(point_ids, str):
        return [point_id.strip() for point_id in point_ids.split(",") if point_id.strip()]
    return point_ids


def optional_point_ids_from_payload(payload):
    if "point_ids" not in payload or payload.get("point_ids") in (None, ""):
        return None
    return point_ids_from_payload(payload)


def selection_action_from_payload(action_name: str, payload, metadata=None) -> SelectionAction:
    return SelectionAction(
        action=action_name,
        point_ids=point_ids_from_payload(payload),
        source=str(payload.get("source", "api")),
        mode=payload.get("mode"),
        metadata=metadata if metadata is not None else payload.get("metadata", {}),
    )
