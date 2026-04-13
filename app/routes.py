from __future__ import annotations

from flask import Blueprint, abort, current_app, jsonify, render_template

from .module_registry import get_module, get_workflow, list_modules, list_workflows
from .shared.flask_helpers import api_error, api_success

core = Blueprint("core", __name__)


@core.get("/")
def home():
    return render_template(
        "home.html",
        modules=_enabled_modules(),
        workflows=_enabled_workflows(),
    )


@core.get("/health")
def health():
    return jsonify(api_success({"status": "ok", "app": "metric-dashboard"}))


@core.get("/modules/")
def modules_index():
    return render_template("modules_index.html", modules=_enabled_modules())


@core.get("/modules/<module_slug>/")
def module_placeholder(module_slug: str):
    module = _get_enabled_module_or_404(module_slug)
    return render_template("module_placeholder.html", module=module)


@core.get("/modules/<module_slug>/health")
def module_health(module_slug: str):
    module = _get_enabled_module_or_404(module_slug)
    return jsonify(
        api_success(
            {
                "module": module.slug,
                "status": module.status,
            },
            diagnostics={"dependency_mode": "placeholder"},
        )
    )


@core.get("/modules/<module_slug>/api/state")
def module_state(module_slug: str):
    module = _get_enabled_module_or_404(module_slug)
    return jsonify(
        api_success(
            {
                "module": module.slug,
                "title": module.title,
                "purpose": module.purpose,
                "status": module.status,
            },
            diagnostics={
                "dependency_mode": "placeholder",
                "note": "Module-specific implementation has not been built yet.",
            },
        )
    )


@core.get("/workflows/")
def workflows_index():
    return render_template("workflows_index.html", workflows=_enabled_workflows())


@core.get("/workflows/<workflow_slug>/")
def workflow_placeholder(workflow_slug: str):
    workflow = _get_enabled_workflow_or_404(workflow_slug)
    return render_template("workflow_placeholder.html", workflow=workflow)


@core.errorhandler(404)
def not_found(error):
    return jsonify(api_error("not_found", "The requested resource was not found.")), 404


def _enabled_modules():
    return list_modules(current_app.config.get("ENABLED_MODULES"))


def _enabled_workflows():
    return list_workflows(current_app.config.get("ENABLED_MODULES"))


def _get_enabled_module_or_404(module_slug: str):
    module = get_module(module_slug)
    if module is None:
        abort(404)

    enabled_slugs = {enabled_module.slug for enabled_module in _enabled_modules()}
    if module.slug not in enabled_slugs:
        abort(404)

    return module


def _get_enabled_workflow_or_404(workflow_slug: str):
    workflow = get_workflow(workflow_slug)
    if workflow is None:
        abort(404)

    enabled_slugs = {enabled_workflow.slug for enabled_workflow in _enabled_workflows()}
    if workflow.slug not in enabled_slugs:
        abort(404)

    return workflow
