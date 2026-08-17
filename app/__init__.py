from __future__ import annotations

from flask import Blueprint, Flask


def _install_blueprint_method_shortcuts() -> None:
    """Backfill Flask 2-style Blueprint shortcuts for older local Flask."""

    if not hasattr(Blueprint, "get"):
        Blueprint.get = lambda self, rule, **options: self.route(rule, methods=["GET"], **options)
    if not hasattr(Blueprint, "post"):
        Blueprint.post = lambda self, rule, **options: self.route(rule, methods=["POST"], **options)
    if not hasattr(Blueprint, "delete"):
        Blueprint.delete = lambda self, rule, **options: self.route(rule, methods=["DELETE"], **options)
    if not hasattr(Blueprint, "put"):
        Blueprint.put = lambda self, rule, **options: self.route(rule, methods=["PUT"], **options)
    if not hasattr(Blueprint, "patch"):
        Blueprint.patch = lambda self, rule, **options: self.route(rule, methods=["PATCH"], **options)


_install_blueprint_method_shortcuts()

from .module_registry import register_modules, register_workflows
from .routes import core


def create_app(enabled_modules: list[str] | None = None) -> Flask:
    app = Flask(__name__)
    app.config["ENABLED_MODULES"] = enabled_modules

    register_modules(app, enabled_modules)
    register_workflows(app, enabled_modules)
    app.register_blueprint(core)
    return app
