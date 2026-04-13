from __future__ import annotations

from flask import Flask

from .mockups import mockups
from .module_registry import list_modules
from .modules.data_workspace import create_blueprint as create_data_workspace_blueprint
from .modules.projection import create_blueprint as create_projection_blueprint
from .workflows.data_projection import create_blueprint as create_data_projection_blueprint
from .routes import core


def create_app(enabled_modules: list[str] | None = None) -> Flask:
    app = Flask(__name__)
    app.config["ENABLED_MODULES"] = enabled_modules
    list_modules(enabled_modules)

    if enabled_modules is None or "data-workspace" in enabled_modules:
        app.register_blueprint(create_data_workspace_blueprint())

    if enabled_modules is None or "projection" in enabled_modules:
        app.register_blueprint(create_projection_blueprint())

    if enabled_modules is None or {"data-workspace", "projection"}.issubset(set(enabled_modules)):
        app.register_blueprint(create_data_projection_blueprint())

    app.register_blueprint(mockups)
    app.register_blueprint(core)
    return app
