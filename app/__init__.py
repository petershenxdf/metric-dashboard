from __future__ import annotations

from flask import Flask

from .mockups import mockups
from .module_registry import register_modules, register_workflows
from .routes import core


def create_app(enabled_modules: list[str] | None = None) -> Flask:
    app = Flask(__name__)
    app.config["ENABLED_MODULES"] = enabled_modules

    register_modules(app, enabled_modules)
    register_workflows(app, enabled_modules)
    app.register_blueprint(mockups)
    app.register_blueprint(core)
    return app
