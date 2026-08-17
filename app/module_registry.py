from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from flask import Blueprint, Flask


BlueprintFactory = Callable[[], "Blueprint"]


def _lazy_blueprint(import_path: str) -> BlueprintFactory:
    def factory() -> "Blueprint":
        module = importlib.import_module(import_path)
        return module.create_blueprint()

    return factory


@dataclass(frozen=True)
class ModuleInfo:
    slug: str
    package_name: str
    title: str
    purpose: str
    status: str = "working"
    blueprint_factory: Optional[BlueprintFactory] = None


@dataclass(frozen=True)
class WorkflowInfo:
    slug: str
    title: str
    purpose: str
    modules: Tuple[str, ...]
    status: str = "working"
    blueprint_factory: Optional[BlueprintFactory] = None


MODULES: Tuple[ModuleInfo, ...] = (
    ModuleInfo(
        "data-workspace",
        "data_workspace",
        "Data Workspace",
        "Dataset, point-ID, and feature-matrix contracts used by analysis modules.",
        blueprint_factory=_lazy_blueprint("app.modules.data_workspace"),
    ),
    ModuleInfo(
        "projection",
        "projection",
        "Projection",
        "Deterministic MDS projection for the dashboard scatterplot.",
        blueprint_factory=_lazy_blueprint("app.modules.projection"),
    ),
    ModuleInfo(
        "algorithm-adapters",
        "algorithm_adapters",
        "Algorithm Adapters",
        "Stable dashboard boundary around SSDBCODI cluster and outlier output.",
        blueprint_factory=_lazy_blueprint("app.modules.algorithm_adapters"),
    ),
    ModuleInfo(
        "selection",
        "selection",
        "Selection",
        "Point-selection contracts shared by visual debugging surfaces.",
        blueprint_factory=_lazy_blueprint("app.modules.selection"),
    ),
    ModuleInfo(
        "labeling",
        "labeling",
        "Labeling",
        "Manual class and outlier annotation contracts.",
        blueprint_factory=_lazy_blueprint("app.modules.labeling"),
    ),
    ModuleInfo(
        "scatterplot",
        "scatterplot",
        "Scatterplot",
        "Point rendering and visual selection behavior.",
        blueprint_factory=_lazy_blueprint("app.modules.scatterplot"),
    ),
    ModuleInfo(
        "ssdbcodi",
        "ssdbcodi",
        "SSDBCODI",
        "Semi-supervised density clustering with integrated outlier detection.",
        blueprint_factory=_lazy_blueprint("app.modules.ssdbcodi"),
    ),
    ModuleInfo(
        "rule-panel",
        "rule_panel",
        "Rule Panel",
        "Decision-tree surrogate rules that explain current SSDBCODI output.",
        blueprint_factory=_lazy_blueprint("app.modules.rule_panel"),
    ),
)

WORKFLOWS: Tuple[WorkflowInfo, ...] = (
    WorkflowInfo(
        slug="active-learning-dashboard",
        title="Active Learning Dashboard",
        purpose=(
            "Persistent multi-round active learning with deterministic "
            "recommendations, rule evidence, and constrained DeepSeek explanations."
        ),
        modules=tuple(module.slug for module in MODULES),
        blueprint_factory=_lazy_blueprint(
            "app.workflows.active_learning_dashboard"
        ),
    ),
)


def list_modules(
    enabled_modules: Optional[Iterable[str]] = None,
) -> Tuple[ModuleInfo, ...]:
    if enabled_modules is None:
        return MODULES
    enabled = set(enabled_modules)
    unknown = enabled - {module.slug for module in MODULES}
    if unknown:
        raise ValueError(
            f"Unknown module slug(s): {', '.join(sorted(unknown))}"
        )
    return tuple(module for module in MODULES if module.slug in enabled)


def get_module(slug: str) -> Optional[ModuleInfo]:
    return next((module for module in MODULES if module.slug == slug), None)


def list_workflows(
    enabled_modules: Optional[Iterable[str]] = None,
) -> Tuple[WorkflowInfo, ...]:
    if enabled_modules is None:
        return WORKFLOWS
    enabled = {module.slug for module in list_modules(enabled_modules)}
    return tuple(
        workflow
        for workflow in WORKFLOWS
        if set(workflow.modules).issubset(enabled)
    )


def get_workflow(slug: str) -> Optional[WorkflowInfo]:
    return next((workflow for workflow in WORKFLOWS if workflow.slug == slug), None)


def register_modules(
    app: "Flask",
    enabled_modules: Optional[Iterable[str]] = None,
) -> None:
    for module in list_modules(enabled_modules):
        if module.blueprint_factory is not None:
            app.register_blueprint(module.blueprint_factory())


def register_workflows(
    app: "Flask",
    enabled_modules: Optional[Iterable[str]] = None,
) -> None:
    for workflow in list_workflows(enabled_modules):
        if workflow.blueprint_factory is not None:
            app.register_blueprint(workflow.blueprint_factory())
