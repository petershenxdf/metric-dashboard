from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from flask import Blueprint, Flask


BlueprintFactory = Callable[[], "Blueprint"]


def _lazy_blueprint(import_path: str) -> BlueprintFactory:
    """Return a callable that imports and calls ``create_blueprint`` on first use.

    *import_path* is a dotted module path such as
    ``"app.modules.data_workspace"`` or ``"app.workflows.scatter_labeling"``.
    The target module must export ``create_blueprint()``.
    """

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
    status: str = "planned"
    blueprint_factory: Optional[BlueprintFactory] = None


@dataclass(frozen=True)
class WorkflowInfo:
    slug: str
    title: str
    purpose: str
    modules: Tuple[str, ...]
    status: str = "planned"
    blueprint_factory: Optional[BlueprintFactory] = None


MODULES: Tuple[ModuleInfo, ...] = (
    ModuleInfo(
        slug="data-workspace",
        package_name="data_workspace",
        title="Data Workspace",
        purpose="Dataset loading, point IDs, metadata, and feature matrix.",
        status="working",
        blueprint_factory=_lazy_blueprint("app.modules.data_workspace"),
    ),
    ModuleInfo(
        slug="projection",
        package_name="projection",
        title="Projection",
        purpose="MDS projection into 2D coordinates.",
        status="working",
        blueprint_factory=_lazy_blueprint("app.modules.projection"),
    ),
    ModuleInfo(
        slug="algorithm-adapters",
        package_name="algorithm_adapters",
        title="Algorithm Adapters",
        purpose="Wrappers for existing clustering and outlier algorithms.",
        status="working",
        blueprint_factory=_lazy_blueprint("app.modules.algorithm_adapters"),
    ),
    ModuleInfo(
        slug="selection",
        package_name="selection",
        title="Selection",
        purpose="Selected and unselected point state.",
        status="working",
        blueprint_factory=_lazy_blueprint("app.modules.selection"),
    ),
    ModuleInfo(
        slug="labeling",
        package_name="labeling",
        title="Labeling",
        purpose="Manual point annotations, cluster labels, and outlier labels.",
        status="working",
        blueprint_factory=_lazy_blueprint("app.modules.labeling"),
    ),
    ModuleInfo(
        slug="scatterplot",
        package_name="scatterplot",
        title="Scatterplot",
        purpose="Point rendering, clusters, outliers, and visual selection.",
        status="working",
        blueprint_factory=_lazy_blueprint("app.modules.scatterplot"),
    ),
    ModuleInfo(
        slug="ssdbcodi",
        package_name="ssdbcodi",
        title="SSDBCODI",
        purpose="Semi-supervised density-based clustering with integrated outlier detection.",
        status="working",
        blueprint_factory=_lazy_blueprint("app.modules.ssdbcodi"),
    ),
    ModuleInfo(
        slug="chatbox",
        package_name="chatbox",
        title="Chatbox",
        purpose="Dialogue UI for user feedback and clarification.",
    ),
    ModuleInfo(
        slug="intent-instruction",
        package_name="intent_instruction",
        title="Intent Instruction",
        purpose="Message classification and structured instruction output.",
    ),
    ModuleInfo(
        slug="metric-learning-adapter",
        package_name="metric_learning_adapter",
        title="Metric-Learning Adapter",
        purpose="Structured instruction to metric-learning constraints.",
    ),
    ModuleInfo(
        slug="refinement-orchestrator",
        package_name="refinement_orchestrator",
        title="Refinement Orchestrator",
        purpose="Coordinates the refinement update loop.",
    ),
)

WORKFLOWS: Tuple[WorkflowInfo, ...] = (
    WorkflowInfo(
        slug="data-projection",
        title="Data and Projection",
        purpose="Inspect dataset output beside MDS projection output.",
        modules=("data-workspace", "projection"),
        status="working",
        blueprint_factory=_lazy_blueprint("app.workflows.data_projection"),
    ),
    WorkflowInfo(
        slug="default-analysis",
        title="Default Analysis",
        purpose="Inspect projection with default clusters and outliers.",
        modules=("data-workspace", "projection", "algorithm-adapters"),
        status="working",
        blueprint_factory=_lazy_blueprint("app.workflows.default_analysis"),
    ),
    WorkflowInfo(
        slug="selection-context",
        title="Selection Context",
        purpose="Inspect selected and unselected point context.",
        modules=("data-workspace", "selection"),
        status="working",
        blueprint_factory=_lazy_blueprint("app.workflows.selection_context"),
    ),
    WorkflowInfo(
        slug="analysis-selection",
        title="Step 1-4 Analysis Selection",
        purpose="Inspect data, projection, outliers, clusters, and selection on one shared visual layer.",
        modules=("data-workspace", "projection", "algorithm-adapters", "selection"),
        status="working",
        blueprint_factory=_lazy_blueprint("app.workflows.analysis_selection"),
    ),
    WorkflowInfo(
        slug="selection-labeling",
        title="Selection and Labeling",
        purpose="Inspect selected points converted into manual label instructions.",
        modules=("data-workspace", "selection", "labeling"),
        status="working",
        blueprint_factory=_lazy_blueprint("app.workflows.selection_labeling"),
    ),
    WorkflowInfo(
        slug="analysis-labeling",
        title="Step 1-5 Analysis Labeling",
        purpose="Inspect data, projection, outliers, clusters, selection, and labeling on one shared visual layer.",
        modules=("data-workspace", "projection", "algorithm-adapters", "selection", "labeling"),
        status="working",
        blueprint_factory=_lazy_blueprint("app.workflows.analysis_labeling"),
    ),
    WorkflowInfo(
        slug="scatter-selection",
        title="Scatter Selection",
        purpose="Inspect scatterplot interactions with selection and label state.",
        modules=("projection", "algorithm-adapters", "selection", "labeling", "scatterplot"),
        status="working",
        blueprint_factory=_lazy_blueprint("app.workflows.scatter_selection"),
    ),
    WorkflowInfo(
        slug="scatter-labeling",
        title="Scatter Labeling",
        purpose="Inspect visual point selection converted into label annotations.",
        modules=("projection", "algorithm-adapters", "selection", "labeling", "scatterplot"),
        status="working",
        blueprint_factory=_lazy_blueprint("app.workflows.scatter_labeling"),
    ),
    WorkflowInfo(
        slug="chat-selection",
        title="Chat and Selection",
        purpose="Inspect chatbox behavior with selection context.",
        modules=("selection", "chatbox"),
    ),
    WorkflowInfo(
        slug="chat-intent",
        title="Chat and Intent",
        purpose="Inspect chat messages compiled into structured instructions.",
        modules=("chatbox", "intent-instruction"),
    ),
    WorkflowInfo(
        slug="instruction-constraints",
        title="Instruction Constraints",
        purpose="Inspect structured instructions converted into constraints.",
        modules=("intent-instruction", "metric-learning-adapter"),
    ),
    WorkflowInfo(
        slug="refinement-loop",
        title="Refinement Loop",
        purpose="Inspect the full refinement orchestration timeline.",
        modules=("metric-learning-adapter", "projection", "algorithm-adapters", "refinement-orchestrator"),
    ),
)


def list_modules(enabled_modules: Optional[Iterable[str]] = None) -> Tuple[ModuleInfo, ...]:
    if enabled_modules is None:
        return MODULES

    enabled = set(enabled_modules)
    unknown = enabled - {module.slug for module in MODULES}
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown module slug(s): {unknown_list}")

    return tuple(module for module in MODULES if module.slug in enabled)


def get_module(slug: str) -> Optional[ModuleInfo]:
    return next((module for module in MODULES if module.slug == slug), None)


def list_workflows(enabled_modules: Optional[Iterable[str]] = None) -> Tuple[WorkflowInfo, ...]:
    if enabled_modules is None:
        return WORKFLOWS

    enabled = {module.slug for module in list_modules(enabled_modules)}
    return tuple(workflow for workflow in WORKFLOWS if set(workflow.modules).issubset(enabled))


def get_workflow(slug: str) -> Optional[WorkflowInfo]:
    return next((workflow for workflow in WORKFLOWS if workflow.slug == slug), None)


def register_modules(app: "Flask", enabled_modules: Optional[Iterable[str]] = None) -> None:
    for module in list_modules(enabled_modules):
        if module.blueprint_factory is not None:
            app.register_blueprint(module.blueprint_factory())


def register_workflows(app: "Flask", enabled_modules: Optional[Iterable[str]] = None) -> None:
    for workflow in list_workflows(enabled_modules):
        if workflow.blueprint_factory is not None:
            app.register_blueprint(workflow.blueprint_factory())
