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
    group: str = "Future Workflows"
    step: str = "future"
    debug_focus: str = ""
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
        purpose="Provider boundary for clustering and outlier analysis; defaults to SSDBCODI.",
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
        title="Step 1-2 Data Projection",
        purpose="Verify stable point IDs, feature matrix shape, and MDS coordinates together.",
        modules=("data-workspace", "projection"),
        group="Core Pipeline Smoke Tests",
        step="1-2",
        debug_focus="schema continuity from dataset rows to projection coordinates",
        status="working",
        blueprint_factory=_lazy_blueprint("app.workflows.data_projection"),
    ),
    WorkflowInfo(
        slug="default-analysis",
        title="Step 1-3 Analysis Provider",
        purpose="Verify projection plus SSDBCODI-backed clusters and outliers through algorithm_adapters.",
        modules=("data-workspace", "projection", "algorithm-adapters"),
        group="Core Pipeline Smoke Tests",
        step="1-3",
        debug_focus="active provider output and point-ID aligned analysis schemas",
        status="working",
        blueprint_factory=_lazy_blueprint("app.workflows.default_analysis"),
    ),
    WorkflowInfo(
        slug="selection-context",
        title="Step 4 Selection Boundary",
        purpose="Inspect selected/unselected point context without projection or analysis noise.",
        modules=("data-workspace", "selection"),
        group="State Boundary Probes",
        step="4",
        debug_focus="selection ownership, selected/unselected sets, and reusable context payload",
        status="working",
        blueprint_factory=_lazy_blueprint("app.workflows.selection_context"),
    ),
    WorkflowInfo(
        slug="selection-labeling",
        title="Step 5 Selection Labeling Boundary",
        purpose="Verify selected points become manual labeling annotations and structured feedback.",
        modules=("data-workspace", "selection", "labeling"),
        group="State Boundary Probes",
        step="5",
        debug_focus="selection-to-labeling handoff without projection or scatterplot dependencies",
        status="working",
        blueprint_factory=_lazy_blueprint("app.workflows.selection_labeling"),
    ),
    WorkflowInfo(
        slug="analysis-selection",
        title="Step 1-4 Analysis Selection",
        purpose="Inspect data, projection, outliers, clusters, and selection on one shared visual layer.",
        modules=("data-workspace", "projection", "algorithm-adapters", "selection"),
        group="Visual Integration Tests",
        step="1-4",
        debug_focus="analysis result plus click/rectangle selection on one SVG",
        status="working",
        blueprint_factory=_lazy_blueprint("app.workflows.analysis_selection"),
    ),
    WorkflowInfo(
        slug="analysis-labeling",
        title="Step 1-5 Analysis Labeling",
        purpose="Inspect data, projection, outliers, clusters, selection, and labeling on one shared visual layer.",
        modules=("data-workspace", "projection", "algorithm-adapters", "selection", "labeling"),
        group="Visual Integration Tests",
        step="1-5",
        debug_focus="manual labels passed into SSDBCODI and reflected in effective analysis state",
        status="working",
        blueprint_factory=_lazy_blueprint("app.workflows.analysis_labeling"),
    ),
    WorkflowInfo(
        slug="scatter-selection",
        title="Step 1-6 Scatter Selection",
        purpose="Inspect scatterplot interactions with selection and label state.",
        modules=("projection", "algorithm-adapters", "selection", "labeling", "scatterplot"),
        group="Visual Integration Tests",
        step="1-6",
        debug_focus="render payload selection behavior after scatterplot composition",
        status="working",
        blueprint_factory=_lazy_blueprint("app.workflows.scatter_selection"),
    ),
    WorkflowInfo(
        slug="scatter-labeling",
        title="Step 1-6 Scatter Labeling",
        purpose="Inspect visual point selection converted into label annotations.",
        modules=("projection", "algorithm-adapters", "selection", "labeling", "scatterplot"),
        group="Visual Integration Tests",
        step="1-6",
        debug_focus="full completed UI loop: render, select, label, refresh effective analysis",
        status="working",
        blueprint_factory=_lazy_blueprint("app.workflows.scatter_labeling"),
    ),
    WorkflowInfo(
        slug="provider-feedback",
        title="Step 6.5 Provider Feedback Lab",
        purpose="Compare the algorithm_adapters boundary with the standalone SSDBCODI score/debug contract.",
        modules=("data-workspace", "algorithm-adapters", "ssdbcodi"),
        group="Provider Diagnostics",
        step="6.5",
        debug_focus="SSDBCODI provider promotion, per-point scores, and downstream-ready schemas",
        status="working",
        blueprint_factory=_lazy_blueprint("app.workflows.provider_feedback"),
    ),
    WorkflowInfo(
        slug="chat-selection",
        title="Step 7 Chat Selection",
        purpose="Inspect chatbox behavior with selection context.",
        modules=("selection", "chatbox"),
        group="Future Workflows",
        step="7",
        debug_focus="chat UI receives current selection context",
    ),
    WorkflowInfo(
        slug="chat-intent",
        title="Step 8 Chat Intent",
        purpose="Inspect chat messages compiled into structured instructions.",
        modules=("chatbox", "intent-instruction"),
        group="Future Workflows",
        step="8",
        debug_focus="chat text becomes structured instruction deltas",
    ),
    WorkflowInfo(
        slug="instruction-constraints",
        title="Step 9 Instruction Constraints",
        purpose="Inspect structured instructions converted into constraints.",
        modules=("labeling", "intent-instruction", "metric-learning-adapter"),
        group="Future Workflows",
        step="9",
        debug_focus="structured instructions merge with labels into metric-learning constraints",
    ),
    WorkflowInfo(
        slug="refinement-loop",
        title="Step 10 Refinement Loop",
        purpose="Inspect the full refinement orchestration timeline.",
        modules=("data-workspace", "projection", "algorithm-adapters", "labeling", "metric-learning-adapter", "refinement-orchestrator"),
        group="Future Workflows",
        step="10",
        debug_focus="metric fit, transformed projection, rerun analysis, and rollback history",
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
