from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


@dataclass(frozen=True)
class ModuleInfo:
    slug: str
    package_name: str
    title: str
    purpose: str
    status: str = "planned"


@dataclass(frozen=True)
class WorkflowInfo:
    slug: str
    title: str
    purpose: str
    modules: Tuple[str, ...]
    status: str = "planned"


MODULES: Tuple[ModuleInfo, ...] = (
    ModuleInfo(
        slug="data-workspace",
        package_name="data_workspace",
        title="Data Workspace",
        purpose="Dataset loading, point IDs, metadata, and feature matrix.",
        status="working",
    ),
    ModuleInfo(
        slug="projection",
        package_name="projection",
        title="Projection",
        purpose="MDS projection into 2D coordinates.",
        status="working",
    ),
    ModuleInfo(
        slug="algorithm-adapters",
        package_name="algorithm_adapters",
        title="Algorithm Adapters",
        purpose="Wrappers for existing clustering and outlier algorithms.",
    ),
    ModuleInfo(
        slug="selection",
        package_name="selection",
        title="Selection",
        purpose="Selected and unselected point state.",
    ),
    ModuleInfo(
        slug="labeling",
        package_name="labeling",
        title="Labeling",
        purpose="Manual point annotations, cluster labels, and outlier labels.",
    ),
    ModuleInfo(
        slug="scatterplot",
        package_name="scatterplot",
        title="Scatterplot",
        purpose="Point rendering, clusters, outliers, and visual selection.",
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
    ),
    WorkflowInfo(
        slug="default-analysis",
        title="Default Analysis",
        purpose="Inspect projection with default clusters and outliers.",
        modules=("data-workspace", "projection", "algorithm-adapters"),
    ),
    WorkflowInfo(
        slug="selection-context",
        title="Selection Context",
        purpose="Inspect selected and unselected point context.",
        modules=("data-workspace", "selection"),
    ),
    WorkflowInfo(
        slug="selection-labeling",
        title="Selection and Labeling",
        purpose="Inspect selected points converted into manual label instructions.",
        modules=("data-workspace", "selection", "labeling"),
    ),
    WorkflowInfo(
        slug="scatter-selection",
        title="Scatter Selection",
        purpose="Inspect scatterplot interactions with selection and label state.",
        modules=("projection", "algorithm-adapters", "selection", "labeling", "scatterplot"),
    ),
    WorkflowInfo(
        slug="scatter-labeling",
        title="Scatter Labeling",
        purpose="Inspect visual point selection converted into label annotations.",
        modules=("projection", "algorithm-adapters", "selection", "labeling", "scatterplot"),
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


def list_workflows() -> Tuple[WorkflowInfo, ...]:
    return WORKFLOWS


def get_workflow(slug: str) -> Optional[WorkflowInfo]:
    return next((workflow for workflow in WORKFLOWS if workflow.slug == slug), None)
