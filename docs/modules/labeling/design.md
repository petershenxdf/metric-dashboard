# Labeling Design

## Role

labeling owns manual annotation contracts in isolated module labs.

The final product uses the richer active_learning LabelEvent model for persistent, versioned semantic_class, outlier_status, and uncertain labels. The lab module remains useful for testing selection-to-annotation boundaries and SSDBCODI inputs.

## Lab Actions

The module supports assigning an existing cluster label, creating a new class, marking an outlier, and marking a normal point. Actions apply to explicit point IDs or the current SelectionContext.

## Product Mapping

- assign/new class becomes a stable semantic_class LabelEvent;
- outlier or normal becomes an outlier_status LabelEvent;
- insufficient evidence becomes an uncertain LabelEvent;
- corrections supersede older events for the same point and dimension.

Cluster IDs are changing analysis assignments and must not be stored as permanent user semantics in active-learning sessions.

## Invariants

- labels and visual selection have different owners;
- unknown points and unsupported values are rejected;
- uncertain labels never seed SSDBCODI;
- history is append-only in the active-learning store;
- user labels do not directly rewrite RuleSet or recommendation output.

## Debug Surface

~~~text
/modules/labeling/
/modules/labeling/health
/modules/labeling/api/state
/modules/labeling/api/apply
/modules/labeling/api/reset
~~~

## Tests

Tests cover annotation creation, reset, selection integration, validation, structured feedback conversion, and route envelopes.
