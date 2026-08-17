# Selection Design

## Role

selection owns transient user-selected point IDs and named selection groups in module labs.

In the final dashboard, browser selection is a local interaction state used to choose label targets. Persistent truth begins only when the active-learning label API commits LabelEvents.

## Actions

Supported actions are select, deselect, replace, toggle, and clear. Every action validates point IDs against the known dataset and records its source.

Named groups are convenience collections, not semantic classes.

## Invariants

- selected and unselected sets partition known points;
- unknown IDs are rejected;
- order is deterministic;
- visual recommendation highlight is separate from real selection;
- hovering or clicking a recommendation chip must not mutate selected IDs;
- selection never changes clustering, rules, labels, or recommendation plans.

## Debug Surface

~~~text
/modules/selection/
/modules/selection/health
/modules/selection/api/state
/modules/selection/api/select
/modules/selection/api/deselect
/modules/selection/api/replace
/modules/selection/api/toggle
/modules/selection/api/clear
/modules/selection/api/groups
~~~

## Tests

Tests cover every action, validation, group lifecycle, per-dataset store isolation, and route envelopes.
