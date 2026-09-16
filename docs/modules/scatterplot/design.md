# Scatterplot Design

## Role

scatterplot turns projection, analysis, selection, and label state into a render payload. It owns visual encoding and interaction behavior, not source truth.

The final dashboard uses the same rendering concepts with persistent active-learning state.

## Visual Contract

Each rendered point may include:

- point ID and screen coordinates;
- current group color;
- outlier styling;
- selected state;
- recommendation state;
- numbered guidance-callout geometry.

The product uses dashed recommendation rings, solid selection outlines and amber comparison rings.
Cluster colors and diamond outlier markers remain unchanged. The six-score table uses its own
sequential blue scale. Plot and table selection work by pointer or keyboard.

## Invariants

- every rendered point maps to one source point ID;
- selected state comes from selection;
- groups/outliers come from the current analysis;
- recommendation order comes from the selected six-score category and visible-row gates;
- highlighted IDs equal that category's visible recommended batch;
- plot linking is visual only;
- labels and callouts must remain readable at desktop and mobile widths.

## Debug Surface

~~~text
/modules/scatterplot/
/modules/scatterplot/health
/modules/scatterplot/api/render-payload
/modules/scatterplot/api/state
/modules/scatterplot/api/select
/modules/scatterplot/api/toggle
/modules/scatterplot/api/clear
/modules/scatterplot/api/groups
~~~

## Tests

Tests cover payload composition, visual state precedence, selection delegation, group actions, deterministic callout mapping, and route envelopes.
