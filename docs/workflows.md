# Workflows

## Product Workflow

The repository has one registered workflow:

~~~text
/workflows/active-learning-dashboard/
~~~

It provides dataset import, session creation, scatterplot selection, label submission, SSDBCODI output, rule cards, deterministic recommendations, optional DeepSeek explanations, and round history.

A session dashboard is available at:

~~~text
/workflows/active-learning-dashboard/<session_id>/
~~~

The root route and /workflows/ redirect to the product entry.

## Dataset Entry

Users may upload CSV, JSON, or MAT data. The Wine button creates a session through the same generic import and persistence contracts and exists only as a repeatable fixture.

## Round Interaction

Within a session the user:

1. chooses a recommendation category;
2. reviews the fixed candidate and recommended counts;
3. links numbered recommendation chips to scatterplot points;
4. optionally generates a DeepSeek explanation;
5. selects or accepts points and submits labels;
6. receives a new round with updated analysis, rules, delta, and plans;
7. can inspect history or revert to a prior round.

## Module Labs

/modules/ lists isolated pages for data contracts, projection, adapters, selection, labeling, scatterplot, SSDBCODI, and rule generation. These pages are diagnostic surfaces. They do not form separate user workflows and should not own persistent active-learning state.

## Registration Rule

app/module_registry.py must contain exactly the retained modules and the active-learning-dashboard workflow. Adding another workflow requires a genuinely distinct product need, not merely a convenient integration test.
