# Integration Testing

## Test Layers

### Pure services

Test preprocessing, projection, SSDBCODI, rule extraction, recommendation ranking, translation validation, round transitions, and persistence without Flask where possible.

### Module routes

Every retained module lab must load, expose health/state, and preserve its ownership boundary. Module tests must not depend on deleted workflows.

### Product workflow

The active-learning workflow tests cover import, Wine fixture creation through the generic path, session rendering, state API, category switching, labels, history, revert, and interpretation.

## Required Determinism Tests

For fixed dataset version, config, label revision, and category:

- analysis identity is stable;
- RuleSet identity is stable;
- plan_id is stable;
- candidate and recommended order is stable;
- highlighted IDs equal recommended IDs;
- every recommended point has a profile and ranking explanation;
- tie-breaking is stable by point ID after evidence and diversity terms.

Run these checks across all eight categories.

## Multi-Round Tests

Exercise at least five rounds and verify:

- labels accumulate correctly;
- corrections supersede old events;
- uncertain labels are not seeds;
- SSDBCODI reruns each round;
- lineage keeps group identity stable where possible;
- RoundDelta reflects changed groups, outliers, and rules;
- recent points are not repeated without a recheck reason;
- stale submissions fail;
- restart recovers the same session;
- revert restores effective state.

## Data Adapter Tests

Use numeric CSV, mixed CSV, missing values, custom JSON, and MAT. Verify:

- stable generated IDs;
- isolated ground truth;
- finite matrix values;
- categorical missing token behavior;
- raw/model transformation map;
- readable rule conditions;
- explicit capability errors for oversized data.

## DeepSeek Tests

Automated tests do not call the network or consume tokens. Patch the shared DeepSeek client and verify:

- model request is deepseek-v4-pro;
- temperature is zero;
- thinking is disabled;
- JSON response mode is used;
- response model and finish diagnostics are recorded;
- changed, missing, added, or reordered point IDs are rejected;
- unsupported technical prose is rejected;
- one repair attempt is allowed;
- fallback leaves the round usable;
- categories without a typical case skip the call.

## Manual Browser Check

Run python run.py, then inspect:

1. import/session index;
2. one Wine fixture session;
3. category availability and ordinary-language descriptions;
4. recommendation chips and numbered scatterplot callouts;
5. hover/click focus without selection mutation;
6. label commit and next-round timeline;
7. DeepSeek success and fallback status;
8. mobile layout without overlap.

## Commands

~~~bash
python -m unittest discover -s tests
python -m compileall app tests
git diff --check
~~~

Warnings from locally old Flask/Jinja dependencies should be distinguished from product failures. Dependency upgrades belong in a separate compatibility change.
