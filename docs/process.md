# Development Process

## Current Product Baseline

The active-learning dashboard is the product baseline. New work must improve this workflow rather than revive historical Step pages.

A change is consistent only when code, tests, prompts, UI language, schemas, and documentation describe the same responsibilities.

## Change Sequence

1. Identify the owning component.
2. Read its schema, service, tests, and final-dashboard use.
3. Change pure logic before Flask rendering.
4. Preserve deterministic IDs and ordering.
5. Add or update focused tests.
6. Verify the integrated active-learning workflow.
7. Update the relevant current document.
8. Remove superseded code or text instead of keeping parallel narratives.

## Product Invariants

- SSDBCODI is the source of truth for clusters and outliers.
- Decision trees explain SSDBCODI output only.
- Recommendation points are chosen deterministically.
- Category Evidence Matrix deterministically explains why each point is shown.
- DeepSeek rewrites fixed evidence and cannot modify facts, statuses, or points.
- User labels are stored separately from changing cluster assignments.
- Each successful label batch creates a new immutable round.
- Ground truth is evaluation-only.
- Wine-specific behavior must remain inside fixtures.
- Provider failure cannot block deterministic labeling.
- A stale round, label revision, or plan cannot be committed.

## Recommendation Changes

Any ranking change must document:

- the category-specific evidence;
- candidate eligibility rules;
- batch size and diversity behavior;
- history penalty or recheck exception;
- deterministic tie-breaking;
- expected effect on plan_id;
- tests across all eight categories.

Do not move point selection into prompts. A new LLM field is acceptable only when it translates evidence already supplied by deterministic code.

## LLM Changes

The active prompt is prompts/active_learning/deepseek/round_guidance_prompt.txt.

Keep the task translational:

- state supplied facts in ordinary language;
- explain what the user should inspect;
- offer conditional label outcomes;
- preserve immutable IDs;
- avoid unsupported domain conclusions.

A DeepSeek response is product-visible only after model and schema validation. Mock and fallback guidance must use the same PointGuidance contract.

Every explanation change must preserve:

- all fixed evidence dimensions for the selected category;
- `yes`, `partly`, `no`, or `insufficient` status;
- evidence fact IDs and comparison target IDs;
- point and bullet order;
- a local deterministic fallback for any invalid bullet.

The page may call DeepSeek only after an explicit POST action. A normal GET
must not spend tokens.

## Data Changes

Dataset adapters must preserve:

- content and version fingerprints;
- stable source point IDs;
- explicit feature, metadata, and ground-truth roles;
- raw/model feature separation;
- transformation-map reversibility;
- finite model matrices.

Add adapter tests for missing values, mixed types, custom IDs, and hidden ground truth.

## Round Changes

Round transitions must be transactional. Compute the next round provisionally,
then atomically commit label events, parent status, child round, vocabulary,
and session head. Test at least:

- baseline creation;
- valid label commit;
- superseding label event;
- stale submission conflict;
- history-aware exclusion;
- justified recheck;
- revert or branch behavior;
- isolation between two branches created from the same parent;
- failed next-round computation with no partial label commit;
- recovery after process restart.

## UI Changes

The final dashboard should show only information needed to select, label, inspect SSDBCODI/rules, understand recommendations, and review round history.

Use ordinary language in recommendation cards. Each evidence bullet must show
the fixed question, answer it directly, state what the system observed, and
explain why labeling this particular record would clarify that question.
Negative findings must remain visible as counterevidence; they must not be
turned into generic reasons for attention. Technical scores belong in folded
`Technical details` or stored diagnostics, not the primary instruction.

Recommendation chips may highlight scatterplot points but must never mutate the real selection.

Comparison targets must be identified as human-confirmed or system examples.
Highlighting a recommended/comparison pair must not mutate the real
selection.

## Verification

Run before considering a change complete:

~~~bash
python -m unittest discover -s tests
python -m compileall app tests
git diff --check
~~~

For visual changes, run the app and inspect the import page and a session dashboard at desktop and mobile widths.
