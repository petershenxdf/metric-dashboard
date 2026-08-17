# Rule Panel Design

## Role

rule_panel has two deterministic responsibilities:

1. generate shallow decision-tree surrogate rules from current SSDBCODI output;
2. build auditable next-point recommendation plans from rules, analysis evidence, and features.

It does not perform clustering, outlier detection, label persistence, or LLM calls.

## RuleSet

generate_rule_set learns separate explanation rules for current cluster and anomaly targets. Rule cards contain:

- stable rule ID and target;
- source-feature conditions;
- support, coverage, and purity;
- matched and exception point IDs;
- feature-use and tree diagnostics.

The active-learning display layer decodes transformed conditions into original numeric units or categorical values. Projection x/y coordinates are never rule features.

The decision tree is explanation-only. SSDBCODI remains the source of truth even when a rule is imperfect.

## Recommendation Categories

The deterministic engine evaluates:

- label_priority;
- boundary_review;
- overlap_merge_signal;
- split_or_new_cluster_signal;
- anomaly_label_review;
- exception_relabel_review;
- feature_label_strategy;
- rule_confidence_audit.

A category without a defensible typical case returns an unavailable plan instead of filling the UI with unrelated records.

## User-Facing Category Evidence Matrix

Every recommended record has one deterministic `CategoryEvidenceCard`. This
card is the direct source for the visible "Why this record was recommended"
checklist; it is not hidden prompt context.

Each card contains:

- the category question and evidence-policy version;
- every required evidence dimension in a fixed order;
- a `yes`, `partly`, `no`, or `insufficient` status;
- a fixed plain-language question and direct answer;
- the observation supporting that answer;
- the point-specific role of the record and what its human label would clarify;
- stable evidence fact IDs and folded technical details;
- comparison records and the original fields worth comparing;
- the relevant current/previous-round context.

The required dimensions are:

| Category | Fixed checks |
| --- | --- |
| Label Priority | why this question comes first; why the record is a clear example; whether one label covers several questions; new check or recheck; all checks inherited from the delegated category |
| Boundary Review | another group nearby; edge of own group; rule dividing line; mixed nearby groups; 2D/full-space agreement |
| Overlap Merge Signal | fits both descriptions; neighbors from both groups; resemblance to both groups; human labels in shared area; separation away from the overlap |
| Split Or New Cluster Signal | separation from own group; isolated point or coherent pocket; resemblance to an existing group; nearby human labels; persistence across rounds |
| Anomaly Label Review | unusual within own group; isolated or rare pattern; possible data-quality issue; status across rounds; confirmed comparison examples |
| Exception Relabel Review | rule mismatch; size of disagreement; neighbor support; resemblance to another group; isolated exception or broad rule weakness |
| Feature Label Strategy | repeated field use; position relative to the field's dividing line; agreement with the whole record; human-label pattern; shortcut risk |
| Rule Confidence Audit | group scope; current consistency; human-label agreement; exception pattern; stability across rounds |

All geometry and neighbor evidence is computed in the complete model feature
space. Projection coordinates are used only to report whether the visible 2D
story agrees with the full-space evidence. Missing human evidence produces an
explicit `insufficient` check instead of an inferred conclusion.

Main cards use qualitative language. Distances, ratios, percentiles, rule
thresholds, and calculation provenance remain available under `Technical
details`.

## Base RecommendationPlan

app/modules/rule_panel/recommendation.py builds a validated RecommendationPlan with:

- complete candidate pool;
- ordered recommended/highlighted IDs;
- category-specific target rules;
- per-point profiles;
- rank, selection reason, and score components;
- label questions and expected outcomes;
- deterministic plan identity.

The active-learning service upgrades this base plan to RecommendationPlanV2 by applying session history, active-label exclusions, justified rechecks, batch diversity, cross-category coverage, previous-plan diffs, and stop conditions.

RecommendationPlanV2 also carries `category_explanation`,
`evidence_policy_version`, and ordered `category_evidence_cards`. Card point
IDs must exactly match `recommended_point_ids`.

## Three Point Sets

Candidate pool means every eligible record considered for that category.

Recommended now means the smaller ordered batch the user should label in the current round.

Highlighted on plot means the same recommended IDs linked to numbered scatterplot callouts. Highlight is presentation state, not selection state.

These counts must be labeled separately in the UI.

## Stability

For a fixed RuleSet, analysis, feature matrix, session history, and category:

- candidate order is deterministic;
- recommendation order is deterministic;
- point ID is the final tie breaker;
- plan_id changes when decision-relevant state changes;
- DeepSeek cannot modify the plan.

## LLM Boundary

DeepSeek translation lives in active_learning/translation.py. It receives a compact TranslationPacket derived from RecommendationPlanV2. Rule-panel code never parses model prose and never accepts model-selected points.

The model may rewrite only the supplied observation and significance into
natural language. It must preserve point order, dimension order, statuses,
evidence fact IDs, and comparison target IDs. Invalid individual bullets use
their deterministic wording; an otherwise valid card is not discarded.

## Debug Surface

~~~text
/modules/rule-panel/
/modules/rule-panel/health
/modules/rule-panel/api/rules
/modules/rule-panel/api/state
~~~

The lab page displays Wine fixture rules as a regression aid. The final dashboard renders rules for any imported supported dataset.

## Tests

Tests cover raw-feature conditions, rule matching, deterministic RuleSet generation, schema validation, all-category plan stability, candidate/recommended/highlighted subset rules, and route envelopes.
