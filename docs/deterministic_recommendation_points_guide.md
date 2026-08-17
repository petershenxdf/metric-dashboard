# Deterministic Recommendation Points Guide

## Purpose

The dashboard must recommend the same ordered points whenever the decision-relevant session state is the same. This guide documents how point choice stays deterministic while DeepSeek V4 Pro still adds useful human-readable explanations.

## Responsibility Split

Deterministic code owns:

- category availability;
- candidate eligibility;
- all ranking evidence and scores;
- batch diversity and history penalties;
- target rules;
- recommended point IDs and order;
- plot-highlight IDs;
- plan identity and stop reasons.

DeepSeek owns:

- ordinary-language explanation of supplied facts;
- what the user should compare;
- conditional labeling choices;
- what each possible answer would teach the next round.

DeepSeek cannot select, add, delete, replace, or reorder points.

## Inputs To A Plan

A RecommendationPlanV2 is a pure function of:

- dataset and preprocessing version;
- current analysis and RuleSet;
- session and round ID;
- effective label revision;
- focus category;
- label budget and batch configuration;
- prior recommendation history;
- previous-round changes.

Changing any decision-relevant input may produce a new plan_id. Repeating the same state must reproduce the same plan.

## Candidate Eligibility

The engine first creates complete point profiles, then filters candidates.

A point is normally eligible when it belongs to the current category's evidence and does not already have an effective label for the question being asked.

A labeled point may re-enter only when:

- its aligned group changed after later labels;
- its outlier status changed;
- a current rule conflicts with the human label;
- an explicit audit category requires review;
- no fresh candidate can answer the unresolved question.

Every re-entry includes a recheck_reason.

A point recommended in the immediately previous round is deferred unless new evidence justifies repetition or alternatives are exhausted.

## Ranking Order

Ranking is category-specific. Shared ordering stages are:

1. category evidence;
2. unresolved conflict or information value;
3. estimated scope of records affected;
4. preference for currently unlabeled records;
5. diversity from points already chosen for the batch;
6. recent-round repetition penalty;
7. all-history repetition penalty;
8. stable point-ID tie-breaking.

Floating-point components are normalized and stored for audit. The primary UI translates them into qualitative statements rather than score dumps.

## Three Distinct Sets

Candidate pool is the full eligible set considered for a category.

Recommended now is the smaller ordered batch selected for the current labeling action.

Highlighted on plot is the visual mapping of recommended now. It must contain the same IDs in the same order and does not change the user's real selection.

The UI must display candidate and recommended counts separately.

## Batch Construction

The default batch size is four. The default candidate pool size is max(12, batch size times 3). Overlap and exception categories may recommend up to six when a comparison needs more coverage.

The engine uses greedy diversity after category evidence so a batch tests a question from multiple useful positions instead of returning near-duplicates.

If one point answers several category questions, its profile records cross-category coverage. The system does not force artificial differences between categories.

## Category Logic

### Label Priority

A deterministic meta-ranker chooses the unresolved category with the strongest combination of conflict, affected scope, candidate availability, and lack of prior coverage. It then adopts that plan's fixed points.

### Boundary Review

Ranks records closest to meaningful source-feature rule boundaries and constructs comparisons across the boundary.

### Overlap Merge Signal

Ranks records in substantial intersections between rule regions, with optional contrast records. No typical overlap means no plan.

### Split Or New Cluster Signal

Ranks records from separated, weakly covered, or internally inconsistent subregions so labels can test whether distinct concepts exist.

### Anomaly Label Review

Ranks current unusual records and useful normal contrasts using anomaly-rule evidence and changed status.

### Exception Relabel Review

Ranks points that contradict the rule normally describing their current assignment, prioritizing unresolved and consequential exceptions.

### Feature Label Strategy

Ranks records that best test whether repeatedly used source features form a useful human labeling checklist.

### Rule Confidence Audit

Ranks representative matches and exceptions that can confirm or challenge the explanatory reliability of a target rule.

## Stable Tie-Breaking

All unordered collections are normalized before hashing or ranking. Category priority is fixed. Point ID is the final tie breaker. Plan IDs are hashes of canonical decision inputs, not timestamps or LLM text.

This prevents database row order, Python set order, API response order, and repeated model calls from changing the recommendation.

## History-Aware Active Learning

Each label commit creates a new immutable round. Recommendation history records how often and how recently every point was shown.

The next round:

- excludes already resolved points;
- penalizes repeated exposure;
- allows justified rechecks;
- compares the previous and current plan;
- explains added, removed, and retained recommendations;
- stops only for budget exhaustion or lack of eligible candidates;
- may suggest stopping when analysis has stabilized.

## Audit Fields

Each recommended point stores:

- candidate_rank;
- selection_reason;
- ranking_score_components;
- point profile and related rule IDs;
- covered categories;
- recheck reason when applicable;
- previous-plan status.

Candidate-pool points not selected now appear in deferred/not-selected summaries.

## DeepSeek Translation Contract

The TranslationPacket contains the fixed plan, relevant rule summaries, recommended point profiles, allowed label options, previous human labels, and RoundDelta. It excludes the complete dataset, unrelated points, ground truth, and mutable ranking instructions.

Accepted output must preserve:

- plan_id;
- focus_category;
- ordered recommended_point_ids;
- target_rule_ids;
- exactly one PointGuidance object per recommended point.

The request uses deepseek-v4-pro, temperature zero, JSON mode, and disabled thinking. The returned model name and schema are validated. Invalid output receives one repair attempt and then deterministic fallback.

## What Stability Means

The stability target is identical recommendation points, order, evidence contract, and plan identity for the same state.

DeepSeek prose does not need to be byte-for-byte identical. It is permitted to vary only inside validated explanation fields without changing the decision.

## Verification

Tests must cover all categories, invalid RecommendationPlan payloads, history exclusions, justified repeats, plot-highlight equality, stable plan IDs, stale label conflicts, DeepSeek contract rejection, and fallback usability.
