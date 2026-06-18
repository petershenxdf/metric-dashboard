# Rule Interpretation Categories Guide

Generated on: 2026-06-11

This guide explains the eight interpretation categories used by the Rule Panel.
These categories are not generic chat labels and they are not new clustering or
outlier-detection algorithms. Their purpose is to translate decision-tree
surrogate rules into concrete label/refinement guidance: what the user should
label next, why those labels are informative, and how the results could affect
merge, split, new-cluster, boundary, anomaly, or relabel decisions.

## 1. System Role

The current responsibility boundary is:

- SSDBCODI owns current cluster assignments and outlier flags.
- The decision tree only converts SSDBCODI outputs into readable surrogate
  rules. It does not recluster data and does not redetect outliers.
- Rule interpretation converts those rules into label/refinement guidance.
- DeepSeek or the mock interpreter must not mutate cluster, outlier, selection,
  or labeling state.

The interpretation layer answers questions such as:

- Which points should be labeled first?
- Which rule relation needs human confirmation?
- Does the evidence suggest merge review, split review, new-cluster review, or
  only additional labeling?
- Which anomaly or exception points need confirmation?
- Are the current rules strong enough to guide refinement?

## 2. Output Structure

Each `RuleInterpretation` should include:

- `categories`: one or more category ids.
- `category_explanation`: a short explanation of what the category examines.
- `recommendation`: the next action the user should take.
- `summary`: a concise but detailed explanation of why the recommendation
  matters.
- `label_targets`: the points the user should label now, plus the label
  question those points are meant to answer.
- `suspicion_reasons`: why those points are suspicious or strategically useful,
  including both rule-level and point-level evidence.
- `point_label_guidance`: how to label the selected points, including label
  options, a concrete checklist, and the decision impact of likely outcomes.
  This is the main LLM-authored point-level guidance.
- `decision_rationale`: strategic reasoning that connects rule evidence to a
  labeling or refinement decision. This is where the LLM adds value beyond raw
  metrics.
- `label_outcomes`: possible user-label outcomes and what each outcome would
  imply.
- `quantitative_findings`: auditable numbers such as support, coverage, purity,
  Jaccard overlap, exception count, and rule confidence score.
- `suggested_label_actions`: executable labeling actions. Each action should
  include `reason`, `hypothesis`, `why_this_action`, `expected_outcomes`, and
  `risk_note`.
- `evidence`: traceable rule ids, feature thresholds, target ids, and point ids.
- `warnings`: limitations such as no sample-level overlap or no exception
  points.

## 2.1 What the LLM Adds

The system computes quantitative metrics in code so numeric claims remain
auditable. The LLM should not invent support, overlap, thresholds, or point ids.
The LLM's real value is synthesis:

- turn rule overlap, boundary gaps, and exception points into testable
  hypotheses;
- explain why the selected points are more informative than random points;
- combine point-level raw feature values, threshold margins, current cluster
  assignment, and outlier status into concrete labeling guidance;
- describe what different label outcomes would imply;
- surface uncertainty and risk;
- convert raw feature thresholds into a user-facing labeling checklist.

A weak interpretation says only: "Jaccard overlap is 0.20."  
A useful interpretation says: "The two rules have Jaccard overlap 0.20. Label
wine_014 and wine_021 first because they test both the overlap and the boundary
semantics. If those points receive one semantic label, merge or shared-boundary
review becomes plausible. If the labels are mixed, the region is more likely a
boundary ambiguity or a new-cluster candidate."

When the system recommends inspecting raw feature thresholds, it is not saying
that the raw feature is wrong. The actual doubt is that a decision-tree
threshold only proves the surrogate can reproduce current SSDBCODI output; it
does not prove that the threshold matches the user's semantic labels. The user
should therefore compare candidate points' raw feature values and threshold
margins against their human label judgment.

## 3. Common Metrics

### Support

`support_count` is the number of points matched by a rule. High support means
the rule covers a larger region. Low support can still be useful, especially
for anomaly or exception inspection, but it should not define an entire cluster
by itself.

### Coverage

`coverage` is the fraction of the target class covered by the rule. High
coverage means the rule is representative of its target. Low coverage means the
target may require multiple rule regions, which can indicate split or
new-cluster review.

### Purity

`purity` is the fraction of matched points that already belong to the rule
target under the current SSDBCODI output. High purity means the rule is clean
with respect to SSDBCODI. Low purity indicates mixed membership and should
trigger label review.

### Rule Confidence Score

`rule_confidence_score = purity * coverage`. This is not a probability. It is a
ranking signal. A high score identifies stable rule regions; a low score
identifies rules that need audit before refinement.

### Exception Count and Exception Rate

Exception points are matched by a rule but disagree with that rule's target.
They are often the most valuable points to label because they expose direct
conflict between the surrogate rule and the current SSDBCODI assignment.

### Pair Intersection Count

`intersection_count` is the number of matched points shared by two rules. It is
the core evidence for overlap-based merge review. If two rules with different
targets share many points, those shared points should be labeled first.

### Jaccard Overlap

`jaccard_overlap = intersection / union`. It measures similarity between two
rule match sets. If sample-level overlap is zero, the system must not recommend
a merge from rule cards alone.

### Overlap Share A / B

`overlap_share_a` and `overlap_share_b` describe how much of each rule is
covered by the pair intersection. A small rule mostly covered by a larger rule
may be a subregion, a boundary slice, or an anomaly pocket.

### Boundary Gap

`boundary_gap` describes whether two rules are adjacent, overlapping, or
separated along shared raw features. Small or zero gaps are useful for boundary
review.

## 4. Category: Label Priority

### Meaning

Label Priority answers: "Which points or rule regions should the user label
first?"

### When To Use

Use this category when the user needs a global next step rather than a
specialized explanation. It is the best default overview category.

### Evidence Used

- highest-priority rule pair;
- label candidate groups;
- support, coverage, and purity;
- exception count;
- overlap or boundary relation;
- representative matched points.

### Expected Output

The output should state:

- which point ids should be labeled first;
- which rule ids they come from;
- why those labels are more useful than random labels;
- what refinement decision the labels could influence.

### How The User Should Use It

Treat Label Priority as an active-learning suggestion. It helps the user spend
labeling effort where it will most reduce uncertainty.

### Example

If the strongest rule covers 43 points with purity 1.00 and coverage 1.00, the
system may suggest labeling representative points from that rule to verify it
as a stable reference region. If another rule pair has cross-cluster overlap,
Label Priority should instead prioritize the overlap points.

### Caution

Label Priority does not change the model. It only chooses where to inspect
first.

## 5. Category: Boundary Review

### Meaning

Boundary Review answers: "Does a rule boundary correspond to a real semantic
boundary?"

### When To Use

Use it when two cluster rules are adjacent in raw feature space, when their
boundary gap is small, or when different target regions are close.

### Evidence Used

- shared raw feature names;
- threshold intervals;
- boundary gaps;
- pair relations such as `adjacent_cluster_boundary`;
- support, coverage, and purity on both sides;
- candidate point ids near the boundary.

### Expected Output

The output should explain:

- which two rules define the boundary;
- which raw feature threshold forms the boundary;
- which points should be labeled on each side;
- what same-label versus different-label outcomes would imply.

### How The User Should Use It

Label paired points from both sides of the boundary. A boundary cannot be
confirmed by labeling only one side.

### Example

If one rule is `proline <= 484` and the adjacent rule is `proline > 484 and
proline <= 645`, the system may suggest labeling representative points from
both sides of `proline = 484`.

### Caution

Boundary Review is not a direct merge recommendation. It only says the boundary
should be tested.

## 6. Category: Overlap Merge Signal

### Meaning

Overlap Merge Signal answers: "Do overlapping rules suggest that two targets
may share one semantic region?"

### When To Use

Use it when two rules have sample-level overlap, especially if they belong to
different targets.

### Evidence Used

- pair intersection count;
- Jaccard overlap;
- overlap share for each rule;
- shared point ids;
- rule target kinds and target ids;
- shared features and thresholds.

### Expected Output

The output should state:

- how many points are shared;
- the Jaccard overlap;
- the overlap share for both rules;
- which shared points should be labeled;
- what same-label and mixed-label outcomes would imply.

### How The User Should Use It

Label overlap points first. If they receive the same semantic label, merge or
shared-boundary review becomes plausible. If labels are mixed, keep targets
separate or investigate a new cluster.

### Example

If `rule_cluster_2` and `rule_cluster_3` share 12 points with Jaccard overlap
0.30, those 12 shared points should be labeled before proposing merge.

### Caution

If `intersection_count = 0`, the system must not recommend merge from rule
cards alone. It should fall back to Boundary Review or Label Priority.

## 7. Category: Split Or New Cluster Signal

### Meaning

Split Or New Cluster Signal answers: "Does an existing target appear to contain
multiple semantic subregions?"

### When To Use

Use it when the same target requires multiple disjoint rules, when coverage is
low but purity is high, or when separated regions appear inside one cluster.

### Evidence Used

- same-target disjoint regions;
- low coverage with high purity;
- multiple rules for one target;
- boundary gaps and shared features;
- exception groups;
- candidate point groups.

### Expected Output

The output should identify:

- which target may contain separate regions;
- which rule ids represent those regions;
- support, coverage, and purity for each region;
- which representative points should be labeled;
- what label outcomes would support split or new-cluster review.

### How The User Should Use It

Label across multiple regions. A split decision requires evidence that the
regions receive different human labels.

### Example

If one cluster needs three disjoint rules with different feature paths, label
representatives from each rule. Different labels support split; a label not
matching any current cluster can support new-cluster review.

### Caution

Split and new-cluster actions are expensive. They should remain hypotheses until
human labels support them.

## 8. Category: Anomaly Label Review

### Meaning

Anomaly Label Review answers: "Are current outlier-rule points true anomalies
or normal members near a boundary?"

### When To Use

Use it when anomaly rules exist, when anomaly support is small, when coverage is
low, or when anomaly regions are near cluster rules.

### Evidence Used

- anomaly rule support;
- anomaly rule coverage;
- anomaly point ids;
- cluster-anomaly relation;
- outlier score range;
- matched anomaly points.

### Expected Output

The output should state:

- which anomaly points need confirmation;
- which anomaly rule matched them;
- the rule's support, coverage, and purity;
- what true-anomaly versus normal-member labels would imply.

### How The User Should Use It

Label candidate points as true anomaly, normal member, or cluster member. These
labels help decide whether the outlier flag should be trusted.

### Example

If an anomaly rule matches only one point with coverage 0.06, the system should
recommend direct human confirmation rather than treating the rule as a stable
anomaly pattern.

### Caution

This category does not automatically remove outlier flags. It identifies
outlier candidates that need user confirmation.

## 9. Category: Exception Relabel Review

### Meaning

Exception Relabel Review answers: "Which rule exceptions should be relabeled or
used to fix boundaries?"

### When To Use

Use it when rules expose `exception_point_ids`, low purity, or mixed target
membership.

### Evidence Used

- exception point ids;
- exception count;
- exception rate;
- rule purity;
- raw feature thresholds for the exception rule;
- whether exception points cluster into a small region.

### Expected Output

The output should explain:

- which exception points need labels;
- which rule they contradict;
- exception count and exception rate;
- whether repeated exception labels would support relabel, boundary fix, or new
  cluster review.

### How The User Should Use It

Label exception points before ordinary matched points. They are direct evidence
of disagreement between current assignments and the surrogate explanation.

### Example

If `rule_cluster_2` matches 20 points and 4 of them currently belong to
`cluster_3`, those 4 exception points should be prioritized.

### Caution

Some configurations may produce no exception points. In that case the system
should warn, not invent relabel candidates.

## 10. Category: Feature Label Strategy

### Meaning

Feature Label Strategy answers: "Which raw feature thresholds should guide
manual labeling?"

### When To Use

Use it when certain raw features appear repeatedly in rules or when a threshold
is central to cluster or anomaly separation.

### Evidence Used

- feature usage counts;
- rule conditions;
- repeated thresholds;
- strongest rule paths;
- support, coverage, and purity;
- candidate points.

### Expected Output

The output should state:

- the most important raw features;
- repeated thresholds;
- which point ids should be labeled while checking those feature values;
- why projected x/y position is insufficient.

### How The User Should Use It

When labeling candidate points, inspect raw features such as `proline`,
`magnesium`, or `flavanoids`. Do not rely only on the 2D scatterplot location.

### Example

If `proline` appears 9 times and `magnesium` appears 2 times in the rules, the
system should tell the user to treat those features as the labeling checklist.

### Caution

Feature importance here is surrogate importance, not causal importance.

## 11. Category: Rule Confidence Audit

### Meaning

Rule Confidence Audit answers: "Are the rules reliable enough to guide
refinement?"

### When To Use

Use it before merge, split, new-cluster, or anomaly decisions. It is also useful
for broad, narrow, low-purity, low-coverage, or high-exception rules.

### Evidence Used

- support;
- coverage;
- purity;
- rule confidence score;
- exception count and exception rate;
- quality warnings;
- condition count;
- matched point previews.

### Expected Output

The output should identify:

- stable rules;
- weak rules;
- rules that need more labels;
- rules that should not yet drive refinement.

### How The User Should Use It

Treat it as a safety check. If rule quality is weak, gather labels before
changing cluster or anomaly state.

### Example

A rule with purity 1.00 but coverage 0.06 is clean but local. It is useful for
inspection, not for defining a whole cluster.

### Caution

High confidence means consistency with SSDBCODI, not semantic correctness.
Human labels remain the source of semantic truth.

## 12. Choosing A Category

- Unsure where to start: use `label_priority`.
- Suspect a weak cluster boundary: use `boundary_review`.
- Two rules share matched points: use `overlap_merge_signal`.
- One cluster appears to contain separated regions: use
  `split_or_new_cluster_signal`.
- Inspecting outliers: use `anomaly_label_review`.
- Exception points exist: use `exception_relabel_review`.
- Need raw-feature guidance for labeling: use `feature_label_strategy`.
- Before acting on rules: use `rule_confidence_audit`.

## 13. Core Principles

1. Rule interpretation provides label guidance; it does not mutate state.
2. The decision tree is a surrogate rule extractor, not a new clustering
   algorithm.
3. Merge, split, and new-cluster decisions require human labels.
4. If sample-level overlap is zero, do not recommend merge directly.
5. Low-support or low-coverage rules can still be useful for local inspection.
6. Exception points are often more valuable than ordinary matched points.
7. Projection is helpful for viewing, but explanations should return to raw
   feature thresholds.
