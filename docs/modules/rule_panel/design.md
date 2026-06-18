# Rule Panel Module Design

## Purpose

The rule panel is the active post-Step-8.5 product direction.

SSDBCODI remains responsible for clustering and outlier/anomaly detection. The
rule panel does not rerun, replace, or compete with SSDBCODI. Its job is to
explain the current SSDBCODI output and recommend the next labels that can
improve or validate it.

The system converts current SSDBCODI cluster assignments and anomaly flags into
human-readable decision rules. DeepSeek then interprets those generated rules
as quantitative label/refinement guidance: which points to label first, whether
overlap supports merge review, whether separated regions suggest split or
new-cluster review, and which anomaly or exception points need confirmation.

The default Step 8.6 visual and test dataset is the uploaded `wine.mat` file.
Rule conditions must use raw dataset feature names such as `alcohol`,
`malic_acid`, `flavanoids`, and `proline`; they must not fall back to projected
axis names such as `x` or `y`.

This design follows the explainable-clustering idea used by decision-tree
surrogates: represent an existing assignment as a short path of single-feature
thresholds. The decision tree is a rule extractor only. It is trained against
SSDBCODI outputs as fixed targets and must never be treated as the source of
cluster or anomaly truth.

Reference direction:
https://diglib.eg.org/server/api/core/bitstreams/887c6ef5-dfba-4e3f-be09-00e5d8de7641/content

## Responsibilities

1. Read the current dataset, feature matrix, SSDBCODI cluster result, anomaly
   result, point scores, and optional labeling state.
2. Train shallow, deterministic decision-tree surrogates using SSDBCODI output
   as fixed labels:
   - all non-outlier SSDBCODI clusters as multi-class rules,
   - each SSDBCODI cluster as one-vs-rest rules when needed,
   - SSDBCODI anomaly flags as anomaly-vs-normal rules.
3. Convert tree paths into rule cards with feature thresholds, support,
   coverage, purity, matched points, exception points, and source target.
4. Provide a Flask debug page for inspecting the generated rules beside the
   scatterplot and SSDBCODI diagnostics.
5. Send rule cards to the configured LLM provider for categorized
   interpretation.
6. Keep generated rules and LLM interpretations read-only. The rule panel does
   not relabel points, rerun SSDBCODI, mutate selection state, or write new
   cluster/outlier assignments.

## Not Responsible For

1. Producing the original clusters or anomaly scores.
2. Changing SSDBCODI parameters.
3. Running clustering or outlier/anomaly detection.
4. Owning selection or labeling state.
5. Treating LLM interpretation as a source of truth.
6. Continuing any old branching update roadmap.

## Target Files

```text
app/modules/rule_panel/
  __init__.py
  schemas.py
  decision_tree_rules.py
  service.py
  fixtures.py
  routes.py
  templates/rule_panel/index.html

tests/modules/rule_panel/
  test_decision_tree_rules.py
  test_service.py
  test_routes.py

app/workflows/rule_panel_validation.py
app/workflows/rule_interpretation.py
```

The initial implementation may keep the existing `chatbox` package during the
migration, but the product-facing target is `rule_panel`.

## Rule Generation Contract

The rule panel produces a `RuleSet`:

```json
{
  "rule_set_id": "rules_001",
  "dataset_id": "wine_mat",
  "source_analysis_run_id": "analysis_abc",
  "model": {
    "algorithm": "decision_tree_surrogate",
    "max_depth": 3,
    "min_samples_leaf": 2
  },
  "rules": [
    {
      "rule_id": "rule_cluster_2_001",
      "target_kind": "cluster",
      "target_id": "cluster_2",
      "conditions": [
        {"feature": "flavanoids", "operator": ">", "threshold": 1.95},
        {"feature": "proline", "operator": "<=", "threshold": 755.0}
      ],
      "support_count": 8,
      "coverage": 0.67,
      "purity": 0.88,
      "matched_point_ids": ["p3", "p7"],
      "exception_point_ids": ["p9"],
      "diagnostics": {
        "tree_depth": 2,
        "leaf_id": "leaf_4"
      }
    }
  ]
}
```

Rule quality fields:

1. `support_count` - number of points matching the rule.
2. `coverage` - share of the target class/anomaly set covered by the rule.
3. `purity` - share of matched points that belong to the target.
4. `exception_point_ids` - matched points whose current SSDBCODI label differs
   from the rule target.
5. `target_kind` - `cluster` or `anomaly`.

## Decision-Tree Boundary

Decision trees are trained only after SSDBCODI has produced cluster assignments
and anomaly flags.

They may:

1. approximate those assignments for explanation,
2. expose feature-threshold paths,
3. compute rule coverage, purity, support, matched points, and exceptions,
4. surface quality warnings when a cluster or anomaly region is not well
   represented by simple thresholds.

They must not:

1. create cluster ids,
2. create anomaly flags,
3. override SSDBCODI scores,
4. decide whether a point is normal or anomalous,
5. mutate selection, labels, projection, or stored SSDBCODI output.

## LLM Rule Interpretation Categories

DeepSeek's role is to parse generated rule cards into categorized, auditable
label/refinement recommendations. The output must be user-facing first:
what points to label, why those points are suspicious or strategically useful,
and how the user should label them. Quantitative findings remain available as
audit details, but they are not the primary reading path.

The DeepSeek prompt is stored in
`prompts/rule_interpretation/deepseek/label_guidance_prompt.txt`. In
DeepSeek mode, the rule interpreter calls `deepseek-v4-pro` with thinking
enabled and `reasoning_effort=high`, then validates the JSON output before
showing it.

DeepSeek calls can consume tokens even when the final JSON is not usable, for
example when thinking tokens use the generation budget before a final
`message.content` is emitted, or when the returned content is malformed JSON.
The rule interpreter therefore records DeepSeek response metadata such as
`finish_reason`, message keys, token usage, and attempt errors. If the first
DeepSeek JSON attempt has no final content or cannot be parsed, it retries once
with the same `deepseek-v4-pro` model in direct JSON mode before falling back
to the deterministic interpreter.

### Plain-Language Label Guidance

Rule interpretation is designed for a user who is deciding what to label next,
not for a data scientist auditing every metric. The primary reading path must
therefore use ordinary labeling language:

1. **Which category is being inspected?** Each category is a separate lens,
   such as boundary review, anomaly review, or rule confidence audit.
2. **Which points should the user label?** The answer must name concrete point
   ids. If the category has no typical case, the UI should say that directly
   instead of forcing unrelated points into the panel.
3. **Why are these points worth checking?** Explanations should translate rule
   evidence into human terms: the same wine is covered by two rules, a point is
   close to a raw-feature cutoff, a high-confidence rule has not been validated
   by human labels, or an anomaly score may still describe a normal member.
4. **How should the user label them?** Guidance should give label options and
   explain what each outcome would imply for merge, boundary, split/new-cluster,
   anomaly, or rule-audit decisions.

Recommendation point ids must be visually connected to the scatterplot. The
integrated dashboard highlights every recommended point with a non-mutating
halo on the plot and renders the same point ids as chips in the interpretation
panel. Hovering or focusing a chip temporarily highlights the corresponding
plot point; clicking a chip scrolls the scatterplot into view and keeps that
temporary highlight. This interaction is only for locating points and must not
change selection or labels.

Quantitative metrics such as support, coverage, purity, Jaccard overlap, and
cutoff distance remain available as audit evidence, but they should not be
the first thing shown to the user. When a number matters, DeepSeek must explain
what the number means for the next labeling decision.

User-facing terminology should be consistent:

1. Use `outlier score`, not `unusualness score` or `anomaly score`.
2. Use `current analysis` for SSDBCODI in user guidance.
3. Use `human label` or `user label`, not `semantic label`.
4. Use `cutoff` for a user-facing rule boundary; reserve `threshold` for audit
   payload fields and rule-card internals.

Every interpretation should be assigned one or more categories:

| Category | Meaning |
| --- | --- |
| `label_priority` | Ranks the next points or rule regions the user should label first. |
| `boundary_review` | Uses rule boundaries to decide where neighboring regions need labels. |
| `overlap_merge_signal` | Explains whether overlapping rules suggest merge/shared-boundary review. |
| `split_or_new_cluster_signal` | Explains whether separated or weakly covered regions suggest split/new-cluster review. |
| `anomaly_label_review` | Identifies outlier-rule points needing true-anomaly vs normal-member labels. |
| `exception_relabel_review` | Prioritizes rule exception points for relabel or boundary correction. |
| `feature_label_strategy` | Turns raw feature cutoffs into a checklist for manual labeling. |
| `rule_confidence_audit` | Audits support, coverage, purity, exceptions, and warnings before refinement. |

Suggested `RuleInterpretation` output:

```json
{
  "interpretation_id": "interp_001",
  "rule_set_id": "rules_001",
  "categories": ["label_priority", "boundary_review"],
  "target_rule_ids": ["rule_cluster_2_001"],
  "category_explanation": "Label priority ranks the next points whose labels reduce uncertainty.",
  "summary": "Rule rule_cluster_2_001 covers 34 points with purity 0.88 and coverage 0.67. Its nearest boundary rule shares 7 matched points, so these 7 labels should be inspected before changing the cluster structure.",
  "recommendation": "Label the 7 shared boundary points first; merge only if user labels show one semantic group.",
  "label_targets": [
    {
      "priority": "high",
      "rule_ids": ["rule_cluster_2_001", "rule_cluster_3_001"],
      "point_ids": ["wine_014", "wine_021"],
      "label_question": "Do these points belong to one semantic group or two?",
      "why_label_these_points": "They test the highest-impact boundary/overlap hypothesis."
    }
  ],
  "suspicion_reasons": [
    {
      "rule_ids": ["rule_cluster_2_001", "rule_cluster_3_001"],
      "point_ids": ["wine_014", "wine_021"],
      "suspicious_signal": "The points sit in the rule relation that can change the next refinement decision.",
      "rule_based_reason": "The linked rules share boundary evidence and should not drive merge/split without labels.",
      "point_based_reason": "The supplied point profiles expose raw feature values and threshold margins for these points."
    }
  ],
  "point_label_guidance": [
    {
      "rule_ids": ["rule_cluster_2_001", "rule_cluster_3_001"],
      "point_ids": ["wine_014", "wine_021"],
      "suggested_label_frame": "Choose cluster_2, cluster_3, true anomaly, new group, or uncertain.",
      "how_to_label": "Compare each point's raw feature values with the cited cutoffs, then record the user's human label.",
      "decision_impact": "Same labels support merge/shared-boundary review; different labels preserve the boundary.",
      "llm_analysis_note": "DeepSeek synthesizes this from rule metrics plus point-level profiles."
    }
  ],
  "quantitative_findings": [
    {
      "metric": "pair_jaccard_overlap",
      "value": 0.21,
      "rule_ids": ["rule_cluster_2_001", "rule_cluster_3_001"],
      "interpretation": "shared points divided by the union of both rule matches"
    }
  ],
  "suggested_label_actions": [
    {
      "action_type": "inspect_points",
      "priority": "high",
      "rule_ids": ["rule_cluster_2_001", "rule_cluster_3_001"],
      "point_ids": ["wine_014", "wine_021"],
      "reason": "these points test whether the overlapping rules represent one cluster or a boundary"
    }
  ],
  "evidence": [
    {
      "rule_ids": ["rule_cluster_2_001", "rule_cluster_3_001"],
      "feature": "flavanoids",
      "threshold": 1.95
    }
  ],
  "confidence": 0.82,
  "warnings": []
}
```

The LLM should never invent features, thresholds, clusters, anomalies, rule IDs,
target IDs, or point IDs. Rule-pair overlap and boundary metrics are computed in
code as `rule_guidance_metrics`. Candidate point raw feature values, current
cluster/outlier state, and threshold margins are provided as
`label_candidate_point_profiles`. If sample-level overlap is 0, the model must
not recommend merging directly; it should recommend boundary or representative
labels first.

## Flask Routes

```text
/modules/rule-panel/                         rule panel debug page
/modules/rule-panel/health                   module health
/modules/rule-panel/api/rules                generated RuleSet
/modules/rule-panel/api/interpret            DeepSeek interpretation for current rules
/modules/rule-panel/api/config               tree depth and leaf-size controls
/workflows/rule-panel-validation/            scatterplot + SSDBCODI + rule cards
/workflows/rule-interpretation/              rule cards + categorized DeepSeek output
```

## Flask Debug Page Requirements

The page should show:

1. current scatterplot summary and selected dataset,
2. one rule card group per cluster,
3. one rule card group for anomalies,
4. coverage and purity badges on each rule,
5. matched and exception point IDs,
6. tree settings (`max_depth`, `min_samples_leaf`) and source analysis run,
7. categorized DeepSeek interpretation output,
8. raw `RuleSet` and `RuleInterpretation` JSON for debugging.
9. raw feature usage and rule warning summaries so users can see which original
   wine features drive the surrogate rules.

## Testing

Unit tests:

1. Generates at least one cluster rule for a fixture with two or more clusters.
2. Generates anomaly-vs-normal rules when outliers exist.
3. Rule conditions reference only known feature names.
4. Rule matched points satisfy every condition.
5. Coverage and purity are deterministic for fixed input.
6. Low-purity rules include exception point IDs.
7. `wine.mat` loads as `wine_mat` with 129 points and 13 raw feature names.
8. Rule interpretation request payload contains only rule/state data, not API
   secrets.
9. LLM category validation rejects unknown categories.

Flask route tests:

1. debug page returns 200.
2. rules API returns a `RuleSet`.
3. interpret API returns categorized output for a mocked provider response.
4. invalid tree settings return a consistent API error envelope.

Manual browser check:

1. open `/modules/rule-panel/`,
2. confirm each current cluster has visible rule cards,
3. confirm anomalies have separate rule cards,
4. adjust tree depth and verify rules become simpler or more detailed,
5. run interpretation and confirm categories, evidence, and warnings are visible.

## Completion Criteria

The rule panel is complete when current SSDBCODI outputs can be explained as
decision-tree rules, every rule can be inspected in Flask, and DeepSeek can
return categorized, grounded interpretations without mutating any dashboard
state.

## Step 8.6 Status

Implemented for rule generation and local validation:

1. `/modules/rule-panel/` renders rule cards for SSDBCODI clusters and anomaly
   flags.
2. `/modules/rule-panel/api/rules` returns a `RuleSet`.
3. `/modules/rule-panel/api/interpret` returns a deterministic interpretation
   preview using the Step 8.7 category schema.
4. `/workflows/rule-panel-validation/` shows source SSDBCODI output, generated
   rules, and interpretation preview together.
5. The fixture path uses `wine.mat` as the default dataset and exposes raw wine
   feature names in rule conditions, diagnostics payloads, and visible rule
   cards.

## Step 8.7 Status

Implemented for categorized rule interpretation and label guidance:

1. `app/modules/rule_panel/interpretation.py` builds the auditable
   rule-interpretation payload from `RuleSet`, SSDBCODI summaries, feature
   names, rule ids, point ids, thresholds, and computed `rule_guidance_metrics`.
2. The parser validates model output into `RuleInterpretation` and rejects
   unknown categories, rule ids, raw features, point ids, target ids,
   thresholds, and action references. It requires `recommendation`,
   `quantitative_findings`, and `suggested_label_actions`.
3. `/modules/rule-panel/api/interpret` returns the current interpretation.
4. `/modules/rule-panel/api/interpretation` returns interpretation, request
   payload, and provider diagnostics.
5. `/workflows/rule-interpretation/` shows rule cards beside categorized
   recommendation, quantitative findings, suggested label actions, evidence,
   request payload, and provider diagnostics.
6. The workflow exposes one button per interpretation category. Clicking a
   category sends `focus_category` into the interpretation payload and returns a
   focused label/refinement recommendation for that category.
7. The default local provider is deterministic `mock`; `provider_kind=deepseek`
   uses the existing DeepSeek runtime configuration and falls back without
   mutating the `RuleSet` when provider calls fail.
