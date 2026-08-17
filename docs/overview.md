# System Overview

## Product Definition

Metric Dashboard is a general interactive active-learning tool for structured tabular data. A user imports a dataset, inspects an SSDBCODI analysis, labels a small deterministic batch, and repeats the loop across persistent rounds.

The application has one final workflow:

~~~text
/workflows/active-learning-dashboard/
~~~

Module pages under /modules/ are engineering labs, not competing product flows.

## End-to-End Flow

~~~text
1. Import CSV, JSON, or MAT
2. Create immutable DatasetVersion and artifacts
3. Create ActiveLearningSession
4. Run Round 0:
   preprocessing -> projection -> SSDBCODI -> RuleSet
5. Build deterministic plans for all recommendation categories
6. Build fixed Category Evidence Matrix cards for every recommended point
7. Show one category, its points, evidence checklist, and comparison records
8. Optionally ask DeepSeek V4 Pro to rewrite that fixed evidence plainly
9. Commit semantic-class, outlier-status, or uncertain labels
10. Create the next round from all effective labels
11. Align cluster lineage, compute RoundDelta, and recommend again
~~~

## Retained Components

| Component | Responsibility |
| --- | --- |
| data_workspace | Shared Dataset and FeatureMatrix contracts |
| projection | Deterministic MDS coordinates for display |
| ssdbcodi | Semi-supervised clustering and integrated outlier detection |
| algorithm_adapters | Stable AnalysisResult provider boundary |
| selection | Transient selected-point state for module labs |
| labeling | Manual annotation contracts for module labs |
| scatterplot | Rendering and visual selection behavior |
| rule_panel | Decision-tree RuleSet and deterministic recommendation engine |
| active_learning | Generic import, persistence, rounds, labels, deltas, and LLM translation |
| active_learning_dashboard | The final user workflow |

## Data Boundary

DatasetAdapter implementations normalize CSV, JSON, and MAT into a PreparedDataset:

- immutable DatasetVersion metadata and fingerprints;
- raw records for display and rule decoding;
- a finite numeric FeatureMatrix for models;
- a feature transformation map;
- isolated optional ground truth.

Numeric missing values use median imputation and robust scaling. Categorical values use deterministic one-hot encoding with an explicit missing token. The transformation map converts model-space tree conditions back into source fields and readable values.

Generated point IDs depend on the content fingerprint and row index. A preprocessing change creates a new dataset version without silently changing source identity.

## Analysis Boundary

SSDBCODI is the source of truth for clusters and outliers. Manual labels become seeds or outlier constraints in the next round through the active-learning service.

A shallow decision tree learns to imitate current SSDBCODI assignments. Its RuleSet is explanation-only. Tree predictions never replace SSDBCODI output.

Cluster display names are not permanent semantic labels. Active learning stores stable user vocabulary separately and aligns cluster lineage between rounds by member overlap.

## Recommendation Boundary

The engine creates one deterministic RecommendationPlanV2 per category. It owns:

- complete eligible candidate profiles;
- category-specific evidence and ranking components;
- label/history exclusions and recheck reasons;
- stable tie-breaking;
- batch diversity;
- ordered recommended and highlighted point IDs;
- previous-plan differences and stop reasons.

The same round state produces the same plan. A point may cover several categories when that is genuinely informative.

For every recommended point, the Category Evidence Matrix asks a fixed set of
category-specific questions. Deterministic code assigns each question a
`yes`, `partly`, `no`, or `insufficient` status and supplies a plain fact, its
importance, comparison records, and folded technical evidence. Neighbor and
edge checks use the complete feature space; the 2D plot is only checked for
agreement.

## LLM Boundary

DeepSeek V4 Pro receives only a compact TranslationPacket containing the
selected plan, relevant rules, recommended point profiles, fixed evidence
cards, label options, previous label context, and RoundDelta.

It translates evidence into:

- natural wording for each supplied evidence check;
- what the user should compare;
- how a non-expert can decide the label;
- what each possible answer would teach the system.

Validation rejects responses that change immutable IDs, omit/reorder points or
checks, change evidence status, invent facts, change comparison targets, or
use disallowed technical prose. An invalid bullet uses deterministic wording
without discarding valid bullets. Provider failure returns complete
deterministic guidance and never blocks labeling.

## Persistence

SQLite stores metadata for datasets, sessions, rounds, LabelEvents, plans, and interpretation diagnostics. Compressed files hold larger raw and matrix artifacts.

Round lifecycle states are computing, ready_for_labeling, labels_committed, failed, and stopped. Label submission validates round ID, label revision, and plan ID to prevent stale writes.
