# Active Learning Design

## Role

active_learning owns the generic, persistent, multi-round product loop. It coordinates existing deterministic services while keeping their ownership boundaries intact.

~~~text
DatasetVersion -> Round 0 -> RecommendationPlanV2
-> LabelEvents -> Round 1 -> RoundDelta -> next plan
~~~

Wine uses the same adapter, store, analysis, recommendation, and UI contracts as uploaded data.

## Generic Data Layer

DatasetAdapter implementations support CSV, JSON, and MAT. import_dataset_bytes and import_records produce PreparedDataset with:

- immutable DatasetVersion and fingerprints;
- raw records and isolated optional ground truth;
- finite model FeatureMatrix;
- feature-role schema;
- transformation map and compressed artifact references.

Numeric missing values use median imputation and robust scaling. Categorical values use a visible missing token and deterministic one-hot encoding. Source point IDs remain stable across preprocessing-version changes.

## Session And Rounds

ActiveLearningSession binds one dataset version to configuration, vocabulary, budget, current round, and label revision.

Round 0 is the baseline. Every successful label batch creates a new ActiveLearningRound using all current effective labels. SSDBCODI is rerun rather than incrementally updated.

Round states are computing, ready_for_labeling, labels_committed, failed, and
stopped. Stale writes are rejected using expected round, revision, and plan
IDs. A label submission first computes a provisional next round, then commits
label events, the parent status, next-round snapshot, vocabulary, and session
head in one SQLite transaction. Failed analysis leaves the current round
unchanged.

## Label Model

Persistent dimensions are semantic_class, outlier_status, and uncertain.

Human semantic labels are stable and separate from changing cluster IDs. Corrections create superseding LabelEvents. Uncertain events are recorded but never become SSDBCODI seeds.

Already-labeled records are excluded by default. A labeled record may return only after group movement, outlier-status movement, or a current rule conflict, and the plan must provide a recheck_reason.

Before an SSDBCODI rerun, equal semantic labels are compiled into a must-link
seed relation. Different semantic labels are mapped to distinct existing
bootstrap groups when possible. The mapping is deterministic and recorded in
analysis diagnostics; the user-facing semantic vocabulary never becomes a
volatile cluster ID.

## History-Aware Recommendation

Each round starts with deterministic category plans from rule_panel and then applies:

- complete point profiles before ranking;
- active-label and history exclusions;
- category evidence;
- estimated affected scope;
- batch diversity;
- recent and all-history repetition penalties;
- justified rechecks;
- stable point-ID tie-breaking;
- cross-category coverage;
- previous-plan differences.

Default recommendation batch size is four. Candidate pool size is max(12, batch_size times 3); overlap and exception categories may recommend up to six.

label_priority is a deterministic meta-ranker over unresolved category plans. It never delegates priority or point choice to DeepSeek.

Recommendation history distinguishes:

- computed: a plan included the record;
- shown: the user opened a category that displayed the record;
- selected: the record was included in a submitted selection;
- labeled: the submission supplied a non-uncertain label.

Shown events are idempotent per round/plan/record. Ranking repetition penalties
use shown history, not every plan computed in the background.

After the ordered points are fixed, `category_evidence_v2` builds one
CategoryEvidenceCard per recommended record. It calculates full-feature-space
neighbors, own-group position, rule-line proximity, comparison exemplars,
human-label agreement, data-quality checks, rule exceptions, and round
stability. All required category dimensions remain visible, including an
explicit insufficient state when evidence is unavailable.

Every evidence dimension separates four deterministic fields: the fixed
question, its direct answer, the connection between that finding and this
specific record, and what a human label would clarify. Negative findings are
kept as counterevidence instead of being rewritten as reasons for attention.
Stored rounds using the earlier evidence contract are refreshed in memory
before rendering or translation, without changing their recommendation IDs.

## RoundDelta And Lineage

Cluster lineage aligns groups between rounds by maximum member overlap so display identities do not swap merely because internal ordering changed.

RoundDelta summarizes group membership changes, outlier changes, rule changes, resolved issues, and recommendation changes. This context explains why a record appears again or why the next batch differs.

Rule changes compare condition/threshold fingerprints, not only rule IDs.
Revert reconstructs effective labels from the selected round's parent ancestry.
Events record the resulting child round so two branches created from the same
parent remain distinguishable.

## DeepSeek Translation

The shared DeepSeekClient is configured for deepseek-v4-pro, temperature zero, direct JSON, and thinking disabled.

TranslationPacket contains only the fixed plan, relevant rules, recommended profiles, CategoryEvidenceCards without folded technical numbers, allowed labels, prior label context, and RoundDelta. It excludes full datasets, unrelated rows, ground truth, and non-recommended candidate rows.

DeepSeek writes one PointGuidance per recommended record. Every evidence bullet
must preserve its dimension ID, status, fact IDs, order, and comparison target
IDs. Prompt version `active_learning_round_translation_v5` requires three
non-overlapping outputs for each fixed question: a direct answer, one
supporting observation, and a point-specific explanation of what the human
label would clarify. The model cannot rewrite the fixed question or category
explanation.
Validation rejects changed IDs/order/rules/category, invented evidence,
unsupported direct analysis changes, and technical user-facing prose. A
failed bullet falls back locally to deterministic wording; immutable-contract
or provider failure falls back for the whole response and never blocks the
round.

Only a validated response whose returned model is deepseek-v4-pro is shown as DeepSeek-generated.

The dashboard requests interpretation only through the explicit POST action.
Page GETs never spend tokens, and cached interpretation is accepted only when
its prompt-template version matches the current contract.

## Persistence

SQLite stores dataset metadata, sessions, rounds, LabelEvents, plan snapshots, and interpretation diagnostics. Raw tables and model matrices use compressed artifacts referenced by fingerprint.

Service restart must recover the current session head and complete history.

MDS coordinates are cached by immutable FeatureMatrix. The full-feature
distance/neighbor graph is cached by dataset and preprocessing version.
Comparison exemplars are selected from that cached graph using current labels,
so label-dependent meaning is never stale.

## Stop Behavior

The session stops automatically only when the label budget is reached or no eligible candidate remains. Stable analysis produces a stop suggestion but allows the user to continue.

## Public Routes

~~~text
/workflows/active-learning-dashboard/
/workflows/active-learning-dashboard/<session_id>/
/api/datasets
/api/active-learning/sessions
/api/active-learning/sessions/<session_id>/state
/api/active-learning/sessions/<session_id>/history
/api/active-learning/sessions/<session_id>/rounds/<round_id>/labels
/api/active-learning/sessions/<session_id>/rounds/<round_id>/revert
/api/active-learning/sessions/<session_id>/rounds/<round_id>/categories/<category>/interpret
~~~

## Tests

Tests cover mixed data import, versioning, ground-truth isolation, five-round behavior, stable plan identity, history exclusions, rechecks, superseding events, stale conflicts, revert, restart recovery, TranslationPacket minimization, model/schema validation, fallback, and final workflow routes.
