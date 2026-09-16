# Six-score active learning

The current implementation follows the supplied Dashboard_Active_Learning_LaTeX design.
The previous eight-category RecommendationPlanV2 / DeepSeek system is removed.

## Ownership and loop

active_learning owns dataset versions, preprocessing, sessions, effective human labels,
saved rounds and the six-score snapshot. SSDBCODI remains model truth; tree rules only explain it.
Each new round runs SSDBCODI with all effective labels, records model changes, reuses the parent
projection, calculates the six scores and saves them with the analysis. GET never recomputes scores.

## Definitions

Let h = MinPts, d = ordinary Euclidean model-feature distance, r = mutual reachability distance,
and L = human semantic references. Exclude the candidate itself from every reference/neighborhood.
Human-confirmed normal references include semantic examples and explicit Normal feedback,
excluding explicitly confirmed outliers. Predicted outliers and bootstrap anchors are not human references.

| Category ID | Raw score (larger means more reason to review) | Availability / initial recommendation gate |
| --- | --- | --- |
| cluster_assignment_conflict | fraction of nearest normal neighbors in a different normal group | normal candidate, at least one comparison, up to h normal neighbors; G > 0 |
| label_coverage_gap | minimum r to another human semantic example | at least one reference, support check passes; P ≥ 75 |
| weak_group_reachability | 1 − exp(−minimum full expansion-path bottleneck to another human normal reference) | complete path and reference; G > 0 and P ≥ 75 |
| local_sparsity | 1 − exp(−mean r to h nearest other records) | at least h other records, including predicted outliers; G > 0 and P ≥ 75 |
| known_outlier_similarity | exp(−minimum d to another human-confirmed outlier) | reference exists; P ≥ 75 and distance ≤ reference's local radius |
| model_instability | mean pairwise 1 − Jaccard(normal-group member sets) | at least two complete nearby-setting runs; G > 0 and P ≥ 75 |

Coverage support radius is the distance to the min(h, n−1)th other data point, using all records.
The cutoff is the eligible radius at ceil(0.90 × eligible count), with boundary ties included.
A failed coverage support check means no calculable coverage score for that pool.
The known-outlier local radius uses the same ordinary-distance neighborhood at its reference.
A failed local match excludes recommendation but preserves the similarity raw score and percentile.

For each category's eligible, calculable pool:
P = 100 × (number lower + 0.5 × number equal) / pool size.
All exact ties get the same value; an all-tied pool gets 50. All tied is explicitly displayed.
No other category, browser sort or visible-row filter changes this pool.
No aggregate score or cross-category comparison of expected benefit is implied.

## Full expansion and instability

The previous rScore used direct distance to a seed. It is now based on recorded full expansion
paths. SSDBCODI records a deterministic Prim minimum-spanning expansion tree on the complete
mutual-reachability graph. Its paths preserve minimax bottlenecks. The core rScore uses model
seeds plus explicit Normal feedback, while review G3 uses only human-supported normal references
on that same recorded tree. Normal feedback does not create a semantic class seed.
This is an explicit implementation choice for the document's previously missing expansion-path
support; the updated model feedback contract is expansion_normal_feedback_v2. Existing saved
analyses are not rewritten; subsequent rounds use the updated model.
Point details reconstruct the complete selected path and show its bottleneck.

Instability fixes input data, preprocessing, labels, all other settings and deterministic
bootstrap behavior, then runs the complete SSDBCODI pipeline at h−1, h and h+1 within [1,n−1].
The current h result is reused. Group member sets include the candidate. Outliers have empty
sets: two empties disagree by zero, one empty by one. Group names never affect the score.
Splits and merges are detected by member overlap. Missing outputs or failed runs yield NA,
not a silently reduced ensemble. Run settings, exact group memberships and errors are stored.

## Eligibility, feedback and stopping

Normally only unlabeled points enter percentile pools. Previously confirmed points may re-enter
when their normal-group member set or outlier status changes relative to the parent round.
Immediately submitted confirmations are not instantly requested again for that same transition.
The re-review reason is shown. Semantic and outlier labels remain separate dimensions.

Unsure saves an uncertain LabelEvent, excludes the point for one subsequent round and never
enters any confirmed reference set. Later rounds can include it again as unlabeled.
An existing confirmed label is not erased merely by an unsure answer.

Recommendations use their individual category gate, descending unrounded percentile and stable
point ID. Batches keep one representative per identical transformed feature vector.
Default batch size is 4. Default numeric percentile threshold is 75 and is session-configurable.
Empty queues are legitimate; manual labeling remains possible. Only the configured effective
label-dimension budget automatically stops a session. The 2,000-point exact-analysis limit remains.

Label commits validate current round and label revision, compute the next snapshot, then atomically
save superseding events, the next round and the new session head. Failed analysis commits nothing.
Corrections, lineage alignment, history and branch-aware revert are retained.

## Persistence and legacy sessions

New round JSON contains review with version six_scores_expansion_v1, rows, categories, parameters,
recorded expansion tree and instability group evidence. It no longer contains recommendation_plans.
Old round payloads load with an empty review field. The dashboard asks for an explicit POST upgrade,
which adds a new round from existing labels. It never reruns the legacy model during GET.
Existing legacy interpretation/recommendation tables, if present, are left untouched and unused.
New databases do not create these tables. No existing dataset or label history is deleted.

## UI

One point per row and six fixed columns; ColorBrewer Blues bands use the unrounded percentile:
[0,25), [25,50), [50,75), [75,90), [90,100]. Fills are #EFF3FF, #BDD7E7, #6BAED6,
#3182BD, #08519C; text is black except white on the last band. NA uses gray diagonal stripes.
Numbers, full-name hover details, focus states and keyboard controls supplement color.

Sort keeps NA last in both directions and ties by point ID. Status, group and ID filters always
restrict results; numeric and explicit NA conditions can use Match ALL or Match ANY.
Recommendable-only additionally enforces the chosen category's gates.
Filters have removable chips, visible counts, reset and empty-state messaging.
Selecting a row/tile/plot point links both views without changing cluster colors. Hidden selection
is explicitly announced. Raw scores, precise percentiles, full paths, neighbors and alternate
member sets are available in details. Explicit labels refresh plot, scores and queue together.

Labeling mode is explicit: Single targets the focused point; checking records enters Batch mode.
Purple double rings show every checked target in the plot, independently of the solid focused-point
outline. Batch submission lists the exact IDs and requires confirmation. An empty batch never falls
back to the focused point. Switching to Single clears the batch; successful submissions clear batch
targets so that a later action cannot silently reuse them. Request failures preserve the batch.

Null percentile with a finite raw value and ineligible status is shown as Reviewed / Deferred,
with the raw evidence visible in details. Missing evidence remains patterned NA with a visible reason.
Per-column reference guidance explains cold start without substituting model predictions for human
labels. Zero instability explicitly says that the nearby model settings agree.

Search, group/status, recommendation filter, sort column/direction and ALL/ANY score conditions
are serialized in the page URL's validated view parameter. They survive reload, label submission
and revert without affecting frozen scores. A group absent in a later round is retained as an
explicit no-match filter rather than silently broadened. Selected point/category remain independent
of the sort column. No labels or ground truth are stored in the view parameter.

## Evidence boundary

Tests verify formulas and implementation behavior, not policy effectiveness. The specification's
equal-budget random-review comparison and human usability/color study remain unperformed research.
