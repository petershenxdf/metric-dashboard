# Deterministic six-score review

Recommendations are computed locally from the saved dataset, model settings and effective human labels.
All six raw scores and their eligible midrank percentile pools are frozen before sorting or filtering.
Each category supplies its own recommendation gate; no weighted category total exists.

Stable point IDs break exact ranking ties. Each recommended batch contains at most one representative
per identical transformed feature vector. Previously confirmed points require a specific model-status
or member-set change to re-enter the review pool. Unsure causes a one-round deferral, not a reference.

Full paths on the recorded deterministic mutual-reachability expansion tree supply Weak Group
Reachability. Model Instability compares complete fixed-input SSDBCODI runs at nearby MinPts values
using actual member sets, so cluster renaming does not create disagreement.

No request is sent to a language model. The old category plans, translator and interpretation cache
APIs are removed. Existing stored legacy tables remain untouched but unused.

See [the scoring design](modules/active_learning/design.md) and [contracts](state_and_api_contracts.md).
The existing PDF is historical, not the current algorithm specification.
