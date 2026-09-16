# Six review categories

The dashboard implements Cluster Assignment Conflict, Label Coverage Gap, Weak Group Reachability,
Local Sparsity, Known-Outlier Similarity and Model Instability.

See [the complete scoring design](modules/active_learning/design.md) for raw formulas, availability,
midrank percentile pools, full expansion paths, nearby-MinPts reruns and recommendation gates.
See [the user workflow](../README.md#review) for the linked matrix controls.

Each column supplies an independent reason to ask for human feedback. No combined utility score,
old rule-category meta-ranker or DeepSeek interpretation is used. NA is not zero; percentile is
not probability. A high score alone does not override the support or local-match checks.
The similarly named PDF is historical and predates the six-score implementation.
