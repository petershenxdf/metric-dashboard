# Development and review process

1. Import structured data with disjoint feature, metadata, ID and evaluation-only roles.
2. Start a session with a valid MinPts and optional label budget.
3. Run SSDBCODI and explanation-only rules.
4. Calculate all six independent review scores, including complete nearby-MinPts reruns.
5. Freeze raw scores, eligible percentile pools, evidence and model results in the round.
6. Sort/filter the matrix without recalculating scores; inspect linked point comparisons.
7. Save explicit human semantic/normal/outlier/unsure feedback.
8. Compute and atomically persist the next label-to-round transition.
9. Inspect history or return to a previous branch as needed.

Keep modules independent of workflow code. Tests must cover deterministic formulas, exact ties,
NA, missing-reference rules, ground-truth isolation, duplicate suppression and feedback persistence.
Do not reintroduce category meta-ranking or LLM judgments.
