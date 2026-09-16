# Rule panel

rule_panel generates shallow deterministic decision-tree rules explaining current SSDBCODI
cluster and outlier outputs. Its RuleSet is read-only model explanation, never model truth.

The former category recommendation engine and RecommendationPlan schema have been removed.
Six-score review belongs to active_learning/review.py and does not depend on tree thresholds,
purity, rule overlaps or exceptions.

Retain /modules/rule-panel/ for inspecting tree configuration, rules and fixtures.
Tests cover rule support, purity, stable IDs, conditions and cluster/outlier target semantics.
