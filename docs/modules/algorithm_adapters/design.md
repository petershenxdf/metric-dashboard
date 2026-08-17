# Algorithm Adapters Design

## Role

algorithm_adapters provides the stable AnalysisResult boundary consumed by the dashboard. The active provider is SSDBCODI.

The adapter keeps projection, rule generation, recommendations, and active-learning persistence independent from SSDBCODI implementation details.

## Output

AnalysisResult contains:

- analysis_run_id;
- ClusterResult assignments;
- OutlierResult per-point flags and scores;
- provider diagnostics and execution order.

Cluster and outlier results must cover the same known point universe according to their schema rules.

## Current Provider

run_default_analysis uses the SSDBCODI provider. Bootstrap clustering supplies deterministic initial normal seeds; SSDBCODI then performs the integrated semi-supervised analysis.

Manual active-learning labels are passed through the adapter boundary. The provider does not read UI selection or call DeepSeek.

## Invariants

- SSDBCODI remains the analysis source of truth;
- provider identity and parameters contribute to analysis identity;
- invalid cluster counts and unsupported dataset sizes fail explicitly;
- adapters do not generate explanation rules or choose recommendation points;
- no silent sampling is allowed.

## Debug Surface

~~~text
/modules/algorithm-adapters/
/modules/algorithm-adapters/health
/modules/algorithm-adapters/api/analysis
/modules/algorithm-adapters/api/clusters
/modules/algorithm-adapters/api/outliers
~~~

## Tests

Tests cover deterministic results, parameter validation, manual-label integration, provider diagnostics, and route envelopes.
