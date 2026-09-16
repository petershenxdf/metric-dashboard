# Architecture overview

One Flask product workflow connects import, SSDBCODI, six-score review and human feedback.
The app uses server-rendered Jinja HTML with vanilla JavaScript and an SVG projection; no build step.

## Ownership

| Component | Responsibility |
| --- | --- |
| active_learning/data.py | CSV/JSON/MAT, raw/model separation, stable identities and role validation |
| ssdbcodi | model truth, density-safe bootstrap, confirmed-label constraints, full expansion paths |
| algorithm_adapters | shared analysis boundary |
| projection | stable MDS display coordinates |
| rule_panel | explanation-only shallow trees |
| active_learning/review.py | six independent raw scores, frozen eligible percentile pools, recommendation gates |
| active_learning/service.py | complete feedback/rerun loop and round snapshots |
| active_learning/store.py | SQLite and artifacts; atomic label commits; branch history |
| workflow and review_matrix.js | routes, linked matrix, filtering and explicit human review |

There is no external language model, old eight-category plan or cross-category meta-ranker.
Read [the active-learning design](modules/active_learning/design.md) for algorithms and defaults,
and [state/API contracts](state_and_api_contracts.md) for integration.
