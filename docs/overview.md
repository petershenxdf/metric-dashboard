# Project Overview and Top-Level Design

## 1. Goal

Build a local Flask dashboard for human-in-the-loop clustering and anomaly
explanation.

The app should let a user:

1. Load or use a sample dataset.
2. Project data into 2D with MDS.
3. View default clustering and outlier detection results.
4. Select points in a scatterplot.
5. Assign selected points to a cluster/class or mark them as outliers.
6. Inspect generated rules that explain each cluster and anomaly group.
7. Use DeepSeek to turn the generated rules into categorized label/refinement recommendations.
8. Compare rule recommendations with the visible scatterplot and SSDBCODI scores.

Deployment is out of scope for now. The app only needs to run locally.

## 2. Flask-First Architecture

The project should be one simple Flask app with a module lab.

The module lab is a set of local pages that expose each module independently. This is required because unit tests alone are not enough for a dashboard project.

Every module should have:

1. A service layer for pure logic.
2. A Flask route layer for visible local testing.
3. Fixtures or mock data for standalone demos.
4. API endpoints for inspecting internal state.
5. Clear contracts for integration with other modules.

## 3. Route Model

```text
/                                      integrated dashboard
/health                                health check
/modules/                              module lab index
/modules/<module_name>/                module debug page
/modules/<module_name>/api/...         module debug/state APIs
/workflows/<workflow_name>/            multi-module interaction demo
```

The module debug page should answer the question: "Is this module working correctly enough to trust it before integration?"

The workflow page should answer the question: "Do these modules interact correctly when combined?"

## 4. Core User Flow

The current direction after Step 8.5 is explanation-first. The old branching
update roadmap is no longer active.

```text
data workspace
  -> projection
  -> algorithm adapters
  -> scatterplot
  -> selection
  -> labeling / annotation
  -> rule panel
      -> decision-tree surrogate rules over SSDBCODI cluster labels
      -> decision-tree surrogate rules over SSDBCODI anomaly labels
      -> rule cards with thresholds, support, coverage, purity, exceptions
  -> DeepSeek rule interpretation
      -> next label priorities
      -> overlap / merge hypotheses
      -> split or new-cluster hypotheses
      -> anomaly and exception label review
      -> feature-threshold labeling strategy
```

Detailed flow:

1. Data workspace creates a dataset with stable point IDs.
2. Projection computes 2D coordinates with MDS.
3. Algorithm adapters call clustering and outlier detection through replaceable providers.
   The current default provider is SSDBCODI, which emits cluster assignments and
   outlier flags through the same dashboard schemas. The old LOF-then-KMeans
   provider remains available explicitly for comparison.
4. Scatterplot renders points with cluster colors and outlier markers.
5. User selects points through clicks, lasso, rectangle, API calls, or future selection gestures.
6. Selection module stores selected/unselected state, can save reusable named selection groups, and exposes reusable selection context.
7. Labeling module converts direct label actions into manual annotations or structured feedback instructions.
8. The legacy chatbox and intent runtime remain as the completed Step 7/8/8.5
   foundation, but they are no longer the next product surface.
9. Step 8.6 introduces a `rule_panel` that trains shallow decision-tree
   surrogates against current SSDBCODI cluster and anomaly outputs. These trees
   only extract explanatory rules; they do not perform clustering or outlier
   detection.
10. The rule panel renders each tree path as a rule card: feature thresholds,
   target cluster/anomaly, support, coverage, purity, matched points, and
   exception points.
11. Step 8.7 uses DeepSeek to parse generated rule cards into categorized
   label guidance: which points to label, why those points need checking, and
   how to label them. It does not parse free-form update requests.
12. `/workflows/wine-dashboard/` is the current Step 8.8 integrated rule
   dashboard: it uses `wine.mat` and combines data, projection, SSDBCODI,
   selection, labeling, scatterplot rendering, rule cards, and explicit
   DeepSeek/mock interpretation status without the chatbox UI.

## 5. Product Constraints

1. Existing clustering and outlier detection algorithms must not be redesigned.
2. Chatbox must not directly perform clustering or outlier detection.
3. Scatterplot must not parse language.
4. Selection state must be accessible outside the scatterplot.
5. Labeling state must be owned by the labeling module, not by scatterplot or chatbox.
6. LLM output and decision-tree rules must not become a source of clustering,
   anomaly, or labeling truth.
7. All cross-module contracts should use shared schemas.
8. Every module must be independently visible in Flask.

## 6. Target File Structure

```text
metric-dashboard/
  run.py
  requirements.txt
  README.md

  app/
    __init__.py
    module_registry.py
    routes.py

    shared/
      schemas.py
      flask_helpers.py
      fixtures.py
      request_helpers.py
      effective_analysis.py

    templates/
      base.html
      home.html
      modules_index.html
      workflows_index.html

    static/
      app.css
      app.js

    modules/
      dashboard_shell/
        routes.py
        service.py
        templates/dashboard_shell/
        static/dashboard_shell/

      data_workspace/
        schemas.py
        service.py
        fixtures.py
        routes.py
        templates/data_workspace/
        static/data_workspace/

      projection/
        schemas.py
        mds.py
        service.py
        fixtures.py
        routes.py
        templates/projection/
        static/projection/

      algorithm_adapters/
        schemas.py
        clustering.py
        outliers.py
        service.py
        fixtures.py
        routes.py
        templates/algorithm_adapters/

      selection/
        schemas.py
        store.py
        service.py
        fixtures.py
        routes.py
        templates/selection/
        static/selection/ optional for larger debug-page interactions

      labeling/
        schemas.py
        service.py
        fixtures.py
        routes.py
        templates/labeling/
        static/labeling/ optional for larger debug-page interactions

      scatterplot/
        schemas.py
        service.py
        fixtures.py
        routes.py
        templates/scatterplot/
        static/scatterplot/

      ssdbcodi/
        algorithm.py
        schemas.py
        service.py
        store.py
        fixtures.py
        routes.py
        templates/ssdbcodi/

      chatbox/                         legacy Step 7/8 intake surface
        schemas.py
        service.py
        fixtures.py
        routes.py
        templates/chatbox/
        static/chatbox/

      intent_instruction/
        schemas.py
        router.py
        extractor.py
        service.py
        store.py
        memory.py
        drafts.py
        evaluation.py
        providers/
          base.py
          mock.py
          ollama.py
        fixtures.py
        routes.py
        templates/intent_instruction/

      rule_panel/
        schemas.py
        decision_tree_rules.py
        service.py
        fixtures.py
        routes.py
        templates/rule_panel/

    workflows/
      fixtures.py               re-exports from app.shared.fixtures
      effective_analysis.py      re-exports from app.shared.effective_analysis
      data_projection.py
      default_analysis.py
      selection_context.py
      selection_labeling.py
      analysis_selection.py
      analysis_labeling.py
      scatter_selection.py
      scatter_labeling.py
      provider_feedback.py
      chat_selection.py
      chat_intent.py
      intent_runtime_validation.py     Step 8.5 live-model validation gate
      rule_panel_validation.py         decision-tree rule cards
      rule_interpretation.py           categorized DeepSeek rule parsing

  tests/
    modules/
      data_workspace/
      projection/
      algorithm_adapters/
      selection/
      labeling/
      scatterplot/
      ssdbcodi/
      chatbox/
      intent_instruction/
      rule_panel/
    flask_app/
      test_module_pages.py
      test_workflow_pages.py

  docs/
    overview.md
    flask_app.md
    process.md
    module_debug_checklist.md
    integration_testing.md
    state_and_api_contracts.md
    modules/
      <module_name>/
        design.md
```

This is the target structure. The project can migrate toward it gradually.

## 7. Module Contract

Each module should expose these boundaries where applicable:

1. `service.py`
   - pure logic.
   - independently unit tested.

2. `schemas.py`
   - module-local schemas or re-exports of shared schemas.

3. `fixtures.py`
   - sample data for tests and Flask module demo.

4. `routes.py`
   - Flask blueprint for module debug page and APIs.

5. `templates/<module_name>/`
   - HTML for visible local testing.

6. `static/<module_name>/`
   - small module-specific JavaScript or CSS when needed.

## 8. Module List

| Module | Main Job | Flask Debug Page |
| --- | --- | --- |
| Dashboard Shell | App factory, module registry, integrated pages | `/`, `/modules/`, `/workflows/` |
| Data Workspace | Dataset identity and feature matrix | `/modules/data-workspace/` |
| Projection | MDS 2D coordinates | `/modules/projection/` |
| Algorithm Adapters | Active SSDBCODI provider boundary plus legacy LOF/KMeans comparison provider | `/modules/algorithm-adapters/` |
| Selection | Selected/unselected point state | `/modules/selection/` |
| Labeling | Manual point annotations, cluster labels, and outlier labels | `/modules/labeling/` |
| Scatterplot | Visual point rendering and selection UI | `/modules/scatterplot/` |
| SSDBCODI | Active semi-supervised density-based clustering with integrated outlier detection and score diagnostics | `/modules/ssdbcodi/` |
| Chatbox | Legacy dialogue UI from the original Step 7/8 direction | `/modules/chatbox/` |
| Intent Instruction | Legacy router/extractor and provider runtime; reusable for rule-interpretation provider calls | `/modules/intent-instruction/` |
| Rule Panel | Decision-tree rule cards for current clusters and anomalies, plus categorized DeepSeek rule interpretation | `/modules/rule-panel/` |

## 9. Rule Explanation Families

The new post-Step-8.5 direction treats current SSDBCODI outputs as the thing to
explain. A rule generator trains shallow decision-tree surrogates over the
feature matrix using the current SSDBCODI cluster labels and anomaly flags as
targets. The trees are rule extractors only; SSDBCODI remains the clustering
and anomaly-detection source.

### Rule Targets

The rule panel emits two rule families:

1. Cluster rules - multi-class or one-vs-rest paths that explain a target
   `cluster_id`.
2. Anomaly rules - binary anomaly-vs-normal paths that explain current outlier
   flags and SSDBCODI score patterns.

The current Step 8.6 visual/test fixture is the uploaded `wine.mat` dataset, so
rule conditions should display raw wine feature names rather than projected
coordinate names.

Each rule card should show:

1. target kind and target id,
2. feature-threshold conditions,
3. support count,
4. coverage,
5. purity,
6. matched point ids,
7. exception point ids.

### DeepSeek Interpretation Categories

DeepSeek parses generated rules into label/refinement guidance categories:

1. `label_priority` - which points or rule regions the user should label first.
2. `boundary_review` - where neighboring rule regions need boundary labels.
3. `overlap_merge_signal` - whether sample-level overlap supports merge/shared-boundary review.
4. `split_or_new_cluster_signal` - whether separated or weak regions suggest split/new-cluster review.
5. `anomaly_label_review` - which outlier-rule points need true-anomaly vs normal-member labels.
6. `exception_relabel_review` - which exception points should be relabeled or audited.
7. `feature_label_strategy` - which raw feature thresholds should guide manual labels.
8. `rule_confidence_audit` - whether support, coverage, purity, and exceptions are strong enough for refinement.

Step 8.7 is implemented at `/workflows/rule-interpretation/`. It builds a
compact rule payload with `rule_guidance_metrics` and
`label_candidate_point_profiles`, validates the returned `RuleInterpretation`,
and rejects
unknown categories, rule ids, raw features, thresholds, target ids, and point
ids. The workflow includes one focus button per category; each button sends
`focus_category` and returns that category's user-facing label guidance. The
output must include `category_explanation`, `label_targets`,
`suspicion_reasons`, `point_label_guidance`, `recommendation`,
`quantitative_findings`, and `suggested_label_actions`. Local debug mode uses
the deterministic mock provider by default; `provider_kind=deepseek` uses
DeepSeek V4 Pro (`deepseek-v4-pro`) with thinking enabled and high reasoning
effort.

The integrated Step 8.8 dashboard shows these categories as separate selectable
lenses. A category should only recommend points when the current rule cards
contain a typical case for that lens; otherwise it should say that no typical
case is available. User-facing interpretation should be plain-language label
guidance: concrete point ids, why those points are suspicious in human terms,
and how different labels would affect merge, boundary, split/new-cluster,
anomaly, or rule-audit decisions. Metrics remain audit evidence, not the main
explanation.

### Legacy Instruction Work

The original structured-instruction work remains useful as provider/prompt
infrastructure, but old post-Step-8.5 update-strategy adapters are superseded
by the rule-panel direction.

Example delta:

```json
{
  "operations": [
    {
      "op": "add",
      "constraint": {
        "id": "c2",
        "intent": "group_similar",
        "group_a": {"source": "selection_group", "ref": "group_001"},
        "group_b": {"source": "cluster", "ref": "cluster_2"}
      }
    }
  ]
}
```

## 10. Integration Strategy

Do not wait until the end to use Flask.

Each step should add:

1. Pure service behavior.
2. Unit tests.
3. Flask module page.
4. Flask API endpoint.
5. A small integration or workflow page when the module has a neighbor to interact with.

The final dashboard should be built by composing already visible modules.

## 11. Current Working Slice

The currently implemented working slice is:

```text
data_workspace
  -> projection
  -> algorithm_adapters
  -> selection
  -> labeling
  -> scatterplot
```

Browser checks:

```text
/modules/data-workspace/
/modules/projection/
/modules/algorithm-adapters/
/modules/selection/
/workflows/data-projection/
/workflows/default-analysis/
/workflows/selection-context/
/workflows/selection-labeling/
/workflows/analysis-selection/
/workflows/analysis-labeling/
/modules/scatterplot/
/workflows/scatter-selection/
/workflows/scatter-labeling/
/workflows/provider-feedback/
```

The grouped workflow contract lives in `docs/workflows.md`. Use that document
when adding, reordering, or renaming workflow pages.

`/workflows/default-analysis/` uses the `default_analysis_outlier_debug` fixture
so outliers are visible during local debugging. It should not be interpreted as
the final user dataset flow.

`/workflows/selection-context/` uses a selection debug fixture to show how stable
point IDs become selected/unselected context for downstream labeling, chatbox,
and intent modules.

The selection module also supports saved selection groups. These are named point
sets for quickly restoring a previous selection; they are intentionally separate
from semantic labels, which remain the labeling module's responsibility.

`/workflows/analysis-selection/` connects the Step 1-4 path on one shared
fixture: Data Workspace creates point IDs and features, Projection computes MDS
coordinates, Algorithm Adapters mark clusters/outliers, and Selection overlays
active and saved selections on the same SVG plot. It includes a dataset dropdown,
click selection, and rectangle selection. New clicks or rectangle selections are
added to the active selection so the user does not need to choose a selection mode.

`/workflows/selection-labeling/` shows the Step 5 boundary: selection context
is consumed by labeling, and manual annotations are converted into structured
feedback instructions without involving chatbox or rule interpretation.

`/workflows/analysis-labeling/` is the full Step 1-5 visual test page. It uses
the same projection, SSDBCODI clustering/outlier detection, additive
click/rectangle selection, saved selection groups, and labeling controls on one
shared point-ID fixture. Manual labels are limited to `cluster_1...cluster_n`
and `outlier`; they are passed into SSDBCODI and update the effective
cluster/outlier state used by the frontend while baseline outputs remain
available in the state API.

`/modules/scatterplot/` is the Step 6 module page. It turns already-computed
projection, analysis, selection, and labeling state into a render payload and
visible SVG plot without owning selection or label truth. `/workflows/scatter-selection/`
and `/workflows/scatter-labeling/` verify those boundaries with selection and
labeling connected. The Step 1-6 workflow preserves prior interaction behavior:
click selection, rectangle selection, saved selection groups, adjustable
`n_clusters`, and manual cluster/outlier labeling.

`/workflows/provider-feedback/` is the Step 6.5 provider diagnostics page. It
checks that `algorithm_adapters.run_default_analysis()` resolves to SSDBCODI
while the standalone SSDBCODI result still exposes seed records and per-point
scores for rule-panel diagnostics.

`/modules/ssdbcodi/` is the dedicated module page for the SSDBCODI algorithm
([arXiv:2208.05561](https://arxiv.org/abs/2208.05561)). It is the active
clustering/outlier provider behind `algorithm_adapters` and keeps a dedicated
debug page for inspecting its scores. Bootstrap behavior:
the module computes density-safe KMeans center seeds (default `k = 3`,
user-configurable) so obvious far outliers are not promoted to normal seeds.
Those bootstrap seeds stay active as reusable seed inputs across runs;
however, under the current weighted-distance assignment rule a bootstrap
seed point's final `cluster_id` is recomputed each run and can shift to
another class. Only manual cluster annotations are output locks for their
own points. The debug page uses the same selection behavior
as Step 1-6: click and rectangle selection add to the active selection,
selected points use black center dots, and saved selection groups are restored
through the selection module. Label controls are limited to
`cluster_1...cluster_n` plus `outlier`; label actions save pending feedback,
while Run and Store recomputes and persists SSDBCODI. Per-point intermediate
scores (`rScore`, `lScore`, `simScore`, `tScore`) are persisted in
`SsdbcodiStore` for downstream rule cards and LLM evidence. The page also includes
`demo`, `moons`, and `circles` fixtures for browser testing different shapes.
See `docs/modules/ssdbcodi/design.md` for the full contract.
