# Development Process

## 1. Core Rule

Build the project as a local Flask app from the beginning.

Every module must be independently testable in code and independently visible in Flask.

Unit tests are required, but they are not enough. For a dashboard project, each module also needs a browser-visible debug page that shows whether the module is behaving correctly.

## 2. Standard Module Development Loop

For every module, follow this loop:

1. Read:
   - `README.md`
   - `docs/overview.md`
   - `docs/flask_app.md`
   - `docs/modules/<module_name>/design.md`
   - `docs/module_debug_checklist.md`
   - `docs/integration_testing.md`
   - `docs/state_and_api_contracts.md`

2. Define contracts:
   - schemas
   - service inputs
   - service outputs
   - Flask API responses

3. Build pure module logic:
   - `service.py`
   - `schemas.py`
   - small helpers

4. Add fixtures:
   - predictable sample data
   - classic datasets when helpful
   - mock downstream outputs when the real downstream module is not ready

5. Add unit tests:
   - valid inputs
   - invalid inputs
   - edge cases
   - deterministic behavior

6. Add Flask module page:
   - `routes.py`
   - `templates/<module_name>/index.html`
   - `static/<module_name>/...` if needed

7. Add Flask API endpoints:
   - health endpoint
   - state endpoint
   - action endpoint if interactive

8. Add Flask smoke tests:
   - module page returns 200.
   - module API returns expected JSON.

9. Open the page locally:
   - inspect the visible output.
   - confirm the page communicates what is real and what is mocked.

10. Update docs:
   - design notes
   - route list
   - manual check instructions

## 3. Definition of Done for a Module

A module is done only when all of these are true:

1. It has documented input and output contracts.
2. It has pure service logic that can be tested without Flask.
3. Unit tests pass.
4. It has a Flask debug page under `/modules/<module_name>/`.
5. It has at least one JSON/debug API endpoint.
6. Flask route smoke tests pass.
7. A developer can open the module page in the browser and visually inspect behavior.
8. The module does not import unrelated module internals.
9. Any integration with other modules happens through schemas, services, APIs, or orchestrated workflows.
10. Its design document is updated.

## 4. Global Test Commands

Use these during development:

```powershell
python -m compileall app tests
python -m unittest discover -s tests
python run.py
```

After `python run.py`, manually inspect the relevant pages:

```text
http://127.0.0.1:5000/
http://127.0.0.1:5000/modules/
http://127.0.0.1:5000/modules/<module_name>/
```

## 5. Build Order

### Step 0: Flask App Shell and Module Lab

Build first:

```text
dashboard_shell
module registry
module lab index
base templates
base CSS
```

Why:

Every later module needs a place to appear visually. The Flask app shell should exist before module work continues.

Tasks:

1. Create `run.py`.
2. Create `create_app()`.
3. Create `/`, `/health`, `/modules/`, and `/workflows/`.
4. Create module registry.
5. Create a simple base template and shared CSS.
6. Add a placeholder card for every planned module.

Unit tests:

1. App factory returns a Flask app.
2. `/health` returns OK.
3. `/modules/` returns 200.

Flask visual check:

Open `/modules/` and confirm all module cards are visible.

Completion:

The local Flask app runs and can host module pages.

---

### Step 1: Data Workspace

Build:

```text
shared schemas
data_workspace
data workspace Flask page
```

Why:

All later modules need stable point IDs, feature vectors, and dataset state.

Tasks:

1. Define `Point`, `Dataset`, and `FeatureMatrix`.
2. Normalize raw input into point objects.
3. Generate stable IDs if missing.
4. Create classic dataset fixtures, such as Iris sample.
5. Add `/modules/data-workspace/`.
6. Add `/modules/data-workspace/api/dataset`.
7. Show dataset table and feature matrix preview in Flask.

Unit tests:

1. Load fixture data.
2. Reject empty input.
3. Reject missing features.
4. Reject duplicate IDs.
5. Preserve point order.
6. Return valid feature matrix.

Flask visual check:

Open `/modules/data-workspace/` and confirm:

1. point IDs are visible.
2. feature names are visible.
3. feature matrix preview matches the fixture.
4. JSON API link works.

Completion:

The dataset can be inspected in the browser and consumed by later modules.

---

### Step 2: Projection

Build:

```text
projection
MDS service
projection Flask page
data-projection workflow page
```

Why:

The scatterplot needs 2D coordinates. Projection should be visually inspected before scatterplot work.

Tasks:

1. Implement MDS behind a service function.
2. Input `FeatureMatrix`.
3. Output `ProjectionResult`.
4. Add `/modules/projection/`.
5. Add `/modules/projection/api/projection`.
6. Render an SVG scatterplot from fixture data.
7. Add `/workflows/data-projection/` to show data table and projection together.

Unit tests:

1. Every input point gets one coordinate.
2. Coordinates are finite.
3. Projection is deterministic for fixed input.
4. Invalid matrix is rejected.

Flask visual check:

Open `/modules/projection/` and confirm:

1. points appear in SVG.
2. coordinate table is visible.
3. Iris sample visually separates in a plausible way.
4. JSON API returns projection coordinates.

Completion:

Projection can be inspected through Flask without scatterplot, clustering, or chatbox.

---

### Step 3: Algorithm Adapters

Build:

```text
algorithm_adapters
clustering adapter
outlier adapter
adapter Flask page
```

Why:

The app must show default clusters and outliers, but existing algorithms should stay isolated behind adapters.

Tasks:

1. Wrap existing clustering algorithm.
2. Wrap existing outlier detection algorithm.
3. Convert dashboard schemas to algorithm input.
4. Convert algorithm output back to point-ID-based schemas.
5. Add `/modules/algorithm-adapters/`.
6. Add APIs for cluster and outlier output.
7. Show adapter diagnostics in Flask.

Unit tests:

1. Adapter passes expected matrix or representation to algorithms.
2. Cluster assignments map to known point IDs.
3. Outlier scores map to known point IDs.
4. Invalid algorithm output is rejected.

Flask visual check:

Open `/modules/algorithm-adapters/` and confirm:

1. cluster assignments are visible.
2. outlier scores are visible.
3. diagnostics show which algorithm is being called.
4. page clearly marks mock outputs if real algorithms are not connected yet.

Completion:

Default cluster and outlier results can be inspected in Flask.

---

### Step 4: Selection

Build:

```text
selection
selection store
selection Flask page
```

Why:

Selection is the grounding layer for phrases like "these points" and "unselected points".

Tasks:

1. Store selected point IDs.
2. Support select, deselect, replace, and clear.
3. Return selected and unselected IDs.
4. Add `/modules/selection/`.
5. Add selection API endpoints.
6. Build a clickable point list in Flask.

Unit tests:

1. Select known points.
2. Reject unknown IDs.
3. Clear selection.
4. Return selected/unselected IDs.

Flask visual check:

Open `/modules/selection/` and confirm:

1. clicking points changes selection state.
2. selected/unselected JSON updates.
3. clear selection works.

Completion:

Selection can be exercised in Flask without scatterplot or chatbox.

---

### Step 5: Scatterplot

Build:

```text
scatterplot
scatterplot Flask page
scatter-selection workflow page
```

Why:

After data, projection, adapter output, and selection exist, scatterplot can combine them visually.

Tasks:

1. Render projected points.
2. Color points by cluster.
3. Mark outliers.
4. Support selection interaction.
5. Add `/modules/scatterplot/`.
6. Add `/workflows/scatter-selection/`.

Unit tests:

1. Build render payload correctly.
2. Preserve point IDs.
3. Mark selected points correctly.
4. Include cluster and outlier fields.

Flask visual check:

Open `/modules/scatterplot/` and confirm:

1. points render.
2. cluster colors are visible.
3. outlier markers are visible.
4. clicking points updates selection state.

Completion:

The scatterplot module can be visually tested before chatbox work.

---

### Step 6: Chatbox

Build:

```text
chatbox
chatbox Flask page
mock selection context
```

Why:

Chatbox needs selection context, but should not own selection or run algorithms.

Tasks:

1. Build chat UI.
2. Display current selection context.
3. Submit user message.
4. Show assistant response.
5. Add `/modules/chatbox/`.
6. Add API endpoint for message submission.
7. Support mock selection context for standalone testing.

Unit tests:

1. Empty messages are rejected.
2. Message payload includes selection context.
3. Chatbox does not call clustering or outlier detection.

Flask visual check:

Open `/modules/chatbox/` and confirm:

1. chat input works.
2. message appears in history.
3. selection context is visible.
4. response clearly shows whether intent parsing is real or mocked.

Completion:

Chatbox can be manually tested in Flask with mock selection.

---

### Step 7: Intent Instruction

Build:

```text
intent_instruction
intent Flask page
chat-intent workflow page
```

Why:

User language must become structured instructions before metric learning is touched.

Tasks:

1. Define structured instruction schema.
2. Implement deterministic classifier first.
3. Resolve selected/unselected references.
4. Generate clarification requests.
5. Add `/modules/intent-instruction/`.
6. Add `/workflows/chat-intent/`.

Unit tests:

1. Grouping messages become `same_class`.
2. Split messages become `split_into_n_classes`.
3. Outlier messages become `is_outlier`.
4. Vague messages require clarification.
5. Irrelevant messages become `non_actionable`.

Flask visual check:

Open `/modules/intent-instruction/` and confirm:

1. example messages can be submitted.
2. structured instruction JSON is visible.
3. clarification cases are clear.

Completion:

Intent parsing is visible and debuggable before metric-learning integration.

---

### Step 8: Metric-Learning Adapter

Build:

```text
metric_learning_adapter
constraint preview Flask page
```

Why:

Structured instructions should be converted into metric-learning constraints through one narrow boundary.

Tasks:

1. Accept structured instruction.
2. Reject incomplete or non-actionable instruction.
3. Convert actionable instruction into constraint payload.
4. Add `/modules/metric-learning-adapter/`.
5. Show instruction input and constraint output.

Unit tests:

1. `same_class` creates must-link constraints.
2. `different_class` creates cannot-link constraints.
3. `split_into_n_classes` creates split constraints.
4. incomplete instruction is rejected.

Flask visual check:

Open `/modules/metric-learning-adapter/` and confirm:

1. sample instruction produces visible constraints.
2. invalid instruction produces clear error.

Completion:

Metric-learning input can be inspected before real refinement loop.

---

### Step 9: Refinement Orchestrator

Build:

```text
refinement_orchestrator
refinement timeline Flask page
```

Why:

The orchestrator coordinates modules but should not contain their internal logic.

Tasks:

1. Receive structured instruction.
2. Call metric-learning adapter.
3. Trigger updated projection.
4. Rerun clustering and outlier adapters.
5. Return updated dashboard state.
6. Add `/modules/refinement-orchestrator/`.
7. Add `/workflows/refinement-loop/`.

Unit tests:

1. Actionable instruction triggers the flow.
2. incomplete instruction stops early.
3. failures return diagnostics.

Flask visual check:

Open `/modules/refinement-orchestrator/` and confirm:

1. timeline shows each step.
2. intermediate payloads are visible.
3. failure and success states are understandable.

Completion:

The update loop can be debugged visually before full dashboard integration.

---

### Step 10: Integrated Dashboard

Build:

```text
integrated dashboard
```

Why:

Only integrate after individual modules and workflow demos work.

Tasks:

1. Compose data, projection, adapters, scatterplot, selection, chatbox, intent, metric learning, and orchestration.
2. Keep dashboard shell thin.
3. Show current state clearly.
4. Make refinement loop visible.

Unit tests:

1. Integrated route returns 200.
2. APIs return coherent state.
3. irrelevant chat does not trigger refinement.
4. actionable chat triggers expected flow.

Flask visual check:

Open `/` and confirm:

1. scatterplot appears.
2. selecting points updates chatbox context.
3. chat instruction produces structured output.
4. valid instruction updates the visible state.

Completion:

The first complete local human-in-the-loop workflow works in Flask.

## 6. What Not To Do Early

1. Do not add deployment infrastructure.
2. Do not add a heavy frontend framework.
3. Do not add a production database.
4. Do not let modules communicate through hidden globals.
5. Do not let chatbox call algorithms directly.
6. Do not let scatterplot own selection truth.
7. Do not rewrite existing clustering or outlier logic.
8. Do not skip browser-visible module pages.

## 7. Milestones

### Milestone 1: Local Module Lab

Goal:

Flask app runs and lists all modules under `/modules/`.

### Milestone 2: Data and Projection Visible

Goal:

Data workspace and projection can be opened in Flask, and `/workflows/data-projection/` shows their interaction.

### Milestone 3: Visual Inspection and Selection

Goal:

Scatterplot renders projected points, default clusters, outliers, and selection state.

### Milestone 4: Chat and Intent

Goal:

Chatbox receives selection context and intent module outputs structured instructions.

### Milestone 5: Refinement Loop

Goal:

Structured instruction flows through metric-learning adapter and refinement orchestrator.

### Milestone 6: Integrated Dashboard

Goal:

The full local dashboard supports one complete refinement cycle.
