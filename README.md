# Metric Dashboard

Metric Dashboard is a local Flask application for rule-guided, multi-round active learning on structured tabular data.

The final product path combines:

- CSV, JSON, and MAT dataset import;
- deterministic preprocessing and stable point IDs;
- MDS projection and interactive point selection;
- SSDBCODI clustering with integrated outlier detection;
- shallow decision-tree rules that explain SSDBCODI output;
- deterministic next-point recommendations;
- deterministic category evidence in ordinary language;
- optional DeepSeek V4 Pro wording improvements;
- human label commits, round history, and SQLite persistence.

Wine is a regression fixture and demo dataset. It is not part of the recommendation logic, and its ground truth is never sent to the model or used to choose points.

## Core Loop

~~~text
import tabular data
  -> preprocess and project
  -> run SSDBCODI
  -> generate explanation-only rules
  -> build deterministic recommendation plans
  -> build fixed evidence checks and comparison records
  -> optionally rewrite that evidence with DeepSeek V4 Pro
  -> user labels recommended or selected records
  -> persist label events and create the next round
  -> rerun analysis and compare round changes
~~~

The same dataset version, configuration, label revision, and focus category
must produce the same ordered recommendation points and evidence statuses.
DeepSeek may vary the wording, but validation prevents it from changing the
points, evidence checks, statuses, fact references, or comparison records.

## Setup On macOS

~~~bash
conda create -n metric-dashboard python=3.9 -y
conda activate metric-dashboard
python -m pip install -r requirements.txt
cp .env.example .env
~~~

Set the DeepSeek key in .env when generated explanations are needed:

~~~text
METRIC_DASHBOARD_DEEPSEEK_API_KEY=your-key
~~~

The deterministic recommendation and labeling loop works without a key by using plain-language fallback guidance.

## Run

~~~bash
python run.py
~~~

Open http://127.0.0.1:5001. Port 5001 avoids the macOS services that commonly occupy port 5000. Override it when needed:

~~~bash
PORT=5002 python run.py
~~~

The product entry is:

~~~text
/workflows/active-learning-dashboard/
~~~

The root route and /workflows/ redirect there. /modules/ exposes isolated engineering labs for retained modules.

## Supported Data

The first product version accepts structured tabular data:

- numeric and categorical feature columns;
- missing values;
- optional point-ID and metadata columns;
- optional ground-truth columns isolated for offline evaluation.

Built-in adapters support CSV, JSON, and MAT. Preprocessing stores both the model matrix and a transformation map so rule cards can use original field names and units.

## Model Responsibilities

SSDBCODI owns cluster assignments, outlier flags, and per-point analysis scores.

The decision tree is a read-only surrogate. It converts current SSDBCODI output into feature rules but never performs clustering or outlier detection.

The recommendation engine owns candidate generation, ranking, filtering, history penalties, tie-breaking, and final ordered point IDs.

The deterministic Category Evidence Matrix owns the user-visible reasons for
each point. It checks a fixed category-specific checklist in the complete
feature space and exposes exact calculations only under `Technical details`.

DeepSeek V4 Pro receives a compact TranslationPacket and only rewrites supplied
facts into clearer language. A response is accepted only when the returned
model and immutable recommendation/evidence contract pass validation. A bad
bullet falls back locally; an API failure never blocks labeling.

## Persistence

SQLite stores dataset metadata, sessions, immutable rounds, label events, recommendation plans, and interpretation diagnostics. Larger raw and matrix artifacts are stored under runtime_data and referenced by fingerprint.

Label changes create superseding LabelEvents. Stale round, revision, or plan submissions return a conflict instead of overwriting current state.

## Tests

~~~bash
python -m unittest discover -s tests
python -m compileall app tests
git diff --check
~~~

Tests cover all retained module boundaries, generic dataset adapters,
multi-round state, deterministic recommendations and fixed evidence dimensions
across all categories, DeepSeek contract validation, and the integrated
workflow.

## Repository Map

~~~text
app/modules/active_learning/    session, round, data, persistence, translation
app/modules/rule_panel/         decision-tree rules and deterministic plans
app/modules/ssdbcodi/           clustering and integrated outlier detection
app/workflows/                  final active-learning dashboard
app/shared/                     cross-module schemas and DeepSeek client
prompts/active_learning/        constrained explanation prompt
docs/                           current architecture and contracts
tests/                          module and workflow regression coverage
~~~
