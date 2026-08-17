# Flask Application

## App Factory

app.create_app(enabled_modules=None) creates the Flask app, registers retained module blueprints, registers the active-learning workflow when all dependencies are enabled, and then registers core routes.

The optional enabled_modules argument supports isolated module tests. Navigation therefore targets core proxy routes that remain valid when the product workflow is not mounted.

## Core Routes

| Route | Behavior |
| --- | --- |
| / | Redirect to the active-learning product |
| /health | Application health envelope |
| /modules/ | Retained module-lab index |
| /modules/<slug>/ | Module debug page |
| /modules/<slug>/health | Module health |
| /modules/<slug>/api/state | Module state summary |
| /workflows/ | Redirect to the active-learning product |

Unknown module and workflow slugs return the shared 404 envelope.

## Product Routes

| Route | Behavior |
| --- | --- |
| /workflows/active-learning-dashboard/ | Import, fixture, dataset, and session index |
| /workflows/active-learning-dashboard/import | Browser upload and session creation |
| /workflows/active-learning-dashboard/wine-fixture | Generic Wine fixture session |
| /workflows/active-learning-dashboard/<session_id>/ | Final session dashboard |

The JSON API is documented in state_and_api_contracts.md.

## Registry

app/module_registry.py lazily imports blueprints. A module entry contains slug, package name, title, purpose, status, and blueprint factory.

The single WorkflowInfo entry depends on every retained module. When create_app enables only a subset, the workflow is intentionally absent while module labs remain testable.

## Response Envelope

Successful APIs return:

~~~json
{
  "ok": true,
  "data": {},
  "error": null,
  "diagnostics": {}
}
~~~

Errors return ok=false and an error object with a stable code and message. Conflict responses use HTTP 409; validation errors use HTTP 400; unknown resources use HTTP 404.

## Runtime

run.py loads the local .env file, creates the app, and listens on port 5001 by default. Persistent active-learning data is resolved from METRIC_DASHBOARD_ACTIVE_LEARNING_DB_PATH relative to the repository root unless an absolute path is supplied.
