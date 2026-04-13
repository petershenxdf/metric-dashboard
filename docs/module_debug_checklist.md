# Module Debug Page Checklist

Use this checklist for every module page under `/modules/<module_name>/`.

## Required Page Content

1. Module name.
2. Module status: missing, stubbed, working, or integrated.
3. What data is real and what data is mocked.
4. Main visible output.
5. JSON/state preview.
6. Links to relevant API endpoints.
7. Links to related workflow pages.

## Required API Content

Each module should have:

```text
/modules/<module_name>/health
```

Each module should also have at least one state or primary data API. The exact route can match the module purpose, for example:

```text
/modules/data-workspace/api/dataset
/modules/projection/api/projection
/modules/selection/api/state
```

If the module performs actions, add action APIs:

```text
/modules/<module_name>/api/<action>
```

Examples:

```text
/modules/selection/api/select
/modules/chatbox/api/messages
/modules/intent-instruction/api/compile
```

## Required Tests

Each module should have:

1. service tests.
2. route smoke tests.
3. API response shape tests.
4. at least one documented manual browser check.

## Manual Browser Check

After starting Flask:

```powershell
python run.py
```

Open:

```text
http://127.0.0.1:5000/modules/<module_name>/
```

Confirm:

1. page loads.
2. fixture data appears.
3. main output is visible.
4. JSON/state panel matches expected state.
5. interactions work if the module is interactive.
