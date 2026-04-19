---
name: Template tojson pattern
description: In Flask/Jinja templates, avoid `map(attribute='to_dict') | list | tojson` — it serializes bound methods, not dicts. Build the list in the route.
type: feedback
---

Do not write `{{ items | map(attribute='to_dict') | list | tojson(indent=2) }}` in Jinja templates in this repo. The `map(attribute=...)` filter returns the attribute object, which for dataclasses with a `to_dict` *method* is the bound method itself — `tojson` then raises `TypeError: Object of type method is not JSON serializable`.

**Why:** Hit during Step 7 (chatbox). `selection_groups | map(attribute='to_dict') | list | tojson` and the same pattern for `chips` both crashed the workflow template with a 500.

**How to apply:** When a template needs to dump a list of dataclasses as JSON, dictify in the route and pass a `*_payload` list:

```python
return render_template(..., chips_payload=[chip.to_dict() for chip in chips])
```

```jinja
<pre>{{ chips_payload | tojson(indent=2) }}</pre>
```

Matches the existing pattern in `labeling/index.html` (`state.to_dict().annotations | tojson`) and `selection_labeling.html` workflow template.
