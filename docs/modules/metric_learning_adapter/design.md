# Archived Update Adapter Design

This file is intentionally reduced to an archive marker.

The previous two-branch update design is not part of the active roadmap. Do not
implement this module unless the roadmap is explicitly reopened.

The active post-Step-8.5 direction is:

```text
SSDBCODI clusters/anomalies
  -> rule_panel decision-tree surrogate rules
  -> DeepSeek categorized rule interpretation
```

Decision trees are used only to extract explanatory rules from SSDBCODI output.
They do not perform clustering, outlier detection, or state updates.
