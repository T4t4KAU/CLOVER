```python
# DEBUG.py
import pandas as pd
import numpy as np
# env: tables/views/table/view/frame/schema/head/sample/values/search_values
```

{% if last_iteration %}
JSON only. Last iter: {"c":"def collect(env):\n    ...\n    return evidence"}.
{% else %}
JSON only. {"d":"..."} inspects; {"c":"def collect(env):\n    ...\n    return evidence"} submits.
{% endif %}
Iter: {{ iteration }}

```python
{{ prompt_code }}
```

# FEEDBACK
{{ feedback }}
