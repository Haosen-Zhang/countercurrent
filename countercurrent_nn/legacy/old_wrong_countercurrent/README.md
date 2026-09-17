This is the exact pre-audit model/config snapshot, preserved for historical
reproduction. `manifest.json` records SHA-256 hashes of the original files.
No old results or checkpoints were removed or reclassified.

The coupled model computes `corrected_c = c_ref + q` but consumes only the
uncorrected reverse proposals in prediction. Thus the paired C update is a
diagnostic dead end. Do not use this implementation as the current main model.

Historical models can be loaded explicitly with:

```python
from countercurrent_nn.legacy.old_wrong_countercurrent.models import build_model
```

The active registry never imports these models. Its new inference-version buffer
also prevents silently loading legacy checkpoints with strict state-dict loading.
The old README describes the old schedule and its measured compute only.
