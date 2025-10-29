# Phone Inventory Metric

## Usage
```
from phone_inventory_metric import get_metrics

# Corpora are of type Iterable[str] (e.g., list[str])
# Return value is setkeydict[str | tuple[str, ...], float]
get_metrics(ref_corpus, target_corpus)
```
The `setkeydict` is a subclass of `dict` which sorts its (tuple) keys before
getting and setting elements, so that `my_setkeydict["f1_score", "featured"]`
is equivalent to `my_setkeydict["featured", "f1_score"]`.
