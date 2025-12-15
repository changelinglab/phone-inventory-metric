import math

import pytest

from phone_inventory_metric import get_metrics
from phone_inventory_metric.core import max_nan_safe


def test_simple() -> None:
    result = get_metrics("abc", "obc")
    assert isinstance(result, dict)

def test_ordered_top_nan() -> None:
    result = get_metrics("abc", "obc", search_max=True)
    from pprint import pprint
    pprint(result)
    for key, value in result.items():
        assert not math.isnan(value), key

nan = float('nan')

@pytest.mark.parametrize(
    "arg,target",
    [
        ([(0, 0, 0)], (0, 0, 0)),
        ([(nan, 0, 0)], (nan, 0, 0)),
        ([(1, 0, 0), (nan, 0, 0)], (1, 0, 0)),
        ([(0, 0, 0), (nan, 0, 0)], (0, 0, 0)),
        ([(nan, 0, 0), (0, 0, 0)], (0, 0, 0)),
        ([(1, 0.5, 0), (1, 0.6, 0)], (1, 0.6, 0)),
        ([(1, 0.6, 0), (1, 0.5, 0)], (1, 0.6, 0)),
    ],
)
def test_max_nan_safe(arg: list[tuple[float, float, float]], target: tuple[float, float, float]) -> None:
    result = max_nan_safe(arg)
    assert result == target, f"arg = {arg}"
