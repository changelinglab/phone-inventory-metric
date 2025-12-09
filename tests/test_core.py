from phone_inventory_metric import get_metrics


def test_simple() -> None:
    result = get_metrics(["abc", "obc"], ["abc", "ebc"])
    assert isinstance(result, dict)
