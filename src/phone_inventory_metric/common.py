from typing import TypeVar

V = TypeVar("V")


class setkeydict(dict[tuple[str, ...], V]):
    def _transform_key(self, key: tuple[str, ...] | str) -> tuple[str, ...]:
        if isinstance(key, str):
            key = (key,)
        return tuple(sorted(key))

    def __setitem__(self, key: tuple[str, ...] | str, value: V) -> None:
        return super().__setitem__(self._transform_key(key), value)

    def __getitem__(self, key: tuple[str, ...] | str) -> V:
        return super().__getitem__(self._transform_key(key))

    def __delitem__(self, key: tuple[str, ...] | str) -> None:
        return super().__delitem__(self._transform_key(key))

    def __contains__(self, key: object) -> bool:
        if not (isinstance(key, tuple) or isinstance(key, str)):
            return False
        return super().__contains__(self._transform_key(key))
