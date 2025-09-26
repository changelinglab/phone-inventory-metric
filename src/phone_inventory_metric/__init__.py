from builtins import frozenset
from collections import Counter
from typing import Iterable

import panphon
from ipatok import tokenise


def ensure_unique(x: Iterable) -> None:
    if len(list(x)) != len(set(x)):
        raise ValueError("Collection has non-unique elements.")


def get_set_f1_score(
    ref: Iterable[object], target: Iterable[object], search_max=False
) -> float:
    ensure_unique(ref)
    ensure_unique(target)
    ref = frozenset(ref)
    target = list(target)
    if len(ref) == 0 or len(target) == 0:
        return float("nan")

    def _f1(target: Iterable[object]) -> float:
        target = frozenset(target)
        intersection = ref & target
        precision = len(intersection) / len(target)
        recall = len(intersection) / len(ref)
        if precision == 0 or recall == 0:
            return float("nan")
        return 2 / (1 / precision + 1 / recall)

    if not search_max:
        return _f1(target)
    else:
        return max(_f1(target[: i + 1]) for i in range(len(target)))


def get_set_f1_score_featured(
    ref: Iterable[str], target: Iterable[str], search_max: bool = False
) -> float:
    ensure_unique(ref)
    ensure_unique(target)

    feature_table = panphon.FeatureTable()
    # TODO Why does "g" get ignored by panphon?
    ref_segs = feature_table.word_fts("".join(ref))
    other_segs = feature_table.word_fts("".join(target))

    if len(ref_segs) == 0 or len(other_segs) == 0:
        return float("nan")

    def _f1(oss: list[panphon.segment.Segment]) -> float:
        precision = sum(
            1 - min(os.norm_hamming_distance(rs) for rs in ref_segs) for os in oss
        ) / len(oss)
        recall = sum(
            1 - min(rs.norm_hamming_distance(os) for os in oss) for rs in ref_segs
        ) / len(ref_segs)

        if precision == 0 or recall == 0:
            return float("nan")
        return 2 / (1 / precision + 1 / recall)

    if not search_max:
        return _f1(other_segs)
    else:
        return max(_f1(other_segs[: i + 1]) for i in range(len(other_segs)))


def tokenize_corpus(corpus: Iterable[str]) -> list[str]:
    counter = Counter(char for chunk in corpus for char in tokenise(chunk))
    return [c for c, _ in counter.most_common()]


def get_metrics(
    ref: Iterable[str], target: Iterable[str], search_max: bool = False
) -> dict[str, float]:
    ref_set = tokenize_corpus(ref)
    target_set = tokenize_corpus(target)
    raw_f1_score = get_set_f1_score(ref_set, target_set)
    max_f1_score = get_set_f1_score(ref_set, target_set, search_max=search_max)
    raw_f1_score_featured = get_set_f1_score_featured(ref_set, target_set)
    max_f1_score_featured = get_set_f1_score_featured(
        ref_set, target_set, search_max=search_max
    )
    return {
        "raw_f1_score": raw_f1_score,
        "raw_f1_score_featured": raw_f1_score_featured,
        "max_f1_score": max_f1_score,
        "max_f1_score_featured": max_f1_score_featured,
    }
