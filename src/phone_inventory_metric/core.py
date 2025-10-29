from builtins import frozenset
from collections import Counter
from typing import Iterable

import numpy as np
import panphon  # type: ignore[import-not-found]
import scipy
from ipatok import tokenise  # type: ignore[import-untyped]

from src.phone_inventory_metric.common import setkeydict


def ensure_unique(x: Iterable) -> None:
    if len(list(x)) != len(set(x)):
        raise ValueError("Collection has non-unique elements.")


def _preprocess_sets(
    ref: Iterable[object], target: Iterable[object]
) -> tuple[frozenset[object], list[object]]:
    ensure_unique(ref)
    ensure_unique(target)
    return frozenset(ref), list(target)


def get_set_f1_score(
    ref: Iterable[object], target: Iterable[object], search_max=False
) -> float:
    ref, target = _preprocess_sets(ref, target)
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
    ref: Iterable[str],
    target: Iterable[str],
    search_max: bool = False,
    exclusive: bool = False,
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
        sim_matrix = np.array(
            [[1 - os.norm_hamming_distance(rs) for os in oss] for rs in ref_segs]
        )

        if exclusive:
            match_idxs = scipy.optimize.linear_sum_assignment(sim_matrix, maximize=True)
            new_sim_matrix = np.zeros_like(sim_matrix)
            new_sim_matrix[match_idxs] = sim_matrix[match_idxs]
            sim_matrix = new_sim_matrix
        precision = sim_matrix.max(0).mean()
        recall = sim_matrix.max(-1).mean()

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
) -> setkeydict[float]:
    results: setkeydict[float] = setkeydict()
    ref_set = tokenize_corpus(ref)
    target_set = tokenize_corpus(target)
    results["f1_score"] = get_set_f1_score(ref_set, target_set)
    if search_max:
        results["f1_score", "max"] = get_set_f1_score(
            ref_set, target_set, search_max=True
        )
        results["f1_score", "exclusive", "max", "featured"] = get_set_f1_score_featured(
            ref_set, target_set, exclusive=True, search_max=True
        )
    results["f1_score", "featured"] = get_set_f1_score_featured(ref_set, target_set)
    results["f1_score", "featured", "max"] = get_set_f1_score_featured(
        ref_set, target_set, search_max=search_max
    )
    results["f1_score", "exclusive", "featured"] = get_set_f1_score_featured(
        ref_set, target_set, exclusive=True
    )
    return results
