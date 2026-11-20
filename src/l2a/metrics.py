"""Evaluation metrics: pairwise P/R/F1 with component bootstrap, duplicate rates,
lead-match outcomes."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np


def _c2(n: int) -> int:
    return n * (n - 1) // 2


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def pairwise_counts(pred: Sequence[int], truth: Sequence[str]) -> tuple[int, int, int]:
    cells: Counter[tuple[int, str]] = Counter(zip(pred, truth, strict=True))
    tp = sum(_c2(n) for n in cells.values())
    pp = sum(_c2(n) for n in Counter(pred).values())
    tt = sum(_c2(n) for n in Counter(truth).values())
    return tp, pp - tp, tt - tp


def pairwise_prf_ci(
    pred: Sequence[int], truth: Sequence[str], n_boot: int = 1000, seed: int = 7
) -> dict[str, Any]:
    """Pairwise P/R/F1 with 95% bootstrap CIs over evaluation components
    (connected components of predicted clusters + true entities), so every
    TP/FP/FN pair falls inside exactly one resampling unit."""
    n = len(pred)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first: dict[tuple[str, object], int] = {}
    for i, (c, t) in enumerate(zip(pred, truth, strict=True)):
        for key in (("c", c), ("t", t)):
            if key in first:
                a, b = find(first[key]), find(i)
                if a != b:
                    parent[b] = a
            else:
                first[key] = i
    comp: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comp[find(i)].append(i)
    rows = []
    for members in comp.values():
        tp, fp, fn = pairwise_counts([pred[i] for i in members], [truth[i] for i in members])
        rows.append((tp, fp, fn))
    arr = np.asarray(rows, dtype=np.float64)
    tot = arr.sum(axis=0)
    p, r, f = _prf(float(tot[0]), float(tot[1]), float(tot[2]))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        bs = arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)
        boots.append(_prf(float(bs[0]), float(bs[1]), float(bs[2])))
    boot_arr = np.asarray(boots)
    lo, hi = np.percentile(boot_arr, 2.5, axis=0), np.percentile(boot_arr, 97.5, axis=0)
    return {
        "precision": p,
        "recall": r,
        "f1": f,
        "precision_ci": [float(lo[0]), float(hi[0])],
        "recall_ci": [float(lo[1]), float(hi[1])],
        "f1_ci": [float(lo[2]), float(hi[2])],
        "tp_pairs": float(tot[0]),
        "fp_pairs": float(tot[1]),
        "fn_pairs": float(tot[2]),
    }


def duplicate_rates(pred: Sequence[int], truth: Sequence[str]) -> dict[str, float]:
    records = len(truth)
    entities = len(set(truth))
    clusters = len(set(pred))
    touch: dict[str, set[int]] = defaultdict(set)
    ents_in: dict[int, set[str]] = defaultdict(set)
    for c, t in zip(pred, truth, strict=True):
        touch[t].add(c)
        ents_in[c].add(t)
    acc_per_entity_after = sum(len(v) for v in touch.values()) / entities
    return {
        "records": float(records),
        "true_entities": float(entities),
        "accounts_after_merge": float(clusters),
        "records_per_entity_before": records / entities,
        "accounts_per_entity_after": acc_per_entity_after,
        "duplicate_rate_before": 1 - entities / records,
        "duplicate_rate_after": 1 - entities / sum(len(v) for v in touch.values()),
        "overmerged_cluster_rate": sum(1 for v in ents_in.values() if len(v) > 1) / clusters,
    }


def lead_prf_ci(
    outcomes: Sequence[tuple[str, str]], n_boot: int = 1000, seed: int = 11
) -> dict[str, Any]:
    """outcomes: (group, kind) with kind in {correct, wrong, missed, false_match, true_none}."""
    by_group: dict[str, Counter[str]] = defaultdict(Counter)
    for g, k in outcomes:
        by_group[g][k] += 1
    keys = ["correct", "wrong", "missed", "false_match", "true_none"]
    arr = np.asarray([[c[k] for k in keys] for c in by_group.values()], dtype=np.float64)

    def stats(v: np.ndarray) -> tuple[float, float, float]:
        correct, wrong, missed, false_match, _tn = v
        returned = correct + wrong + false_match
        in_crm = correct + wrong + missed
        p = correct / returned if returned else 1.0
        r = correct / in_crm if in_crm else 1.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    tot = arr.sum(axis=0)
    p, r, f = stats(tot)
    rng = np.random.default_rng(seed)
    b = np.asarray(
        [stats(arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)) for _ in range(n_boot)]
    )
    lo, hi = np.percentile(b, 2.5, axis=0), np.percentile(b, 97.5, axis=0)
    out: dict[str, Any] = {k: float(v) for k, v in zip(keys, tot, strict=True)}
    nocrm = tot[3] + tot[4]
    out.update(
        {
            "n": float(tot.sum()),
            "precision": p,
            "recall": r,
            "f1": f,
            "precision_ci": [float(lo[0]), float(hi[0])],
            "recall_ci": [float(lo[1]), float(hi[1])],
            "f1_ci": [float(lo[2]), float(hi[2])],
            "no_match_accuracy": float(tot[4] / nocrm) if nocrm else float("nan"),
        }
    )
    return out
