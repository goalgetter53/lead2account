"""Multi-key blocking with block-size caps, plus pair-completeness metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from l2a.records import Rec, acr

DEFAULT_MAX_BLOCK = 300


def block_keys(r: Rec) -> set[str]:
    keys: set[str] = set()
    kt = r.name.key_tokens
    for reg in r.registrables:
        keys.add("d:" + reg)
        keys.add("j:" + reg.split(".", 1)[0].replace("-", ""))
    if r.name.core:
        keys.add("n:" + r.name.core)
    j = r.joined
    if j:
        keys.add("j:" + j)
    if len(kt) >= 2:
        keys.add("t:" + kt[0] + "|" + kt[1])
    if kt and len(kt[0]) >= 4:
        keys.add("f:" + kt[0])
    if len(j) >= 8:
        keys.add("ps:" + j[:4] + "|" + j[-4:])
    if len(kt) >= 3:
        a = acr(r)
        if len(a) >= 3:
            keys.add("j:" + a)
    return keys


class BlockIndex:
    """Inverted index key -> record positions; oversized blocks are dropped."""

    def __init__(self, records: Sequence[Rec], max_block: int = DEFAULT_MAX_BLOCK) -> None:
        self.records = records
        self.max_block = max_block
        idx: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(records):
            for k in block_keys(r):
                idx[k].append(i)
        self.index = {k: v for k, v in idx.items() if len(v) <= max_block}
        self.dropped = {k: len(v) for k, v in idx.items() if len(v) > max_block}

    def pairs(self) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        for members in self.index.values():
            n = len(members)
            for a in range(n):
                ia = members[a]
                for b in range(a + 1, n):
                    ib = members[b]
                    out.add((ia, ib) if ia < ib else (ib, ia))
        return out

    def candidates(self, q: Rec) -> set[int]:
        out: set[int] = set()
        for k in block_keys(q):
            out.update(self.index.get(k, ()))
        return out


def pair_completeness(pairs: Iterable[tuple[int, int]], labels: Sequence[str]) -> dict[str, float]:
    """PC = true pairs retained / all true pairs; RR = 1 - candidates / all pairs."""
    groups: dict[str, int] = defaultdict(int)
    for lab in labels:
        groups[lab] += 1
    total_true = sum(c * (c - 1) // 2 for c in groups.values())
    cand = 0
    hit = 0
    for a, b in pairs:
        cand += 1
        if labels[a] == labels[b]:
            hit += 1
    n = len(labels)
    all_pairs = n * (n - 1) // 2
    return {
        "candidate_pairs": float(cand),
        "true_pairs": float(total_true),
        "pair_completeness": hit / total_true if total_true else 1.0,
        "reduction_ratio": 1 - cand / all_pairs if all_pairs else 0.0,
        "pairs_quality": hit / cand if cand else 0.0,
    }
