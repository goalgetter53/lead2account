"""Greedy constrained clustering: union-find over scored edges, highest first,
refusing merges that would violate cannot-link constraints."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from l2a.records import Rec


@dataclass
class _Cluster:
    countries: set[str] = field(default_factory=set)
    registrables: set[str] = field(default_factory=set)
    registry_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class CannotLink:
    registry_id_conflict: bool = False
    country_without_domain: bool = True
    # pairs scored below this are cannot-link: blocks chaining A~B~C when A!~C
    negative_edge_below: float = 0.0


def _violates(a: _Cluster, b: _Cluster, rules: CannotLink) -> bool:
    if (
        rules.registry_id_conflict
        and a.registry_ids
        and b.registry_ids
        and not (a.registry_ids & b.registry_ids)
    ):
        return True
    return bool(
        rules.country_without_domain
        and a.countries
        and b.countries
        and not (a.countries & b.countries)
        and not (a.registrables & b.registrables)
    )


def cluster(
    records: Sequence[Rec],
    edges: Sequence[tuple[float, int, int]],
    threshold: float,
    rules: CannotLink,
) -> tuple[list[int], int]:
    """Return (cluster label per record, number of merges refused by constraints)."""
    parent = list(range(len(records)))
    info = [
        _Cluster(
            countries={r.country} if r.country else set(),
            registrables=set(r.registrables),
            registry_ids={r.registry_id} if r.registry_id else set(),
        )
        for r in records
    ]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    members: list[list[int]] = [[i] for i in range(len(records))]
    forbidden: list[set[int]] = [set() for _ in records]
    if rules.negative_edge_below > 0:
        for p, a, b in edges:
            if p < rules.negative_edge_below:
                forbidden[a].add(b)
                forbidden[b].add(a)

    refused = 0
    for p, a, b in sorted(edges, key=lambda e: -e[0]):
        if p < threshold:
            break
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if _violates(info[ra], info[rb], rules):
            refused += 1
            continue
        if len(members[ra]) < len(members[rb]):
            ra, rb = rb, ra
        if forbidden[ra] and any(m in forbidden[ra] for m in members[rb]):
            refused += 1
            continue
        parent[rb] = ra
        members[ra].extend(members[rb])
        forbidden[ra] |= forbidden[rb]
        members[rb] = []
        forbidden[rb] = set()
        ia, ib = info[ra], info[rb]
        ia.countries |= ib.countries
        ia.registrables |= ib.registrables
        ia.registry_ids |= ib.registry_ids
    return [find(i) for i in range(len(records))], refused
