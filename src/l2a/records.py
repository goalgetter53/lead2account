"""Common record representation for accounts and leads."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from l2a.domains import DomainInfo, parse_domain
from l2a.normalize import NormName, acronym, normalize_name


@dataclass(frozen=True)
class Rec:
    rid: str
    name: NormName
    domains: tuple[DomainInfo, ...]  # non-freemail only
    country: str
    freemail: bool = False
    registry_id: str = ""

    @property
    def registrables(self) -> frozenset[str]:
        return frozenset(d.registrable for d in self.domains if d.registrable)

    @property
    def site_keys(self) -> frozenset[str]:
        return frozenset(d.site_key for d in self.domains if d.site_key)

    @property
    def joined(self) -> str:
        return "".join(self.name.key_tokens)


def make_record(
    rid: str,
    name: str,
    website: str = "",
    email: str = "",
    country: str = "",
    registry_id: str = "",
) -> Rec:
    doms: list[DomainInfo] = []
    freemail = False
    for v in (website, email):
        d = parse_domain(v)
        if not d.registrable:
            continue
        if d.is_freemail:
            freemail = True
            continue
        if all(x.site_key != d.site_key for x in doms):
            doms.append(d)
    return Rec(
        rid=rid,
        name=normalize_name(name),
        domains=tuple(doms),
        country=(country or "").strip().upper(),
        freemail=freemail,
        registry_id=registry_id,
    )


class Idf:
    def __init__(self, records: Iterable[Rec]) -> None:
        df: Counter[str] = Counter()
        n = 0
        for r in records:
            n += 1
            df.update(set(r.name.tokens))
        self.n = max(n, 1)
        self.df = df

    def __call__(self, tok: str) -> float:
        return math.log((self.n + 1) / (self.df.get(tok, 0) + 1)) + 1.0


def acr(rec: Rec) -> str:
    return acronym(rec.name.key_tokens)
