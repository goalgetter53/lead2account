"""Dedupe accounts and match leads to accounts."""

from __future__ import annotations

import csv
import gzip
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from l2a.blocking import BlockIndex
from l2a.cluster import CannotLink, cluster
from l2a.features import compare, feature_matrix
from l2a.hierarchy import Hierarchy
from l2a.model import PairModel
from l2a.records import Idf, Rec, make_record


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def account_records(rows: Sequence[dict[str, str]], with_registry_id: bool = True) -> list[Rec]:
    """Accounts loaded from the ERP/registry feed carry their registry id (LEI),
    used only for hierarchy linking and the optional registry cannot-link rule,
    never as a match feature."""
    return [
        make_record(
            r["record_id"],
            r["name"],
            website=r["website"],
            country=r["country"],
            registry_id=r["true_lei"] if with_registry_id and r["source"] == "erp_import" else "",
        )
        for r in rows
    ]


def lead_records(rows: Sequence[dict[str, str]]) -> list[Rec]:
    return [
        make_record(
            r["lead_id"], r["company"], website=r["website"], email=r["email"], country=r["country"]
        )
        for r in rows
    ]


@dataclass
class DedupeResult:
    labels: list[int]
    edges: list[tuple[float, int, int]]
    candidate_pairs: int
    refused_merges: int


def score_pairs(
    records: Sequence[Rec], pairs: Sequence[tuple[int, int]], model: PairModel, idf: Idf
) -> NDArray[np.float64]:
    out = np.zeros(len(pairs), dtype=np.float64)
    step = 20000
    for i in range(0, len(pairs), step):
        chunk = pairs[i : i + step]
        x = feature_matrix([(records[a], records[b]) for a, b in chunk], idf)
        out[i : i + len(chunk)] = model.proba(x)
    return out


def dedupe(
    records: Sequence[Rec], model: PairModel, rules: CannotLink, idf: Idf | None = None
) -> DedupeResult:
    idf = idf or Idf(records)
    pairs = sorted(BlockIndex(records).pairs())
    p = score_pairs(records, pairs, model, idf)
    edges = [(float(s), a, b) for s, (a, b) in zip(p, pairs, strict=True)]
    labels, refused = cluster(records, edges, model.threshold, rules)
    return DedupeResult(labels, edges, len(pairs), refused)


Route = Literal["child", "ultimate_parent"]


@dataclass
class Match:
    lead_id: str
    account: int | None  # cluster label
    record: int | None  # best record index
    confidence: float
    reasons: list[str] = field(default_factory=list)
    routed_account: int | None = None


class LeadMatcher:
    def __init__(
        self,
        accounts: Sequence[Rec],
        labels: Sequence[int],
        model: PairModel,
        idf: Idf,
        hierarchy: Hierarchy | None = None,
        route: Route = "child",
    ) -> None:
        self.accounts = accounts
        self.labels = labels
        self.model = model
        self.idf = idf
        self.index = BlockIndex(accounts)
        self.hierarchy = hierarchy
        self.route = route
        # cluster <-> registry id (majority vote of ERP-fed records in the cluster)
        votes: dict[int, Counter[str]] = defaultdict(Counter)
        for r, lab in zip(accounts, labels, strict=True):
            if r.registry_id:
                votes[lab][r.registry_id] += 1
        self.cluster_reg = {lab: c.most_common(1)[0][0] for lab, c in votes.items()}
        self.reg_cluster: dict[str, int] = {}
        for lab, reg in self.cluster_reg.items():
            self.reg_cluster.setdefault(reg, lab)

    def match(self, lead: Rec) -> Match:
        cands = sorted(self.index.candidates(lead))
        if not cands:
            return Match(lead.rid, None, None, 0.0, ["NO_CANDIDATES"])
        x = feature_matrix([(lead, self.accounts[i]) for i in cands], self.idf)
        p = self.model.proba(x)
        best_by_cluster: dict[int, tuple[float, int]] = {}
        for prob, i in zip(p, cands, strict=True):
            lab = self.labels[i]
            if lab not in best_by_cluster or prob > best_by_cluster[lab][0]:
                best_by_cluster[lab] = (float(prob), i)
        lab, (conf, rec) = max(best_by_cluster.items(), key=lambda kv: kv[1][0])
        reasons = reason_codes(lead, self.accounts[rec], self.idf)
        if conf < self.model.threshold:
            return Match(lead.rid, None, rec, conf, ["BELOW_THRESHOLD", *reasons])
        routed = lab
        if self.route == "ultimate_parent" and self.hierarchy is not None:
            reg = self.cluster_reg.get(lab)
            if reg:
                top = self.hierarchy.rollup(reg, set(self.reg_cluster))
                if top != reg:
                    routed = self.reg_cluster[top]
                    reasons.append("PARENT_ROLLUP")
        return Match(lead.rid, lab, rec, conf, reasons, routed)


def reason_codes(lead: Rec, acc: Rec, idf: Idf) -> list[str]:
    f = compare(lead, acc, idf)
    out: list[str] = []
    if f["domain_site_equal"]:
        out.append("DOMAIN_EXACT")
    elif f["domain_reg_equal"]:
        out.append("DOMAIN_SAME_REGISTRABLE_DIFFERENT_SITE")
    if f["name_core_equal"] or f["name_joined_equal"]:
        out.append("NAME_EXACT_NORMALISED")
    elif f["acronym_match"]:
        out.append("NAME_ACRONYM")
    elif f["name_token_set"] >= 0.85:
        out.append(f"NAME_FUZZY_{f['name_token_set']:.2f}")
    if f["domain_label_in_name"]:
        out.append("DOMAIN_LABEL_EQUALS_NAME")
    if f["country_equal"]:
        out.append("COUNTRY_MATCH")
    if f["country_conflict"]:
        out.append("COUNTRY_CONFLICT")
    if lead.freemail:
        out.append("FREEMAIL_DOMAIN_IGNORED")
    return out


def training_pairs(
    left: Sequence[Rec],
    left_ent: Sequence[str],
    right: Sequence[Rec],
    right_ent: Sequence[str],
    index: BlockIndex,
    idf: Idf,
    same_set: bool,
    rng: np.random.Generator,
    max_neg_per_query: int = 30,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Candidate pairs from blocking, labelled by shared entity id."""
    pairs: list[tuple[Rec, Rec]] = []
    ys: list[int] = []
    for i, q in enumerate(left):
        cands = [j for j in index.candidates(q) if not (same_set and j <= i)]
        pos = [j for j in cands if right_ent[j] == left_ent[i]]
        neg = [j for j in cands if right_ent[j] != left_ent[i]]
        if len(neg) > max_neg_per_query:
            neg = list(rng.choice(neg, size=max_neg_per_query, replace=False))
        for j in pos:
            pairs.append((q, right[j]))
            ys.append(1)
        for j in neg:
            pairs.append((q, right[int(j)]))
            ys.append(0)
    return feature_matrix(pairs, idf), np.asarray(ys, dtype=np.int_)
