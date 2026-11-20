"""Pairwise comparison features (shared by account dedupe and lead matching)."""

from __future__ import annotations

import re

import numpy as np
from numpy.typing import NDArray
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from l2a.records import Idf, Rec, acr

FEATURES: tuple[str, ...] = (
    "name_jw",
    "name_token_set",
    "name_core_equal",
    "name_joined_equal",
    "idf_overlap",
    "idf_unmatched_max",
    "name_subset",
    "acronym_match",
    "domain_reg_equal",
    "domain_site_equal",
    "domain_site_conflict",
    "domain_both_missing",
    "domain_reg_conflict",
    "domain_label_in_name",
    "country_equal",
    "country_conflict",
    "legal_form_conflict",
    "digits_conflict",
    "short_name",
)

_DIGITS = re.compile(r"\d+")


def _digits(r: Rec) -> frozenset[str]:
    return frozenset(_DIGITS.findall(r.name.core))


def compare(a: Rec, b: Rec, idf: Idf) -> dict[str, float]:
    ca, cb = a.name.core, b.name.core
    ta, tb = set(a.name.key_tokens), set(b.name.key_tokens)
    shared = ta & tb
    w_shared = sum(idf(t) for t in shared)
    w_all = sum(idf(t) for t in ta | tb) or 1.0
    unmatched = [idf(t) for t in (ta ^ tb)]
    ra, rb = a.registrables, b.registrables
    sa, sb = a.site_keys, b.site_keys
    ja, jb = a.joined, b.joined
    labels_a = {x.split(".", 1)[0].replace("-", "") for x in ra}
    labels_b = {x.split(".", 1)[0].replace("-", "") for x in rb}
    da, db = _digits(a), _digits(b)
    acr_a, acr_b = acr(a), acr(b)
    return {
        "name_jw": JaroWinkler.normalized_similarity(ca, cb) if ca and cb else 0.0,
        "name_token_set": fuzz.token_set_ratio(ca, cb) / 100.0 if ca and cb else 0.0,
        "name_core_equal": float(bool(ca) and ca == cb),
        "name_joined_equal": float(bool(ja) and ja == jb),
        "idf_overlap": w_shared / w_all,
        "idf_unmatched_max": max(unmatched) if unmatched else 0.0,
        "name_subset": float(bool(ta) and bool(tb) and (ta <= tb or tb <= ta)),
        "acronym_match": float(
            (len(acr_a) >= 3 and acr_a == jb) or (len(acr_b) >= 3 and acr_b == ja)
        ),
        "domain_reg_equal": float(bool(ra & rb)),
        "domain_site_equal": float(bool(sa & sb)),
        "domain_site_conflict": float(bool(ra & rb) and not (sa & sb)),
        "domain_both_missing": float(not ra or not rb),
        "domain_reg_conflict": float(bool(ra) and bool(rb) and not (ra & rb)),
        "domain_label_in_name": float(bool((labels_a & {jb}) or (labels_b & {ja}))),
        "country_equal": float(bool(a.country) and a.country == b.country),
        "country_conflict": float(bool(a.country and b.country) and a.country != b.country),
        "legal_form_conflict": float(
            bool(a.name.legal_form and b.name.legal_form) and a.name.legal_form != b.name.legal_form
        ),
        "digits_conflict": float(da != db and bool(da or db)),
        "short_name": float(min(len(ja), len(jb)) <= 4),
    }


def feature_matrix(pairs: list[tuple[Rec, Rec]], idf: Idf) -> NDArray[np.float64]:
    out = np.zeros((len(pairs), len(FEATURES)), dtype=np.float64)
    for i, (a, b) in enumerate(pairs):
        f = compare(a, b, idf)
        out[i] = [f[k] for k in FEATURES]
    return out
