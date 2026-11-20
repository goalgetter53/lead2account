"""End-to-end experiment: train + tune on DEV entity groups, evaluate once on TEST.

Writes models/*.json and results/metrics.json. All numbers in the README come
from results/metrics.json produced by this script.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from l2a.blocking import BlockIndex, pair_completeness
from l2a.cluster import CannotLink, cluster
from l2a.dataset import stable_hash
from l2a.hierarchy import Hierarchy
from l2a.metrics import duplicate_rates, lead_prf_ci, pairwise_counts, pairwise_prf_ci
from l2a.model import PairModel
from l2a.pipeline import (
    LeadMatcher,
    account_records,
    dedupe,
    lead_records,
    read_csv_gz,
    score_pairs,
    training_pairs,
)
from l2a.records import Idf, Rec

PROC = Path("data/processed")
MODELS = Path("models")
RESULTS = Path("results")


def _subset(rows: list[dict[str, str]], pred: Any) -> list[dict[str, str]]:
    return [r for r in rows if pred(r)]


def _dev_part(r: dict[str, str]) -> str:
    return "train" if (stable_hash(r["group"]) >> 4) % 2 == 0 else "tune"


def baseline_exact(records: Sequence[Rec]) -> list[int]:
    """Rule baseline: same normalised core name OR same site key -> same account."""
    parent = list(range(len(records)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen: dict[str, int] = {}
    for i, r in enumerate(records):
        keys = ["n:" + r.name.core] if r.name.core else []
        keys += ["s:" + s for s in r.site_keys]
        for k in keys:
            if k in seen:
                a, b = find(seen[k]), find(i)
                if a != b:
                    parent[b] = a
            else:
                seen[k] = i
    return [find(i) for i in range(len(records))]


def lead_outcomes(
    matcher: LeadMatcher,
    leads: Sequence[Rec],
    rows: Sequence[dict[str, str]],
    acc_truth: Sequence[str],
    use_routed: bool = False,
    target: dict[str, str] | None = None,
) -> list[tuple[str, str, str]]:
    """Return (group, outcome, hard_case) per lead."""
    members: dict[int, Counter[str]] = defaultdict(Counter)
    for lab, t in zip(matcher.labels, acc_truth, strict=True):
        members[lab][t] += 1
    majority = {lab: c.most_common(1)[0][0] for lab, c in members.items()}
    out = []
    for q, row in zip(leads, rows, strict=True):
        m = matcher.match(q)
        acc = m.routed_account if use_routed else m.account
        want = (target or {}).get(row["true_lei"], row["true_lei"])
        in_crm = row["in_crm"] == "1"
        if acc is None:
            kind = "missed" if in_crm else "true_none"
        elif not in_crm:
            kind = "false_match"
        else:
            kind = "correct" if majority[acc] == want else "wrong"
        out.append((row["group"], kind, row["hard_case"] or "none"))
    return out


def run(n_boot: int) -> dict[str, Any]:
    t0 = time.time()
    accounts = read_csv_gz(PROC / "accounts.csv.gz")
    leads = read_csv_gz(PROC / "leads.csv.gz")
    edges = read_csv_gz(PROC / "hierarchy_edges.csv.gz")
    hier = Hierarchy((e["child_lei"], e["parent_lei"], e["source"]) for e in edges)
    res: dict[str, Any] = {"dataset": json.loads((PROC / "dataset_stats.json").read_text())}
    rng = np.random.default_rng(0)

    dev_acc = _subset(accounts, lambda r: r["split"] == "dev")
    test_acc = _subset(accounts, lambda r: r["split"] == "test")
    tr_acc = _subset(dev_acc, lambda r: _dev_part(r) == "train")
    tu_acc = _subset(dev_acc, lambda r: _dev_part(r) == "tune")
    dev_leads = _subset(leads, lambda r: r["split"] == "dev")
    test_leads = _subset(leads, lambda r: r["split"] == "test")
    tr_leads = _subset(dev_leads, lambda r: _dev_part(r) == "train")
    tu_leads = _subset(dev_leads, lambda r: _dev_part(r) == "tune")
    res["split_sizes"] = {
        "dev_train_accounts": len(tr_acc),
        "dev_tune_accounts": len(tu_acc),
        "test_accounts": len(test_acc),
        "dev_train_leads": len(tr_leads),
        "dev_tune_leads": len(tu_leads),
        "test_leads": len(test_leads),
        "test_entity_groups": len({r["group"] for r in test_acc}),
    }

    # ---------------- blocking quality
    res["blocking"] = {}
    for name, rows in (("dev", dev_acc), ("test", test_acc)):
        recs = account_records(rows)
        idx = BlockIndex(recs)
        res["blocking"][name] = pair_completeness(idx.pairs(), [r["true_lei"] for r in rows])
        res["blocking"][name]["dropped_oversized_blocks"] = float(len(idx.dropped))

    # ---------------- account pair model (fit on dev-train)
    tr_recs = account_records(tr_acc)
    tr_idf = Idf(tr_recs)
    x, y = training_pairs(
        tr_recs,
        [r["true_lei"] for r in tr_acc],
        tr_recs,
        [r["true_lei"] for r in tr_acc],
        BlockIndex(tr_recs),
        tr_idf,
        same_set=True,
        rng=rng,
    )
    acc_model = PairModel.fit(x, y)
    res["account_model"] = {
        "train_pairs": len(y),
        "train_positive": int(y.sum()),
        "coef": dict(zip(acc_model.features, map(float, acc_model.coef), strict=True)),
    }

    # ---------------- tune threshold + cannot-link rules on dev-tune (cluster-level F1)
    tu_recs = account_records(tu_acc)
    tu_truth = [r["true_lei"] for r in tu_acc]
    tu_idf = Idf(tu_recs)
    tu_pairs = sorted(BlockIndex(tu_recs).pairs())
    tu_p = score_pairs(tu_recs, tu_pairs, acc_model, tu_idf)
    tu_edges = [(float(s), a, b) for s, (a, b) in zip(tu_p, tu_pairs, strict=True)]
    grid: list[dict[str, Any]] = []
    rule_sets: list[tuple[str, CannotLink]] = []
    for neg in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4):
        rule_sets += [
            (f"neg<{neg}", CannotLink(False, False, neg)),
            (f"country,neg<{neg}", CannotLink(False, True, neg)),
            (f"country,registry,neg<{neg}", CannotLink(True, True, neg)),
        ]
    for rules_name, rules in rule_sets:
        for t in np.round(np.arange(0.15, 0.96, 0.05), 2):
            labels, _ = cluster(tu_recs, tu_edges, float(t), rules)
            tp, fp, fn = pairwise_counts(labels, tu_truth)
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            grid.append({"rules": rules_name, "threshold": float(t), "f1": f1})
    # primary configuration excludes the registry-id rule (pure fuzzy matching)
    best = max((g for g in grid if "registry" not in g["rules"]), key=lambda g: g["f1"])
    acc_model.threshold = best["threshold"]
    primary_rules = dict(rule_sets)[best["rules"]]
    res["tuning"] = {"grid": grid, "chosen": best}
    MODELS.mkdir(exist_ok=True)
    acc_model.to_json(MODELS / "account_pair_model.json")

    # ---------------- lead model (fit on dev-train leads vs dev-train accounts)
    tr_leads_recs = lead_records(tr_leads)
    xl, yl = training_pairs(
        tr_leads_recs,
        [r["true_lei"] for r in tr_leads],
        tr_recs,
        [r["true_lei"] for r in tr_acc],
        BlockIndex(tr_recs),
        tr_idf,
        same_set=False,
        rng=rng,
    )
    lead_model = PairModel.fit(xl, yl)
    tu_labels, _ = cluster(tu_recs, tu_edges, acc_model.threshold, primary_rules)
    tu_lead_recs = lead_records(tu_leads)
    lead_grid: list[dict[str, float]] = []
    for t in np.round(np.arange(0.2, 0.96, 0.05), 2):
        lead_model.threshold = float(t)
        m = LeadMatcher(tu_recs, tu_labels, lead_model, tu_idf)
        oc = lead_outcomes(m, tu_lead_recs, tu_leads, tu_truth)
        st = lead_prf_ci([(g, k) for g, k, _h in oc], n_boot=1)
        lead_grid.append({"threshold": float(t), "f1": st["f1"]})
    lead_best = max(lead_grid, key=lambda g: g["f1"])
    lead_model.threshold = lead_best["threshold"]
    lead_model.to_json(MODELS / "lead_pair_model.json")
    res["lead_model"] = {
        "train_pairs": len(yl),
        "train_positive": int(yl.sum()),
        "tuning_grid": lead_grid,
        "chosen_threshold": lead_best["threshold"],
        "coef": dict(zip(lead_model.features, map(float, lead_model.coef), strict=True)),
    }

    # ================= TEST (touched once, frozen models) =================
    test_recs = account_records(test_acc)
    test_truth = [r["true_lei"] for r in test_acc]
    test_idf = Idf(test_recs)
    t1 = time.time()
    dd = dedupe(test_recs, acc_model, primary_rules, test_idf)
    dedupe_secs = time.time() - t1
    res["test_dedupe"] = {
        "config": {"threshold": acc_model.threshold, "cannot_link": best["rules"]},
        "pairwise": pairwise_prf_ci(dd.labels, test_truth, n_boot),
        "duplicates": duplicate_rates(dd.labels, test_truth),
        "candidate_pairs": dd.candidate_pairs,
        "refused_merges": dd.refused_merges,
        "throughput_records_per_sec": len(test_recs) / dedupe_secs,
    }
    reg_labels, _ = cluster(
        test_recs,
        dd.edges,
        acc_model.threshold,
        CannotLink(True, primary_rules.country_without_domain),
    )
    res["test_dedupe_with_registry_cannot_link"] = {
        "pairwise": pairwise_prf_ci(reg_labels, test_truth, n_boot),
        "duplicates": duplicate_rates(reg_labels, test_truth),
    }
    base = baseline_exact(test_recs)
    res["test_baseline_exact_name_or_site"] = {
        "pairwise": pairwise_prf_ci(base, test_truth, n_boot),
        "duplicates": duplicate_rates(base, test_truth),
    }

    test_lead_recs = lead_records(test_leads)
    matcher = LeadMatcher(test_recs, dd.labels, lead_model, test_idf, hier, route="child")
    t2 = time.time()
    oc = lead_outcomes(matcher, test_lead_recs, test_leads, test_truth)
    lead_secs = time.time() - t2
    res["test_leads"] = lead_prf_ci([(g, k) for g, k, _h in oc], n_boot)
    res["test_leads"]["throughput_leads_per_sec"] = len(test_leads) / lead_secs
    strata: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for g, k, h in oc:
        strata[h].append((g, k))
    res["test_leads_by_hard_case"] = {h: lead_prf_ci(v, n_boot) for h, v in sorted(strata.items())}

    # ---- parent rollup routing: target = highest true ancestor that has a CRM account
    present = {r["true_lei"] for r in test_acc}
    target = {lei: hier.rollup(lei, present) for lei in present}
    roll_rows = [
        (q, r)
        for q, r in zip(test_lead_recs, test_leads, strict=True)
        if r["in_crm"] == "1" and target.get(r["true_lei"], r["true_lei"]) != r["true_lei"]
    ]
    rmatcher = LeadMatcher(
        test_recs, dd.labels, lead_model, test_idf, hier, route="ultimate_parent"
    )
    roc = lead_outcomes(
        rmatcher,
        [q for q, _ in roll_rows],
        [r for _, r in roll_rows],
        test_truth,
        use_routed=True,
        target=target,
    )
    res["test_parent_rollup"] = lead_prf_ci([(g, k) for g, k, _h in roc], n_boot)
    res["runtime_seconds"] = time.time() - t0
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    res = run(a.n_boot)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    print(json.dumps({k: res[k] for k in ("test_dedupe", "test_leads")}, indent=2, default=float))


if __name__ == "__main__":
    main()
