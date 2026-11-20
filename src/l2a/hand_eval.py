"""Evaluate the model against Hussain's hand labels (labels/hand_labels.csv).

Reports progress (n/500) and, once labels exist: model P/R/F1 on labelled
'match' vs 'no_match'/'parent_child' pairs (unsure excluded) with bootstrap CIs,
plus agreement between the automatic LEI/QID labels and the human labels
(a direct estimate of automatic-label noise).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from l2a.model import PairModel

LABELS = Path("labels")


def evaluate(
    pairs_csv: Path = LABELS / "pairs_to_label.csv",
    hand_csv: Path = LABELS / "hand_labels.csv",
    auto_csv: Path = LABELS / "auto_labels.csv",
    n_boot: int = 1000,
) -> dict[str, Any]:
    with pairs_csv.open(encoding="utf-8") as fh:
        pairs = {r["pair_id"]: r for r in csv.DictReader(fh)}
    with hand_csv.open(encoding="utf-8") as fh:
        hand = {r["pair_id"]: r["label"].strip() for r in csv.DictReader(fh)}
    auto: dict[str, str] = {}
    if auto_csv.exists():
        with auto_csv.open(encoding="utf-8") as fh:
            auto = {r["pair_id"]: r["auto_label"] for r in csv.DictReader(fh)}
    labelled = {k: v for k, v in hand.items() if v}
    out: dict[str, Any] = {"labelled": len(labelled), "total": len(pairs)}
    usable = [k for k, v in labelled.items() if v in {"match", "no_match", "parent_child"}]
    if not usable:
        out["status"] = f"pending ({len(labelled)}/{len(pairs)} labelled)"
        return out
    thr = {
        "account": PairModel.from_json(Path("models/account_pair_model.json")).threshold,
        "lead": PairModel.from_json(Path("models/lead_pair_model.json")).threshold,
    }
    y = np.asarray([labelled[k] == "match" for k in usable])
    pred = np.asarray(
        [float(pairs[k]["model_score"]) >= thr[pairs[k]["left_kind"]] for k in usable]
    )

    def prf(idx: np.ndarray) -> tuple[float, float, float]:
        tp = float(np.sum(pred[idx] & y[idx]))
        fp = float(np.sum(pred[idx] & ~y[idx]))
        fn = float(np.sum(~pred[idx] & y[idx]))
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / (tp + fn) if tp + fn else 1.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    all_idx = np.arange(len(usable))
    p, r, f = prf(all_idx)
    rng = np.random.default_rng(3)
    b = np.asarray([prf(rng.integers(0, len(usable), len(usable))) for _ in range(n_boot)])
    lo, hi = np.percentile(b, 2.5, axis=0), np.percentile(b, 97.5, axis=0)
    out.update(
        {
            "status": "partial" if len(labelled) < len(pairs) else "complete",
            "usable_pairs": len(usable),
            "precision": p,
            "recall": r,
            "f1": f,
            "precision_ci": [float(lo[0]), float(hi[0])],
            "recall_ci": [float(lo[1]), float(hi[1])],
            "f1_ci": [float(lo[2]), float(hi[2])],
        }
    )
    if auto:
        agree = [auto.get(k) == labelled[k] for k in usable if k in auto]
        out["auto_vs_hand_agreement"] = float(np.mean(agree)) if agree else None
    return out


def main() -> None:
    res = evaluate()
    Path("results").mkdir(exist_ok=True)
    Path("results/hand_eval.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
