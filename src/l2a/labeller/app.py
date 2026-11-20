"""Local hand-labelling tool: FastAPI + a single HTML page.

Run:  make label   (or: python -m l2a.labeller.app --port 8765)
Then open http://127.0.0.1:8765. Labels are written to labels/hand_labels.csv
after every click, so closing the tab loses nothing.
"""

from __future__ import annotations

import argparse
import csv
import threading
from importlib import resources
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

VALID = {"match", "no_match", "parent_child", "unsure", ""}


class LabelIn(BaseModel):
    pair_id: str
    label: str
    note: str = ""


class Store:
    def __init__(self, pairs_csv: Path, labels_csv: Path) -> None:
        self.pairs_csv = pairs_csv
        self.labels_csv = labels_csv
        self.lock = threading.Lock()
        with pairs_csv.open(encoding="utf-8") as fh:
            self.pairs = list(csv.DictReader(fh))
        self.labels: dict[str, dict[str, str]] = {
            p["pair_id"]: {"label": "", "note": ""} for p in self.pairs
        }
        if labels_csv.exists():
            with labels_csv.open(encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if row["pair_id"] in self.labels:
                        self.labels[row["pair_id"]] = {
                            "label": row.get("label", ""),
                            "note": row.get("note", ""),
                        }

    def save(self) -> None:
        tmp = self.labels_csv.with_suffix(".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["pair_id", "label", "note"])
            for p in self.pairs:
                lab = self.labels[p["pair_id"]]
                w.writerow([p["pair_id"], lab["label"], lab["note"]])
        tmp.replace(self.labels_csv)

    def export(self) -> str:
        with self.labels_csv.open(encoding="utf-8") as fh:
            return fh.read()


def create_app(pairs_csv: Path, labels_csv: Path) -> FastAPI:
    store = Store(pairs_csv, labels_csv)
    app = FastAPI(title="lead2account labeller")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return resources.files("l2a.labeller").joinpath("static/index.html").read_text("utf-8")

    @app.get("/api/pairs")
    def pairs() -> list[dict[str, Any]]:
        # model_score is withheld so it cannot anchor the labeller
        keep = [k for k in store.pairs[0] if k != "model_score"] if store.pairs else []
        return [{**{k: p[k] for k in keep}, **store.labels[p["pair_id"]]} for p in store.pairs]

    @app.post("/api/label")
    def label(item: LabelIn) -> dict[str, Any]:
        if item.pair_id not in store.labels:
            raise HTTPException(404, "unknown pair_id")
        if item.label not in VALID:
            raise HTTPException(422, f"label must be one of {sorted(VALID)}")
        with store.lock:
            store.labels[item.pair_id] = {"label": item.label, "note": item.note}
            store.save()
        done = sum(1 for v in store.labels.values() if v["label"])
        return {"ok": True, "done": done, "total": len(store.labels)}

    @app.get("/api/export", response_class=PlainTextResponse)
    def export() -> str:
        return store.export()

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="labels/pairs_to_label.csv")
    ap.add_argument("--labels", default="labels/hand_labels.csv")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    uvicorn.run(create_app(Path(a.pairs), Path(a.labels)), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
