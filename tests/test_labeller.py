import csv
from pathlib import Path

from fastapi.testclient import TestClient

from l2a.labeller.app import create_app


def _write_pairs(path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_id", "stratum", "left_kind", "left_name", "right_name", "model_score"])
        w.writerow(["P000", "acc_high", "account", "Siemens AG", "Siemens", "0.99"])
        w.writerow(["P001", "lead_freemail", "lead", "acme", "Acme Ltd", "0.40"])


def test_label_roundtrip(tmp_path: Path) -> None:
    pairs, labels = tmp_path / "pairs.csv", tmp_path / "labels.csv"
    _write_pairs(pairs)
    c = TestClient(create_app(pairs, labels))
    assert "labeller" in c.get("/").text
    got = c.get("/api/pairs").json()
    assert len(got) == 2 and "model_score" not in got[0]
    r = c.post("/api/label", json={"pair_id": "P001", "label": "parent_child", "note": "x"})
    assert r.json()["done"] == 1
    assert c.post("/api/label", json={"pair_id": "P001", "label": "bogus"}).status_code == 422
    assert c.post("/api/label", json={"pair_id": "NOPE", "label": "match"}).status_code == 404
    exported = c.get("/api/export").text
    assert "P001,parent_child,x" in exported
    # labels persist across restarts
    c2 = TestClient(create_app(pairs, labels))
    assert c2.get("/api/pairs").json()[1]["label"] == "parent_child"
