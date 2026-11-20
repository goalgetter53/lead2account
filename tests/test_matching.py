import numpy as np

from l2a.blocking import BlockIndex, pair_completeness
from l2a.cluster import CannotLink, cluster
from l2a.features import FEATURES, compare, feature_matrix
from l2a.hierarchy import Hierarchy
from l2a.metrics import duplicate_rates, lead_prf_ci, pairwise_counts, pairwise_prf_ci
from l2a.model import PairModel
from l2a.pipeline import LeadMatcher
from l2a.records import Idf, make_record

ACCS = [
    make_record("a0", "Siemens Aktiengesellschaft", "https://www.siemens.com", country="DE"),
    make_record("a1", "Siemens", "siemens.com", country="DE"),
    make_record("a2", "Siemens Healthineers AG", "siemens-healthineers.com", country="DE"),
    make_record("a3", "Acme Ltd", "acme.co.uk", country="GB"),
    make_record("a4", "Acme Inc", "acme.com", country="US"),
]


def test_blocking_finds_true_pair() -> None:
    idx = BlockIndex(ACCS)
    pc = pair_completeness(idx.pairs(), ["S", "S", "H", "A1", "A2"])
    assert pc["pair_completeness"] == 1.0


def test_features_shape_and_values() -> None:
    idf = Idf(ACCS)
    f = compare(ACCS[0], ACCS[1], idf)
    assert f["name_core_equal"] == 1.0
    assert f["domain_site_equal"] == 1.0
    assert feature_matrix([(ACCS[0], ACCS[2])], idf).shape == (1, len(FEATURES))


def test_cannot_link_country_without_domain() -> None:
    edges = [(0.99, 3, 4), (0.95, 0, 1)]
    labels, refused = cluster(ACCS, edges, 0.5, CannotLink(False, True))
    assert labels[3] != labels[4]
    assert labels[0] == labels[1]
    assert refused == 1
    labels2, _ = cluster(ACCS, edges, 0.5, CannotLink(False, False))
    assert labels2[3] == labels2[4]


def test_registry_cannot_link() -> None:
    recs = [make_record("x", "A", registry_id="L1"), make_record("y", "A", registry_id="L2")]
    labels, _ = cluster(recs, [(0.9, 0, 1)], 0.5, CannotLink(True, False))
    assert labels[0] != labels[1]


def test_hierarchy_rollup_and_cycles() -> None:
    h = Hierarchy(
        [("c", "p", "gleif_direct"), ("p", "u", "wikidata_p749"), ("x", "y", "a"), ("y", "x", "a")]
    )
    assert h.ancestors("c") == ["p", "u"]
    assert h.ultimate_parent("c") == "u"
    assert h.rollup("c", {"c", "p"}) == "p"
    assert len(h.ancestors("x")) <= 2


def test_metrics() -> None:
    pred = [0, 0, 1, 2]
    truth = ["a", "a", "a", "b"]
    assert pairwise_counts(pred, truth) == (1, 0, 2)
    r = pairwise_prf_ci(pred, truth, n_boot=20)
    assert r["precision"] == 1.0
    assert abs(r["recall"] - 1 / 3) < 1e-9
    d = duplicate_rates(pred, truth)
    assert d["records_per_entity_before"] == 2.0
    assert d["accounts_per_entity_after"] == 1.5
    lp = lead_prf_ci([("g", "correct"), ("g", "missed"), ("h", "true_none")], n_boot=10)
    assert lp["precision"] == 1.0 and lp["recall"] == 0.5 and lp["no_match_accuracy"] == 1.0


def _toy_model() -> PairModel:
    coef = np.zeros(len(FEATURES))
    coef[FEATURES.index("name_core_equal")] = 4.0
    coef[FEATURES.index("domain_site_equal")] = 4.0
    return PairModel(coef=coef, intercept=-3.0, threshold=0.5)


def test_lead_matcher_reasons_and_freemail() -> None:
    idf = Idf(ACCS)
    labels = [0, 0, 2, 3, 4]
    m = LeadMatcher(ACCS, labels, _toy_model(), idf)
    res = m.match(make_record("l1", "siemens ag", email="info@siemens.com"))
    assert res.account == 0
    assert "DOMAIN_EXACT" in res.reasons
    res2 = m.match(make_record("l2", "Totally Unknown Co", email="x@gmail.com"))
    assert res2.account is None


def test_lead_matcher_parent_rollup() -> None:
    accs = [
        make_record("a", "Siemens Healthineers AG", "siemens-healthineers.com", registry_id="C"),
        make_record("b", "Siemens AG", "siemens.com", registry_id="P"),
    ]
    h = Hierarchy([("C", "P", "gleif_direct")])
    m = LeadMatcher(accs, [0, 1], _toy_model(), Idf(accs), h, route="ultimate_parent")
    res = m.match(make_record("l", "Siemens Healthineers", website="siemens-healthineers.com"))
    assert res.account == 0 and res.routed_account == 1
    assert "PARENT_ROLLUP" in res.reasons


def test_model_roundtrip(tmp_path: object) -> None:
    from pathlib import Path

    x = np.random.default_rng(0).random((60, len(FEATURES)))
    y = (x[:, 0] > 0.5).astype(int)
    m = PairModel.fit(x, y)
    m.tune_threshold(m.proba(x), y)
    p = Path(str(tmp_path)) / "m.json"
    m.to_json(p)
    m2 = PairModel.from_json(p)
    assert np.allclose(m.proba(x), m2.proba(x), atol=1e-4)
