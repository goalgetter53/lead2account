# lead2account: evaluation report

All numbers come from `results/metrics.json`, written by `python -m l2a.experiment`.
**Labels here are automatic** (records share an LEI from GLEIF / Wikidata P1278). Hand-labelled evaluation: **pending (0/500 labelled)**.

## Data

- 51,581 records: 38,669 CRM account records + 12,912 leads, covering 17,987 real legal entities (16,075 with CRM accounts) in 11,492 corporate groups.
- 7,640 parent-child edges: gleif_direct 1,364, gleif_ultimate 5,767, wikidata_p355 234, wikidata_p749 275
- Split by corporate group: TEST = 19,330 accounts, 6,504 leads, 4,906 groups. Models and thresholds were fitted/tuned on DEV only.

## Account dedupe (TEST, pairwise, 95% bootstrap CI over evaluation components)

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Baseline: exact normalised name OR same site | 0.723 [0.673, 0.769] | 0.672 [0.662, 0.683] | 0.697 [0.673, 0.718] |
| **l2a** (LR + constrained clustering, t=0.15, cannot-link: country,neg<0.2) | 0.794 [0.776, 0.812] | 0.749 [0.739, 0.758] | 0.771 [0.760, 0.782] |
| l2a + registry-id cannot-link | 0.867 [0.834, 0.893] | 0.778 [0.769, 0.787] | 0.820 [0.801, 0.834] |

## Duplicate rate (TEST)

| | Before | After l2a merge | After baseline |
|---|---|---|---|
| Accounts per true entity | 2.389 | 1.281 | 1.382 |
| Duplicate rate | 58.1% | 21.9% | 27.6% |
| Over-merged clusters (>1 entity) | 0 | 7.43% | 5.33% |

## Lead-to-account matching (TEST)

- Precision 0.905 [0.897, 0.913], recall 0.877 [0.869, 0.886], F1 0.891 [0.883, 0.899] over 6,504 leads; correct no-match for leads with no account: 0.918.

| Hard case | Leads | Precision | Recall |
|---|---|---|---|
| alias_name | 2,630 | 0.928 [0.916, 0.940] | 0.870 [0.855, 0.884] |
| freemail | 1,696 | 0.891 [0.874, 0.907] | 0.858 [0.840, 0.877] |
| none | 1,469 | 0.937 [0.923, 0.950] | 0.957 [0.945, 0.968] |
| parent_domain | 709 | 0.815 [0.785, 0.841] | 0.806 [0.777, 0.833] |

- Parent rollup routing (route_to=ultimate_parent, 2,157 leads whose entity has a parent account): routed to the correct top account 0.841 [0.814, 0.867] (precision), recall 0.811 [0.783, 0.836].

## Blocking

| Split | Candidate pairs | Pair completeness | Reduction ratio |
|---|---|---|---|
| dev | 76,577 | 0.8124 | 0.99959 |
| test | 72,662 | 0.7950 | 0.99961 |

## Throughput (2 vCPU container, single process)

- Dedupe: 10,374 records/s (72,662 scored pairs).
- Lead matching: 5,971 leads/s.

## Hand-labelled evaluation

- Pending: 0/500 labelled. Run `make label`, then `make hand-eval report`.
