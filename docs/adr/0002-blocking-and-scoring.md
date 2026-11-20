# ADR 0002: Multi-key blocking + logistic regression + constrained greedy clustering

Status: accepted (2026-11-20)

## Decision
* Blocking: union of keys (registrable domain, domain label, normalised core name, joined
  key tokens, first two tokens, first token, prefix|suffix of joined name, acronym). Blocks
  larger than 300 records are dropped. Pair completeness and reduction ratio are reported.
* Scoring: 19 interpretable features and a logistic regression (coefficients stored as JSON,
  no pickles). Chosen over Fellegi-Sunter/EM because labels are plentiful from LEIs and LR
  copes with correlated features (domain equality vs site-key equality).
* Clustering: union-find over edges sorted by score, refusing merges that violate
  cannot-link rules: (a) clusters with disjoint country sets and no shared registrable
  domain, (b) any pair inside the merged cluster scored below a negative threshold (stops
  A~B~C chaining through a shared parent website), (c) optionally conflicting registry ids.
* Threshold and rule set are chosen on the DEV-tune half only, by cluster-level pairwise F1.

## Consequences
Greedy clustering is fast (tens of thousands of records per second) and explainable; it is
not globally optimal (correlation clustering would be), which is listed as a limitation.
