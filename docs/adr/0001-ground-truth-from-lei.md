# ADR 0001: Automatic ground truth from shared LEI, hand labels kept separate

Status: accepted (2026-11-20)

## Context
Entity-resolution results are only as honest as the labels. We need tens of thousands of
labelled pairs to measure blocking recall and pairwise P/R with tight intervals, and we
also need human judgement on hard cases (rebrands, shared parent domains, subsidiaries).

## Decision
* Every CRM-like record is generated from a real legal entity identified by its GLEIF LEI
  (Wikidata items joined through P1278). Two records are a true match iff they share an LEI.
  This gives automatic labels for every pair.
* The LEI is never a matching feature. Only `erp_import` records carry it, as a CRM
  "registry id" would, and it is used for hierarchy linking and for an *optional*
  registry cannot-link rule that is reported separately.
* 500 TEST pairs are sampled (stratified to over-weight hard cases) for Hussain to label in
  a local tool. Hand labels are evaluated separately and also used to measure how noisy the
  automatic labels are (`auto_vs_hand_agreement`).

## Consequences
Automatic labels treat parent and subsidiary as different entities (correct for LEI, and for
LeanData-style "match to child" routing), even where a salesperson would call them "the same
company". The hand labels have an explicit `parent_child` class so this gap can be measured.
