# ADR 0003: Dev/test split by corporate group

Status: accepted (2026-11-20)

Records are split by the connected component of the parent-child graph (the corporate
group), hashed with SHA-1. A parent and its subsidiaries are therefore always in the same
split, so the hardest negatives (Siemens AG vs Siemens Healthineers AG) never leak from dev
to test. DEV is further split into train (model fitting) and tune (threshold / constraint
selection). TEST is evaluated once with frozen models.
