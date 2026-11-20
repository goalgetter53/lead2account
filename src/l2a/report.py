"""Render results/metrics.json (+ results/hand_eval.json) to Markdown and HTML."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

RESULTS = Path("results")


def _ci(d: dict[str, Any], k: str) -> str:
    lo, hi = d[f"{k}_ci"]
    return f"{d[k]:.3f} [{lo:.3f}, {hi:.3f}]"


def render_md(m: dict[str, Any], hand: dict[str, Any]) -> str:
    ds = m["dataset"]
    dd = m["test_dedupe"]
    reg = m["test_dedupe_with_registry_cannot_link"]
    base = m["test_baseline_exact_name_or_site"]
    ld = m["test_leads"]
    ru = m["test_parent_rollup"]
    dup = dd["duplicates"]
    lines = [
        "# lead2account: evaluation report",
        "",
        "All numbers come from `results/metrics.json`, written by `python -m l2a.experiment`.",
        "**Labels here are automatic** (records share an LEI from GLEIF / Wikidata P1278). "
        f"Hand-labelled evaluation: **{hand.get('status', 'pending')}**.",
        "",
        "## Data",
        "",
        f"- {ds['total_records']:,} records: {ds['account_records']:,} CRM account records + "
        f"{ds['lead_records']:,} leads, covering {ds['entities']:,} real legal entities "
        f"({ds['entities_in_crm']:,} with CRM accounts) in {ds['groups']:,} corporate groups.",
        f"- {ds['hierarchy_edges']:,} parent-child edges: "
        + ", ".join(f"{k} {v:,}" for k, v in ds["edge_sources"].items()),
        f"- Split by corporate group: TEST = {m['split_sizes']['test_accounts']:,} accounts, "
        f"{m['split_sizes']['test_leads']:,} leads, "
        f"{m['split_sizes']['test_entity_groups']:,} groups. "
        "Models and thresholds were fitted/tuned on DEV only.",
        "",
        "## Account dedupe (TEST, pairwise, 95% bootstrap CI over evaluation components)",
        "",
        "| Method | Precision | Recall | F1 |",
        "|---|---|---|---|",
        f"| Baseline: exact normalised name OR same site | {_ci(base['pairwise'], 'precision')} | "
        f"{_ci(base['pairwise'], 'recall')} | {_ci(base['pairwise'], 'f1')} |",
        f"| **l2a** (LR + constrained clustering, t={dd['config']['threshold']}, cannot-link: "
        f"{dd['config']['cannot_link']}) | {_ci(dd['pairwise'], 'precision')} | "
        f"{_ci(dd['pairwise'], 'recall')} | {_ci(dd['pairwise'], 'f1')} |",
        f"| l2a + registry-id cannot-link | {_ci(reg['pairwise'], 'precision')} | "
        f"{_ci(reg['pairwise'], 'recall')} | {_ci(reg['pairwise'], 'f1')} |",
        "",
        "## Duplicate rate (TEST)",
        "",
        "| | Before | After l2a merge | After baseline |",
        "|---|---|---|---|",
        f"| Accounts per true entity | {dup['records_per_entity_before']:.3f} | "
        f"{dup['accounts_per_entity_after']:.3f} | "
        f"{base['duplicates']['accounts_per_entity_after']:.3f} |",
        f"| Duplicate rate | {dup['duplicate_rate_before']:.1%} | "
        f"{dup['duplicate_rate_after']:.1%} | "
        f"{base['duplicates']['duplicate_rate_after']:.1%} |",
        f"| Over-merged clusters (>1 entity) | 0 | {dup['overmerged_cluster_rate']:.2%} | "
        f"{base['duplicates']['overmerged_cluster_rate']:.2%} |",
        "",
        "## Lead-to-account matching (TEST)",
        "",
        f"- Precision {_ci(ld, 'precision')}, recall {_ci(ld, 'recall')}, F1 {_ci(ld, 'f1')} "
        f"over {int(ld['n']):,} leads; correct no-match for leads with no account: "
        f"{ld['no_match_accuracy']:.3f}.",
        "",
        "| Hard case | Leads | Precision | Recall |",
        "|---|---|---|---|",
    ]
    for h, d in m["test_leads_by_hard_case"].items():
        lines.append(f"| {h} | {int(d['n']):,} | {_ci(d, 'precision')} | {_ci(d, 'recall')} |")
    lines += [
        "",
        f"- Parent rollup routing (route_to=ultimate_parent, {int(ru['n']):,} leads whose entity "
        f"has a parent account): routed to the correct top account {_ci(ru, 'precision')} "
        f"(precision), recall {_ci(ru, 'recall')}.",
        "",
        "## Blocking",
        "",
        "| Split | Candidate pairs | Pair completeness | Reduction ratio |",
        "|---|---|---|---|",
    ]
    for s, b in m["blocking"].items():
        lines.append(
            f"| {s} | {int(b['candidate_pairs']):,} | {b['pair_completeness']:.4f} | "
            f"{b['reduction_ratio']:.5f} |"
        )
    lines += [
        "",
        "## Throughput (2 vCPU container, single process)",
        "",
        f"- Dedupe: {dd['throughput_records_per_sec']:,.0f} records/s "
        f"({int(dd['candidate_pairs']):,} scored pairs).",
        f"- Lead matching: {ld['throughput_leads_per_sec']:,.0f} leads/s.",
        "",
        "## Hand-labelled evaluation",
        "",
    ]
    if hand.get("usable_pairs"):
        lines.append(
            f"- {hand['labelled']}/{hand['total']} labelled. "
            f"Model precision {_ci(hand, 'precision')}, "
            f"recall {_ci(hand, 'recall')}; automatic-vs-hand label agreement "
            f"{hand.get('auto_vs_hand_agreement')}."
        )
    else:
        lines.append(
            f"- Pending: {hand.get('labelled', 0)}/{hand.get('total', 500)} labelled. "
            "Run `make label`, then `make hand-eval report`."
        )
    return "\n".join(lines) + "\n"


def md_to_html(md: str) -> str:
    out = [
        "<!doctype html><html><head><meta charset='utf-8'><title>lead2account report</title>",
        "<style>body{font:15px/1.5 system-ui,sans-serif;max-width:960px;margin:32px auto;"
        "padding:0 16px;color:#1d1d1f}table{border-collapse:collapse;margin:8px 0}"
        "td,th{border:1px solid #ddd;padding:4px 10px;text-align:left}th{background:#f4f4f6}"
        "code{background:#f4f4f6;padding:1px 4px}</style></head><body>",
    ]
    in_table = False
    for line in md.splitlines():
        esc = html.escape(line)
        esc = _inline(esc)
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= {"-"}:
                continue
            if not in_table:
                out.append("<table>")
                in_table = True
                out.append(
                    "<tr>" + "".join(f"<th>{_inline(html.escape(c))}</th>" for c in cells) + "</tr>"
                )
            else:
                out.append(
                    "<tr>" + "".join(f"<td>{_inline(html.escape(c))}</td>" for c in cells) + "</tr>"
                )
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        if line.startswith("## "):
            out.append(f"<h2>{_inline(html.escape(line[3:]))}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{_inline(html.escape(line[2:]))}</h1>")
        elif line.startswith("- "):
            out.append(f"<p>&bull; {_inline(html.escape(line[2:]))}</p>")
        elif line.strip():
            out.append(f"<p>{esc}</p>")
    if in_table:
        out.append("</table>")
    out.append("</body></html>")
    return "\n".join(out) + "\n"


def _inline(s: str) -> str:
    import re

    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", s)


def main() -> None:
    m = json.loads((RESULTS / "metrics.json").read_text())
    hp = RESULTS / "hand_eval.json"
    hand = json.loads(hp.read_text()) if hp.exists() else {"status": "pending (0/500 labelled)"}
    md = render_md(m, hand)
    (RESULTS / "report.md").write_text(md)
    (RESULTS / "report.html").write_text(md_to_html(md))
    print("wrote results/report.md and results/report.html")


if __name__ == "__main__":
    main()
