# Owner setup: hand-labelling 500 pairs (about 60-90 minutes)

Nothing here costs money or needs an account. Everything runs on your laptop.

1. Install Python 3.11, then in this folder run:
   `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
2. `pip install -e '.[dev]'`
3. Start the labeller: `make label` (or `python -m l2a.labeller.app`).
4. Open http://127.0.0.1:8765 in your browser.
5. For each pair press a key:
   * `1` Same company (same legal entity, or same company under another name/domain)
   * `2` Different company
   * `3` Parent / subsidiary (same group, different legal entities)
   * `4` Unsure
   * `←` / `→` move, `N` jumps to the next unlabelled pair. You can add a note.
   Every click is saved to `labels/hand_labels.csv` immediately; you can stop and resume.
   The model score and the automatic label are hidden on purpose.
6. When done (or part-way): `make hand-eval report`. This writes `results/hand_eval.json`
   and refreshes `results/report.md` / `results/report.html`.
7. Commit `labels/hand_labels.csv` and the refreshed `results/` files, then update the
   "Hand-labelled evaluation" line in README.md with the numbers printed by `make hand-eval`.

Docker alternative: `docker build -t l2a . && docker run -p 8765:8765 -v "$PWD/labels:/app/labels" l2a`
