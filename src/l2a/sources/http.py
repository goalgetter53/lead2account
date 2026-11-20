"""Tiny stdlib HTTP helper (urllib) with a project User-Agent and retries.

Wikidata's SPARQL endpoint rejects some HTTP clients; plain urllib with a
descriptive UA works, so every fetcher goes through here.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

USER_AGENT = "lead2account/0.1 (+https://github.com/goalgetter53/lead2account)"


def get_json(url: str, params: dict[str, str] | None = None, retries: int = 6) -> Any:
    return json.loads(get_text(url, params, retries, accept="application/json"))


def get_text(
    url: str,
    params: dict[str, str] | None = None,
    retries: int = 6,
    accept: str = "application/json",
) -> str:
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    delay = 2.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                text: str = resp.read().decode("utf-8")
                if accept == "application/json":
                    json.loads(text)  # validate: truncated bodies raise ValueError
                return text
        except (urllib.error.URLError, TimeoutError, ValueError, ConnectionError) as exc:
            if attempt == retries - 1:
                raise
            code = getattr(exc, "code", None)
            time.sleep(70.0 if code == 429 else delay)
            delay *= 2
    raise RuntimeError("unreachable")
