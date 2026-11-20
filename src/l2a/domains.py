"""Domain normalisation: public-suffix eTLD+1, site keys, free-mail detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

_GENERIC_SUBDOMAINS = frozenset(
    {
        "www",
        "www1",
        "www2",
        "www3",
        "m",
        "mobile",
        "web",
        "home",
        "corporate",
        "corp",
        "about",
        "en",
        "group",
    }
)
_LOCALE = re.compile(r"^[a-z]{2}([-_][a-z]{2,3})?$")

# Matched on the registrable domain's first label (yahoo.co.uk -> "yahoo") or
# on the full registrable domain.
FREEMAIL_LABELS = frozenset(
    {
        "gmail",
        "googlemail",
        "yahoo",
        "ymail",
        "rocketmail",
        "hotmail",
        "outlook",
        "live",
        "msn",
        "aol",
        "icloud",
        "protonmail",
        "gmx",
        "yandex",
        "zohomail",
        "tutanota",
        "tutamail",
        "fastmail",
        "mailinator",
        "rediffmail",
        "naver",
        "hanmail",
        "libero",
        "seznam",
        "inbox",
    }
)
FREEMAIL_DOMAINS = frozenset(
    {
        "me.com",
        "mac.com",
        "proton.me",
        "pm.me",
        "web.de",
        "t-online.de",
        "mail.ru",
        "bk.ru",
        "list.ru",
        "qq.com",
        "163.com",
        "126.com",
        "sina.com",
        "sohu.com",
        "daum.net",
        "orange.fr",
        "free.fr",
        "laposte.net",
        "btinternet.com",
        "comcast.net",
        "verizon.net",
        "att.net",
        "sbcglobal.net",
        "wp.pl",
        "o2.pl",
        "interia.pl",
        "bol.com.br",
        "uol.com.br",
        "terra.com.br",
        "hey.com",
        "mail.com",
        "email.com",
        "usa.net",
        "rambler.ru",
        "ukr.net",
    }
)


class PublicSuffixList:
    def __init__(self, text: str) -> None:
        self.rules: set[str] = set()
        self.wild: set[str] = set()
        self.exceptions: set[str] = set()
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            rule = line.split()[0].lower()
            if rule.startswith("!"):
                self.exceptions.add(rule[1:])
            elif rule.startswith("*."):
                self.wild.add(rule[2:])
            else:
                self.rules.add(rule)

    def suffix_len(self, labels: list[str]) -> int:
        """Number of labels in the public suffix of `labels` (default rule '*')."""
        best = 1
        n = len(labels)
        for i in range(n):
            cand = ".".join(labels[i:])
            if cand in self.exceptions:
                return n - i - 1
            if cand in self.rules:
                best = max(best, n - i)
            if i + 1 < n and ".".join(labels[i + 1 :]) in self.wild:
                best = max(best, n - i)
        return best

    def registrable(self, host: str) -> str:
        labels = [x for x in host.split(".") if x]
        if not labels:
            return ""
        k = self.suffix_len(labels)
        if k >= len(labels):
            return ".".join(labels)
        return ".".join(labels[-(k + 1) :])


@lru_cache(maxsize=1)
def psl() -> PublicSuffixList:
    text = resources.files("l2a").joinpath("data/public_suffix_list.dat").read_text("utf-8")
    return PublicSuffixList(text)


@dataclass(frozen=True)
class DomainInfo:
    host: str
    registrable: str
    site_key: str
    is_freemail: bool

    @property
    def label(self) -> str:
        """First label of the registrable domain (e.g. 'siemens' for siemens.co.uk)."""
        return self.registrable.split(".", 1)[0] if self.registrable else ""


_EMPTY = DomainInfo("", "", "", False)


def is_freemail(registrable: str) -> bool:
    if not registrable:
        return False
    return registrable in FREEMAIL_DOMAINS or registrable.split(".", 1)[0] in FREEMAIL_LABELS


@lru_cache(maxsize=200_000)
def parse_domain(value: str) -> DomainInfo:
    """Parse a URL, bare host or e-mail address into a DomainInfo.

    site_key keeps what distinguishes sites that share one registrable domain:
    a non-generic subdomain (``uk.example.com``) or a locale path segment
    (``microsoft.com/en-ca``).
    """
    v = (value or "").strip().lower()
    if not v:
        return _EMPTY
    if "@" in v and "/" not in v:
        v = v.rsplit("@", 1)[1]
    v = re.sub(r"^[a-z][a-z0-9+.-]*://", "", v)
    host, _, path = v.partition("/")
    host = host.split("?")[0].split("#")[0].rsplit("@", 1)[-1].split(":")[0].strip(".")
    if not host or "." not in host:
        return _EMPTY
    reg = psl().registrable(host)
    sub = host[: -len(reg)].rstrip(".") if host != reg else ""
    sub_labels = [s for s in sub.split(".") if s and s not in _GENERIC_SUBDOMAINS]
    site = reg
    if sub_labels:
        site = ".".join(sub_labels) + "." + reg
    first_seg = path.split("/", 1)[0].split("?")[0].split("#")[0]
    if first_seg and _LOCALE.match(first_seg) and first_seg not in {"en", "us", "www"}:
        site = site + "/" + first_seg.replace("_", "-")
    return DomainInfo(host=host, registrable=reg, site_key=site, is_freemail=is_freemail(reg))
