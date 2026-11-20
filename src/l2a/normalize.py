"""Company-name normalisation: unicode folding, legal forms, core tokens."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from unidecode import unidecode

# Canonical legal-form families. Keys are space-joined token sequences as they
# appear AFTER `basic_clean` (dots removed, punctuation -> space, lowercase).
_LEGAL_FORMS: dict[str, str] = {}


def _add(family: str, *forms: str) -> None:
    for f in forms:
        _LEGAL_FORMS[f] = family


_add("inc", "inc", "incorporated", "corp", "corporation", "co inc", "company inc")
_add("llc", "llc", "l l c", "limited liability company", "llp", "lp", "l p", "lllp", "pllc")
_add(
    "ltd",
    "ltd",
    "limited",
    "co ltd",
    "company limited",
    "co limited",
    "pvt ltd",
    "private limited",
    "pvt limited",
    "private ltd",
    "pte ltd",
    "pte limited",
    "pty ltd",
    "pty limited",
    "pty",
    "ltda",
    "dac",
    "uc",
    "cic",
)
_add("plc", "plc", "public limited company", "p l c", "public company limited", "pcl")
_add("gmbh", "gmbh", "gesellschaft mit beschrankter haftung", "mbh", "gmbh co kg", "gmbh and co kg")
_add("ag", "ag", "aktiengesellschaft", "kgaa", "ag co kg", "se", "kg", "ohg", "eg", "ev")
_add(
    "sa",
    "sa",
    "s a",
    "societe anonyme",
    "sociedad anonima",
    "sa de cv",
    "sab de cv",
    "s a b de c v",
    "s a de c v",
    "saa",
    "sas",
    "sasu",
    "sarl",
    "sca",
    "scs",
    "snc",
    "sl",
    "slu",
    "sll",
    "spa",
    "s p a",
    "srl",
    "s r l",
    "sapa",
    "scrl",
    "sc",
    "sae",
    "saog",
    "saic",
    "sab",
)
_add(
    "nv",
    "nv",
    "n v",
    "bv",
    "b v",
    "naamloze vennootschap",
    "besloten vennootschap",
    "cv",
    "vof",
    "bvba",
    "srl bv",
)
_add(
    "ab",
    "ab",
    "publ",
    "ab publ",
    "asa",
    "as",
    "a s",
    "aps",
    "a s a",
    "oyj",
    "oy",
    "hf",
    "ehf",
    "a/s",
)
_add(
    "kk",
    "kk",
    "k k",
    "kabushiki kaisha",
    "kabushikigaisha",
    "godo kaisha",
    "gk",
    "yk",
    "yugen kaisha",
)
_add(
    "ooo",
    "ooo",
    "oao",
    "zao",
    "pao",
    "ao",
    "pjsc",
    "ojsc",
    "jsc",
    "cjsc",
    "tov",
    "pat",
    "prat",
    "ad",
    "dd",
    "doo",
    "d o o",
    "sro",
    "s r o",
    "as",
    "zrt",
    "nyrt",
    "kft",
    "sp z o o",
    "spolka akcyjna",
    "sp zoo",
)
_add(
    "tbk",
    "tbk",
    "pt",
    "pt tbk",
    "persero",
    "persero tbk",
    "bhd",
    "berhad",
    "sdn bhd",
    "sdn",
    "pcl",
    "jsc",
)
_add("co", "co", "company", "and co", "and company", "cos", "companies", "cie", "et cie")
_add("trust", "trust", "fund", "reit", "sicav", "ucits", "etf")

_MAX_FORM_TOKENS = max(len(k.split()) for k in _LEGAL_FORMS)
_PREFIX_FORMS = {
    "pt",
    "ooo",
    "oao",
    "zao",
    "pao",
    "ao",
    "pjsc",
    "ojsc",
    "jsc",
    "cjsc",
    "tov",
    "the",
    "pat",
    "prat",
}
GENERIC_TOKENS = frozenset(
    {
        "the",
        "group",
        "holding",
        "holdings",
        "international",
        "intl",
        "global",
        "worldwide",
        "and",
        "of",
        "de",
        "du",
        "des",
        "la",
        "le",
        "der",
        "die",
        "das",
        "y",
        "e",
        "i",
        "et",
        "und",
    }
)

# CJK legal forms are removed before transliteration (unidecode would turn
# 株式会社 into four generic syllables shared by every Japanese company).
_CJK_FORMS = re.compile(
    r"(株式会社|有限会社|合同会社|合資会社|合名会社|（株）|\(株\)|㈱|股份有限公司|有限责任公司|"
    r"有限公司|集团|주식회사|\(주\)|㈜|ООО|ОАО|ЗАО|ПАО|АО)"
)
_CJK_FAMILY = {
    "有限公司": "ltd",
    "有限责任公司": "ltd",
    "股份有限公司": "ltd",
    "集团": "ltd",
    "ООО": "ooo",
    "ОАО": "ooo",
    "ЗАО": "ooo",
    "ПАО": "ooo",
    "АО": "ooo",
}
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_PAREN = re.compile(r"\([^)]*\)")


@lru_cache(maxsize=200_000)
def basic_clean(name: str) -> str:
    """Fold to ASCII, lowercase, drop dots, map punctuation to spaces."""
    s = unidecode(name or "").lower()
    s = s.replace("&", " and ").replace("+", " and ").replace("'", "").replace("’", "")
    s = s.replace(".", "")
    s = _NON_ALNUM.sub(" ", s)
    return " ".join(s.split())


@dataclass(frozen=True)
class NormName:
    raw: str
    clean: str
    core: str
    legal_form: str
    tokens: tuple[str, ...]
    key_tokens: tuple[str, ...]


def strip_legal_forms(clean: str) -> tuple[str, str]:
    """Remove trailing (and a few leading) legal-form token runs.

    Returns (core, family) where family is the first legal-form family found
    (from the end) or "" when none.
    """
    toks = clean.split()
    family = ""
    changed = True
    while changed and len(toks) > 1:
        changed = False
        for n in range(min(_MAX_FORM_TOKENS, len(toks) - 1), 0, -1):
            tail = " ".join(toks[-n:])
            if tail in _LEGAL_FORMS:
                family = family or _LEGAL_FORMS[tail]
                toks = toks[:-n]
                changed = True
                break
    while len(toks) > 1 and toks[0] in _PREFIX_FORMS:
        if toks[0] != "the":
            family = family or _LEGAL_FORMS.get(toks[0], "")
        toks = toks[1:]
    return " ".join(toks), family


@lru_cache(maxsize=200_000)
def normalize_name(name: str) -> NormName:
    cjk = _CJK_FORMS.search(name or "")
    if cjk and _CJK_FORMS.sub("", name).strip(" \u3000"):
        stripped = _CJK_FORMS.sub(" ", name)
        n = normalize_name(stripped)
        return NormName(
            name, n.clean, n.core, _CJK_FAMILY.get(cjk.group(0), "kk"), n.tokens, n.key_tokens
        )
    no_paren = _PAREN.sub(" ", name or "")
    clean = basic_clean(no_paren) or basic_clean(name)
    core, family = strip_legal_forms(clean)
    tokens = tuple(core.split())
    key = tuple(t for t in tokens if t not in GENERIC_TOKENS) or tokens
    return NormName(
        raw=name, clean=clean, core=core, legal_form=family, tokens=tokens, key_tokens=key
    )


def acronym(tokens: tuple[str, ...]) -> str:
    return "".join(t[0] for t in tokens if t and t not in GENERIC_TOKENS)
