import pytest

from l2a.normalize import acronym, normalize_name


@pytest.mark.parametrize(
    ("raw", "core", "form"),
    [
        ("Siemens Aktiengesellschaft", "siemens", "ag"),
        ("Tata Consultancy Services Pvt. Ltd.", "tata consultancy services", "ltd"),
        ("PT Bank Rakyat Indonesia (Persero) Tbk", "bank rakyat indonesia", "tbk"),
        ("OOO Gazprom", "gazprom", "ooo"),
        ("Shell Public Limited Company", "shell", "plc"),
        ("Nestlé S.A.", "nestle", "sa"),
        ("Sony Group Kabushiki Kaisha", "sony group", "kk"),
        ("Volkswagen AG & Co. KGaA", "volkswagen", "ag"),
        ("ООО Ромашка", "romashka", "ooo"),
        ("The Coca-Cola Company", "coca cola", "co"),
    ],
)
def test_legal_forms(raw: str, core: str, form: str) -> None:
    n = normalize_name(raw)
    assert n.core == core
    assert n.legal_form == form


def test_never_strips_everything() -> None:
    assert normalize_name("Limited").core == "limited"


def test_key_tokens_drop_generic_words() -> None:
    assert normalize_name("Sony Group Corporation").key_tokens == ("sony",)


def test_acronym() -> None:
    assert acronym(normalize_name("International Business Machines Corp").key_tokens) == "bm"
    assert acronym(("bank", "of", "america")) == "ba"
