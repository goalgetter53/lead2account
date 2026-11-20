from l2a.domains import is_freemail, parse_domain


def test_registrable_with_multi_label_suffix() -> None:
    d = parse_domain("https://shop.bbc.co.uk/path")
    assert d.registrable == "bbc.co.uk"
    assert d.site_key == "shop.bbc.co.uk"


def test_country_path_site_key() -> None:
    d = parse_domain("https://www.microsoft.com/en-ca/")
    assert d.registrable == "microsoft.com"
    assert d.site_key == "microsoft.com/en-ca"
    assert parse_domain("microsoft.com").site_key == "microsoft.com"


def test_email_and_www() -> None:
    assert parse_domain("sales@www.siemens.com").site_key == "siemens.com"


def test_freemail() -> None:
    assert parse_domain("x@yahoo.co.uk").is_freemail
    assert parse_domain("x@gmail.com").is_freemail
    assert is_freemail("web.de")
    assert not parse_domain("x@siemens.de").is_freemail


def test_private_suffix_and_garbage() -> None:
    assert parse_domain("foo.github.io").registrable == "foo.github.io"
    assert parse_domain("").registrable == ""
    assert parse_domain("not a url").registrable == ""
