import pytest

from vault_to_voice.vault.links import find_links, parse_link, render_links, work_id_for


@pytest.mark.parametrize(
    ("inner", "expected"),
    [
        ("Witcher 3", ("Witcher 3", "Witcher 3")),
        ("Witcher 3|W3", ("Witcher 3", "W3")),
        ("Witcher 3#Combat", ("Witcher 3", "Witcher 3")),
        ("Games/Witcher 3#Combat|the Witcher", ("Witcher 3", "the Witcher")),
        ("Witcher 3.md", ("Witcher 3", "Witcher 3")),
    ],
)
def test_parse_link(inner, expected):
    assert parse_link(inner) == expected


def test_find_links_skips_embeds_and_code():
    text = "Played [[Hades]] then `[[Not Link]]` and ![[cover.png]] and [[Dead Cells|DC]]."
    assert find_links(text) == ["Hades", "Dead Cells"]


def test_render_links():
    text = "Played [[Hades|Hades (2020)]] ![[cover.png]] after [[Games/Dune#Ch1]]."
    assert render_links(text) == "Played Hades (2020)  after Dune."


def test_work_id_for():
    assert work_id_for("Witcher 3") == work_id_for("witcher 3") == "witcher-3"
    assert work_id_for("日本").startswith("work-")


def test_block_reference_is_same_work():
    assert parse_link("Hades^abc") == ("Hades", "Hades")
    assert parse_link("Hades#^abc|H") == ("Hades", "H")
