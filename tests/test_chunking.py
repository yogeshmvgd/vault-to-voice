from vault_to_voice.index.chunking import split_paragraphs


def test_splits_on_blank_lines_and_joins_lines():
    assert split_paragraphs("a\nb\n\n c \n\n\n d") == ["a b", "c", "d"]


def test_strips_heading_markers():
    assert split_paragraphs("# Title\nText") == ["Title Text"]


def test_crlf_and_empty():
    assert split_paragraphs("x\r\n\r\ny") == ["x", "y"]
    assert split_paragraphs("  \n\n ") == []
