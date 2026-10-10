from vault_to_voice.style.stats import compute_stats, split_sentences, top_phrases


def test_split_sentences():
    assert split_sentences("I loved it. Honestly, it was great! Was it?") == [
        "I loved it.",
        "Honestly, it was great!",
        "Was it?",
    ]


def test_basic_stats():
    s = compute_stats(["I loved it. Honestly, it was great! Was it?"])
    assert (s.n_sentences, s.n_words) == (3, 9)
    assert s.avg_sentence_len == 3.0 and s.median_sentence_len == 3.0
    assert s.first_person_rate == 0.333
    assert s.punctuation["exclamation"] > 0 and s.punctuation["question"] > 0


def test_em_dash_and_paragraphs():
    s = compute_stats(["One — two.\n\nThree. Four."])
    assert s.punctuation["em_dash"] > 0
    assert s.avg_paragraph_len == 1.5


def test_empty_input_no_division_error():
    s = compute_stats([])
    assert s.n_sentences == 0 and s.avg_sentence_len == 0.0 and s.top_phrases == []


def test_top_phrases_need_repeats_across_notes():
    texts = ["the thing is it works", "the thing is it fails", "and the thing is again"]
    phrases = top_phrases(texts)
    assert "the thing is" in phrases
    assert "it works" not in phrases
