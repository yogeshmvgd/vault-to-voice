from tests.fakes import FakeLLMClient
from vault_to_voice.style.profile import (
    ToneSummary,
    build_style_profile,
    load_profile,
    save_profile,
)
from vault_to_voice.vault.models import Note

NOTES = [
    Note(
        id=f"n{i}",
        title="t",
        body=f"Honestly? The fight is fair. Note number {i}.",
        path=f"n{i}.md",
        hash="h",
    )
    for i in range(3)
]


def tone(exemplars):
    return ToneSummary(tone=["wry"], voice_notes="Short and punchy.", exemplars=exemplars)


def test_build_keeps_only_verbatim_exemplars():
    fake = FakeLLMClient({ToneSummary: tone(["The fight is fair.", "I invented this line."])})
    p = build_style_profile(NOTES, fake)
    assert p.tone.exemplars == ["The fight is fair."]
    assert p.stats.n_notes == 3 and p.version == 1


def test_sample_size_limits_prompt():
    fake = FakeLLMClient({ToneSummary: tone([])})
    build_style_profile(NOTES, fake, sample_size=2)
    assert fake.calls[0][1].count("<note id=") == 2


def test_save_bumps_version_and_roundtrips(tmp_path):
    path = tmp_path / "style.json"
    fake = FakeLLMClient({ToneSummary: lambda _: tone([])})
    first = save_profile(build_style_profile(NOTES, fake), path)
    second = save_profile(build_style_profile(NOTES, fake), path)
    assert (first.version, second.version) == (1, 2)
    assert load_profile(path).version == 2


def test_load_missing_returns_none(tmp_path):
    assert load_profile(tmp_path / "nope.json") is None
