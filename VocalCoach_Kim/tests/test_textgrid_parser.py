"""
tests/test_textgrid_parser.py - Unit tests for reference/textgrid_parser.py

All tests use synthetic TextGrid strings written to temp files — no real
annotation files required. The built-in plain-text parser is tested directly
even when praatio/textgrid packages are available.

Run with:
    cd MusicAI/VocalCoach
    python -m pytest tests/test_textgrid_parser.py -v
"""

from __future__ import annotations

import textwrap

import pytest

from reference.textgrid_parser import (
    _parse_textgrid_plain,
    parse_textgrid,
)
from utils.types import ReferencePhoneme, ReferenceWord


# ---------------------------------------------------------------------------
# Synthetic TextGrid fixtures
# ---------------------------------------------------------------------------

_BASIC_TG = textwrap.dedent("""\
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 2

tiers? <exists>
size = 2
item []:
item [1]:
    class = "IntervalTier"
    name = "phonemes"
    xmin = 0
    xmax = 2
    intervals: size = 4
    intervals [1]:
        xmin = 0
        xmax = 0.3
        text = "AH"
    intervals [2]:
        xmin = 0.3
        xmax = 0.7
        text = "EH"
    intervals [3]:
        xmin = 0.7
        xmax = 1.0
        text = "SIL"
    intervals [4]:
        xmin = 1.0
        xmax = 2.0
        text = "IH"
item [2]:
    class = "IntervalTier"
    name = "words"
    xmin = 0
    xmax = 2
    intervals: size = 2
    intervals [1]:
        xmin = 0
        xmax = 0.7
        text = "HELLO"
    intervals [2]:
        xmin = 1.0
        xmax = 2.0
        text = "WORLD"
""")


_EMPTY_TIER_TG = textwrap.dedent("""\
File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 1
tiers? <exists>
size = 1
item []:
item [1]:
    class = "IntervalTier"
    name = "phonemes"
    xmin = 0
    xmax = 1
    intervals: size = 1
    intervals [1]:
        xmin = 0
        xmax = 1
        text = "AH"
""")


_SILENCE_TG = textwrap.dedent("""\
File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 1
tiers? <exists>
size = 1
item []:
item [1]:
    class = "IntervalTier"
    name = "phonemes"
    xmin = 0
    xmax = 1
    intervals: size = 3
    intervals [1]:
        xmin = 0
        xmax = 0.2
        text = ""
    intervals [2]:
        xmin = 0.2
        xmax = 0.6
        text = "AH"
    intervals [3]:
        xmin = 0.6
        xmax = 1.0
        text = "SIL"
""")


@pytest.fixture()
def basic_tg(tmp_path):
    f = tmp_path / "basic.TextGrid"
    f.write_text(_BASIC_TG, encoding="utf-8")
    return f


@pytest.fixture()
def empty_tier_tg(tmp_path):
    f = tmp_path / "empty_tier.TextGrid"
    f.write_text(_EMPTY_TIER_TG, encoding="utf-8")
    return f


@pytest.fixture()
def silence_tg(tmp_path):
    f = tmp_path / "silence.TextGrid"
    f.write_text(_SILENCE_TG, encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# Plain-text parser unit tests
# ---------------------------------------------------------------------------

class TestPlainTextParser:
    def test_parses_phoneme_tier(self):
        tiers = _parse_textgrid_plain(_BASIC_TG)
        assert "phonemes" in tiers

    def test_parses_word_tier(self):
        tiers = _parse_textgrid_plain(_BASIC_TG)
        assert "words" in tiers

    def test_phoneme_count(self):
        tiers = _parse_textgrid_plain(_BASIC_TG)
        assert len(tiers["phonemes"]) == 4

    def test_phoneme_timestamps(self):
        tiers = _parse_textgrid_plain(_BASIC_TG)
        start, end, label = tiers["phonemes"][0]
        assert start == pytest.approx(0.0)
        assert end == pytest.approx(0.3)
        assert label == "AH"

    def test_word_count(self):
        tiers = _parse_textgrid_plain(_BASIC_TG)
        assert len(tiers["words"]) == 2

    def test_missing_tier_not_in_result(self):
        tiers = _parse_textgrid_plain(_BASIC_TG)
        assert "nonexistent" not in tiers


# ---------------------------------------------------------------------------
# parse_textgrid — basic output types
# ---------------------------------------------------------------------------

class TestParseTextGridBasic:
    def test_returns_tuple(self, basic_tg):
        result = parse_textgrid(basic_tg)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_phoneme_list(self, basic_tg):
        phonemes, _ = parse_textgrid(basic_tg)
        assert isinstance(phonemes, list)

    def test_returns_word_list(self, basic_tg):
        _, words = parse_textgrid(basic_tg)
        assert isinstance(words, list)

    def test_phoneme_objects(self, basic_tg):
        phonemes, _ = parse_textgrid(basic_tg)
        for p in phonemes:
            assert isinstance(p, ReferencePhoneme)

    def test_word_objects(self, basic_tg):
        _, words = parse_textgrid(basic_tg)
        for w in words:
            assert isinstance(w, ReferenceWord)


# ---------------------------------------------------------------------------
# Silence filtering
# ---------------------------------------------------------------------------

class TestSilenceFiltering:
    def test_silence_excluded_by_default(self, silence_tg):
        phonemes, _ = parse_textgrid(silence_tg)
        labels = [p.phoneme for p in phonemes]
        assert "SIL" not in labels
        assert "" not in labels

    def test_silence_preserved_when_disabled(self, silence_tg):
        phonemes, _ = parse_textgrid(silence_tg, skip_silence=False)
        labels = [p.phoneme for p in phonemes]
        assert "" in labels or "SIL" in labels

    def test_sil_excluded(self, basic_tg):
        phonemes, _ = parse_textgrid(basic_tg)
        assert all(p.phoneme != "SIL" for p in phonemes)

    def test_non_silence_included(self, silence_tg):
        phonemes, _ = parse_textgrid(silence_tg)
        assert any(p.phoneme == "AH" for p in phonemes)


# ---------------------------------------------------------------------------
# Timestamp correctness
# ---------------------------------------------------------------------------

class TestTimestamps:
    def test_start_times_positive(self, basic_tg):
        phonemes, _ = parse_textgrid(basic_tg)
        assert all(p.start_time >= 0 for p in phonemes)

    def test_end_after_start(self, basic_tg):
        phonemes, _ = parse_textgrid(basic_tg)
        for p in phonemes:
            assert p.end_time > p.start_time

    def test_positive_duration(self, basic_tg):
        phonemes, _ = parse_textgrid(basic_tg)
        for p in phonemes:
            assert p.duration > 0

    def test_first_phoneme_timestamps(self, basic_tg):
        phonemes, _ = parse_textgrid(basic_tg)
        assert phonemes[0].start_time == pytest.approx(0.0)
        assert phonemes[0].end_time == pytest.approx(0.3)

    def test_word_timestamps(self, basic_tg):
        _, words = parse_textgrid(basic_tg)
        assert words[0].start_time == pytest.approx(0.0)
        assert words[0].end_time == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Back-annotation
# ---------------------------------------------------------------------------

class TestBackAnnotation:
    def test_phoneme_idx_assigned(self, basic_tg):
        phonemes, _ = parse_textgrid(basic_tg)
        for i, p in enumerate(phonemes):
            assert p.phoneme_idx == i

    def test_word_idx_assigned(self, basic_tg):
        _, words = parse_textgrid(basic_tg)
        for i, w in enumerate(words):
            assert w.word_idx == i

    def test_phoneme_word_idx_annotated(self, basic_tg):
        phonemes, words = parse_textgrid(basic_tg)
        # Phonemes in the first word's span should have word_idx=0
        first_word = words[0]
        for p in phonemes:
            if p.start_time >= first_word.start_time and p.end_time <= first_word.end_time + 1e-6:
                assert p.word_idx == 0

    def test_word_phoneme_indices_populated(self, basic_tg):
        phonemes, words = parse_textgrid(basic_tg)
        for w in words:
            # Each word should reference at least one phoneme
            assert len(w.phoneme_indices) >= 1


# ---------------------------------------------------------------------------
# Missing tiers / robustness
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_missing_word_tier_returns_empty_words(self, empty_tier_tg):
        phonemes, words = parse_textgrid(
            empty_tier_tg, phoneme_tier="phonemes", word_tier="words"
        )
        assert words == []

    def test_missing_phoneme_tier_returns_empty_phonemes(self, empty_tier_tg):
        phonemes, words = parse_textgrid(
            empty_tier_tg, phoneme_tier="nonexistent"
        )
        assert phonemes == []

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_textgrid(tmp_path / "missing.TextGrid")

    def test_custom_tier_names(self, tmp_path):
        tg = _BASIC_TG.replace('name = "phonemes"', 'name = "phones"')
        f = tmp_path / "custom.TextGrid"
        f.write_text(tg, encoding="utf-8")
        phonemes, _ = parse_textgrid(f, phoneme_tier="phones")
        # Should find the renamed tier
        assert len(phonemes) > 0
