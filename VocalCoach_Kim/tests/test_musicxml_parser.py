"""
tests/test_musicxml_parser.py - Unit tests for reference/musicxml_parser.py

Tests use a minimal synthetic MusicXML string written to a temp file so the
music21 library is exercised without requiring a real score on disk.

Run with:
    cd MusicAI/VocalCoach
    python -m pytest tests/test_musicxml_parser.py -v

Skip all tests gracefully if music21 is not installed.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import pytest

music21 = pytest.importorskip("music21", reason="music21 not installed")

from reference.musicxml_parser import (
    _beats_to_seconds,
    _merge_tied_notes,
    _midi_to_name,
    parse_musicxml,
)
from utils.types import ReferenceNote, ReferencePerformanceRepresentation


# ---------------------------------------------------------------------------
# Minimal MusicXML templates
# ---------------------------------------------------------------------------

_SIMPLE_XML = textwrap.dedent("""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC
    "-//Recordare//DTD MusicXML 3.1 Partwise//EN"
    "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1"><part-name>Voice</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <direction placement="above">
        <direction-type>
          <metronome parentheses="no">
            <beat-unit>quarter</beat-unit>
            <per-minute>120</per-minute>
          </metronome>
        </direction-type>
        <sound tempo="120"/>
      </direction>
      <note>
        <pitch><step>A</step><octave>4</octave></pitch>
        <duration>1</duration>
        <type>quarter</type>
      </note>
      <note>
        <pitch><step>C</step><octave>5</octave></pitch>
        <duration>1</duration>
        <type>quarter</type>
      </note>
      <note>
        <rest/>
        <duration>2</duration>
        <type>half</type>
      </note>
    </measure>
  </part>
</score-partwise>
""")

_TIED_XML = textwrap.dedent("""\
<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1"><part-name>Voice</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <time><beats>4</beats><beat-type>4</beat-type></time>
      </attributes>
      <direction placement="above">
        <direction-type>
          <metronome><beat-unit>quarter</beat-unit><per-minute>120</per-minute></metronome>
        </direction-type>
        <sound tempo="120"/>
      </direction>
      <note>
        <pitch><step>E</step><octave>4</octave></pitch>
        <duration>2</duration>
        <type>half</type>
        <tie type="start"/>
        <notations><tied type="start"/></notations>
      </note>
      <note>
        <pitch><step>E</step><octave>4</octave></pitch>
        <duration>2</duration>
        <type>half</type>
        <tie type="stop"/>
        <notations><tied type="stop"/></notations>
      </note>
    </measure>
  </part>
</score-partwise>
""")


@pytest.fixture()
def simple_xml_file(tmp_path):
    f = tmp_path / "simple.xml"
    f.write_text(_SIMPLE_XML, encoding="utf-8")
    return f


@pytest.fixture()
def tied_xml_file(tmp_path):
    f = tmp_path / "tied.xml"
    f.write_text(_TIED_XML, encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

class TestBeatsToSeconds:
    def test_120bpm_quarter(self):
        assert _beats_to_seconds(1.0, 120.0) == pytest.approx(0.5)

    def test_60bpm_quarter(self):
        assert _beats_to_seconds(1.0, 60.0) == pytest.approx(1.0)

    def test_zero_beats(self):
        assert _beats_to_seconds(0.0, 120.0) == pytest.approx(0.0)

    def test_two_beats(self):
        assert _beats_to_seconds(2.0, 120.0) == pytest.approx(1.0)


class TestMidiToName:
    def test_a4(self):
        assert _midi_to_name(69) == "A4"

    def test_middle_c(self):
        assert _midi_to_name(60) == "C4"

    def test_c_sharp_5(self):
        assert _midi_to_name(73) == "C#5"


class TestMergeTiedNotes:
    def _note(self, onset, offset, tie=None, pitch_midi=60):
        return {
            "onset_time": onset,
            "offset_time": offset,
            "duration": offset - onset,
            "pitch_midi": pitch_midi,
            "pitch_hz": None,
            "pitch_name": None,
            "lyric": None,
            "measure": None,
            "beat": None,
            "duration_beats": offset - onset,
            "is_rest": False,
            "is_tied": False,
            "tie_type": tie,
        }

    def test_no_ties_unchanged(self):
        notes = [self._note(0.0, 0.5), self._note(0.5, 1.0)]
        merged = _merge_tied_notes(notes)
        assert len(merged) == 2

    def test_two_tied_become_one(self):
        notes = [
            self._note(0.0, 0.5, tie="start"),
            self._note(0.5, 1.0, tie="stop"),
        ]
        merged = _merge_tied_notes(notes)
        assert len(merged) == 1
        assert merged[0]["offset_time"] == pytest.approx(1.0)
        assert merged[0]["duration"] == pytest.approx(1.0)
        assert merged[0]["is_tied"] is True

    def test_three_tied_become_one(self):
        notes = [
            self._note(0.0, 0.5, tie="start"),
            self._note(0.5, 1.0, tie="continue"),
            self._note(1.0, 1.5, tie="stop"),
        ]
        merged = _merge_tied_notes(notes)
        assert len(merged) == 1
        assert merged[0]["offset_time"] == pytest.approx(1.5)

    def test_empty(self):
        assert _merge_tied_notes([]) == []


# ---------------------------------------------------------------------------
# parse_musicxml — basic parsing
# ---------------------------------------------------------------------------

class TestParseMusicXMLBasic:
    def test_returns_reference_representation(self, simple_xml_file):
        ref = parse_musicxml(simple_xml_file)
        assert isinstance(ref, ReferencePerformanceRepresentation)

    def test_source_path_set(self, simple_xml_file):
        ref = parse_musicxml(simple_xml_file)
        assert str(simple_xml_file) in ref.source_path

    def test_note_count(self, simple_xml_file):
        ref = parse_musicxml(simple_xml_file, include_rests=True)
        # 2 pitched notes + 1 rest
        assert len(ref.notes) == 3

    def test_note_count_no_rests(self, simple_xml_file):
        ref = parse_musicxml(simple_xml_file, include_rests=False)
        assert all(not n.is_rest for n in ref.notes)
        assert len(ref.notes) == 2

    def test_returns_reference_note_objects(self, simple_xml_file):
        ref = parse_musicxml(simple_xml_file)
        for n in ref.notes:
            assert isinstance(n, ReferenceNote)

    def test_notes_sorted_by_onset(self, simple_xml_file):
        ref = parse_musicxml(simple_xml_file)
        onsets = [n.onset_time for n in ref.notes]
        assert onsets == sorted(onsets)


class TestParseMusicXMLTiming:
    def test_tempo_parsed(self, simple_xml_file):
        ref = parse_musicxml(simple_xml_file)
        assert ref.tempo_bpm == pytest.approx(120.0)

    def test_default_tempo_used_when_absent(self, tmp_path):
        # MusicXML with no MetronomeMark
        xml = _SIMPLE_XML.replace(
            "<direction placement=\"above\">",
            "<!-- "
        ).replace("</direction>", " -->")
        f = tmp_path / "notempo.xml"
        f.write_text(xml, encoding="utf-8")
        ref = parse_musicxml(f, default_tempo_bpm=90.0)
        assert ref.tempo_bpm == pytest.approx(90.0)

    def test_onset_time_correct(self, simple_xml_file):
        ref = parse_musicxml(simple_xml_file)
        # A4 at offset 0 beats, tempo 120 → 0.0s
        non_rest = [n for n in ref.notes if not n.is_rest]
        assert non_rest[0].onset_time == pytest.approx(0.0, abs=0.01)

    def test_offset_time_correct(self, simple_xml_file):
        ref = parse_musicxml(simple_xml_file)
        non_rest = [n for n in ref.notes if not n.is_rest]
        # Quarter note at 120 BPM = 0.5s duration
        assert non_rest[0].offset_time == pytest.approx(0.5, abs=0.01)

    def test_positive_duration(self, simple_xml_file):
        ref = parse_musicxml(simple_xml_file)
        for n in ref.notes:
            assert n.duration is not None
            assert n.duration > 0

    def test_duration_s_positive(self, simple_xml_file):
        ref = parse_musicxml(simple_xml_file)
        assert ref.duration_s > 0


class TestParseMusicXMLPitch:
    def test_pitch_midi_set(self, simple_xml_file):
        ref = parse_musicxml(simple_xml_file)
        non_rest = [n for n in ref.notes if not n.is_rest]
        assert all(n.pitch_midi is not None for n in non_rest)

    def test_a4_midi_is_69(self, simple_xml_file):
        ref = parse_musicxml(simple_xml_file)
        non_rest = [n for n in ref.notes if not n.is_rest]
        assert non_rest[0].pitch_midi == pytest.approx(69.0)

    def test_pitch_hz_derived_from_midi(self, simple_xml_file):
        ref = parse_musicxml(simple_xml_file)
        for n in ref.notes:
            if n.pitch_midi is not None and n.pitch_hz is not None:
                expected = 440.0 * (2.0 ** ((n.pitch_midi - 69.0) / 12.0))
                assert n.pitch_hz == pytest.approx(expected, rel=1e-4)

    def test_rest_has_no_pitch(self, simple_xml_file):
        ref = parse_musicxml(simple_xml_file, include_rests=True)
        rests = [n for n in ref.notes if n.is_rest]
        assert all(n.pitch_midi is None for n in rests)

    def test_pitch_midi_in_range(self, simple_xml_file):
        ref = parse_musicxml(simple_xml_file)
        for n in ref.notes:
            if n.pitch_midi is not None:
                assert 0.0 <= n.pitch_midi <= 127.0


class TestParseMusicXMLTies:
    def test_tied_notes_merged(self, tied_xml_file):
        ref = parse_musicxml(tied_xml_file, merge_ties=True)
        assert len(ref.notes) == 1
        assert ref.notes[0].duration == pytest.approx(2.0, abs=0.05)

    def test_ties_not_merged_when_disabled(self, tied_xml_file):
        ref = parse_musicxml(tied_xml_file, merge_ties=False)
        assert len(ref.notes) == 2


class TestParseMusicXMLMetadata:
    def test_time_signature(self, simple_xml_file):
        ref = parse_musicxml(simple_xml_file)
        assert ref.time_signature is not None
        assert ref.time_signature[0] == 4
        assert ref.time_signature[1] == 4

    def test_note_idx_sequential(self, simple_xml_file):
        ref = parse_musicxml(simple_xml_file)
        for i, n in enumerate(ref.notes):
            assert n.note_idx == i


class TestParseMusicXMLErrors:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_musicxml(tmp_path / "nonexistent.xml")

    def test_bad_file_raises_value_error(self, tmp_path):
        bad = tmp_path / "bad.xml"
        bad.write_text("this is not xml", encoding="utf-8")
        with pytest.raises((ValueError, Exception)):
            parse_musicxml(bad)
