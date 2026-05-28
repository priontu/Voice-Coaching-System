"""
utils/types.py - Shared data structures for all VocalCoach modules.

All dataclasses use type hints and are JSON-serializable via asdict() or
their to_dict() methods. They act as contracts between preprocessing, model
inference, fusion, metrics, and scoring layers — enabling integration without
tight coupling.

Phase history:
  Phase 1: AudioFeatures, PhonemeSegment, PitchFrame, PitchResult,
           NoteEvent, InferenceResult
  Phase 2: MelSpectrogramFeatures, FrameAlignedFeatures,
           TimestampedFeatureSequence
  Phase 3: UnifiedInferenceResult
  Phase 4: LyricEvent, WordEvent, PhraseEvent, TemporalRegion,
           FusedPerformanceRepresentation; NoteEvent expanded with
           pitch metadata and phoneme linkage
  Phase 5: ReferenceNote, ReferencePhoneme, ReferenceWord, ReferencePhrase,
           ReferencePerformanceRepresentation, AlignmentResult,
           NoteAlignmentMatch, PhonemeAlignmentMatch, WordAlignmentMatch
  Phase 6: MetricBreakdown, PitchMetrics, TimingMetrics, DurationMetrics,
           LyricMetrics, PerformanceMetricsReport
  Phase 7: ScoreBreakdown, CategoryScore, PerformanceScoreReport,
           InterpretationSummary
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Audio features container
# ---------------------------------------------------------------------------

@dataclass
class AudioFeatures:
    """Preprocessed audio ready for model inference."""

    audio: object          # np.ndarray or torch.Tensor, shape (samples,)
    sample_rate: int
    duration_s: float
    source_path: str = ""

    def to_dict(self) -> Dict:
        return {
            "sample_rate": self.sample_rate,
            "duration_s": self.duration_s,
            "source_path": self.source_path,
        }


# ---------------------------------------------------------------------------
# Phoneme structures
# ---------------------------------------------------------------------------

@dataclass
class PhonemeSegment:
    """
    Timed phoneme boundary with confidence.

    Compatible with the PhonemeSegment defined in the original
    phoneme_model.py — field names and types are identical so existing
    code can import from either location.
    """

    phoneme: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    frame_start: int = 0
    frame_end: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


# ---------------------------------------------------------------------------
# Pitch / VAD structures
# ---------------------------------------------------------------------------

@dataclass
class PitchFrame:
    """
    Single pitch analysis frame.

    Compatible with the JSON schema produced by pitch_wrapper.py /
    inference.py (save_pitch_json) and consumed by pitch_score.py.
    """

    time: float
    f0: float
    voiced: bool
    midi: Optional[float] = None
    confidence: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PitchResult:
    """Full output of the VAD+pitch pipeline for one audio file."""

    frames: List[PitchFrame] = field(default_factory=list)
    audio_path: str = ""
    sample_rate: int = 16000
    hop_length: int = 160

    def to_dict(self) -> Dict:
        return {
            "audio_path": self.audio_path,
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "num_frames": len(self.frames),
            "frames": [f.to_dict() for f in self.frames],
        }


# ---------------------------------------------------------------------------
# Note / onset-offset structures
# ---------------------------------------------------------------------------

@dataclass
class NoteEvent:
    """
    A detected (or reference) note with onset and offset times.

    The onset/offset field names match the JSON schema used by NoteDetector.
    Phase 4 adds optional pitch metadata, confidence scores, and phoneme
    linkage. All Phase-4 fields default to None for full backward compatibility.

    Pitch stats (pitch_hz, pitch_midi, pitch_stability, voiced_fraction) are
    populated by fusion/note_events.build_note_events().

    Phoneme linkage (phoneme_labels, lyric_text) is back-annotated by
    fusion/event_alignment.annotate_notes_with_phonemes().
    """

    onset_time: float
    offset_time: Optional[float] = None
    duration: Optional[float] = None

    # Pitch metadata (Phase 4)
    pitch_hz: Optional[float] = None          # mean voiced F0 within note (Hz)
    pitch_midi: Optional[float] = None        # MIDI note number (0–127)
    pitch_stability: Optional[float] = None   # std of voiced F0; lower = more stable

    # Voice quality (Phase 4)
    voiced_fraction: Optional[float] = None   # fraction of note frames that are voiced

    # Confidence (Phase 4)
    confidence: Optional[float] = None        # combined onset+offset peak confidence
    onset_confidence: Optional[float] = None  # onset peak probability
    offset_confidence: Optional[float] = None # offset peak probability

    # Phoneme linkage (Phase 4; back-annotated by event_alignment.py)
    phoneme_labels: Optional[List[str]] = None  # unique phonemes active during note
    lyric_text: Optional[str] = None            # human-readable phoneme sequence

    # Identity (Phase 4)
    note_idx: Optional[int] = None             # position in the note list

    def __post_init__(self) -> None:
        if self.offset_time is not None and self.duration is None:
            self.duration = round(self.offset_time - self.onset_time, 6)

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Phase 4 — Lyric and structural event types
# ---------------------------------------------------------------------------

@dataclass
class LyricEvent:
    """
    One phoneme segment placed in its lyrical context.

    Created 1:1 from PhonemeSegment objects by fusion/lyric_events.py.
    word_idx and note_idx are back-annotated by event_alignment.py after
    the note and word lists are constructed.
    """

    phoneme: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    lyric_idx: int = 0
    word_idx: Optional[int] = None   # index into the parent word_events list
    note_idx: Optional[int] = None   # index into the note_events list

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class WordEvent:
    """
    A word or syllable group assembled from consecutive LyricEvents.

    Constructed by fusion/lyric_events.py via proximity-based phoneme
    grouping. note_idx is back-annotated by event_alignment.py to indicate
    which note this word predominantly falls within.
    """

    text: str                                             # concatenated phoneme labels
    start_time: float
    end_time: float
    phoneme_events: List[LyricEvent] = field(default_factory=list)
    confidence: float = 1.0
    word_idx: int = 0
    note_idx: Optional[int] = None                        # dominant note index

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def n_phonemes(self) -> int:
        return len(self.phoneme_events)

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "confidence": self.confidence,
            "word_idx": self.word_idx,
            "note_idx": self.note_idx,
            "n_phonemes": self.n_phonemes,
            "phoneme_events": [p.to_dict() for p in self.phoneme_events],
        }


@dataclass
class PhraseEvent:
    """
    A musical phrase spanning multiple consecutive notes.

    Constructed by fusion/event_alignment.build_phrase_events() by grouping
    notes whose inter-note gap is below a configurable threshold.
    word_indices is optionally populated when word events are available.
    """

    start_time: float
    end_time: float
    note_indices: List[int] = field(default_factory=list)
    word_indices: List[int] = field(default_factory=list)
    confidence: float = 1.0
    phrase_idx: int = 0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def n_notes(self) -> int:
        return len(self.note_indices)

    def to_dict(self) -> Dict:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "note_indices": self.note_indices,
            "word_indices": self.word_indices,
            "confidence": self.confidence,
            "phrase_idx": self.phrase_idx,
        }


@dataclass
class TemporalRegion:
    """
    Generic labeled time interval.

    Used for voiced/unvoiced/silence annotations produced by
    fusion/event_alignment.build_voiced_regions(). The label field is one of
    'voiced', 'unvoiced', or a domain-specific string.
    """

    label: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Generic inference result
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """
    Generic container for any model's inference output.

    Allows scoring and fusion layers to accept results from any module
    without knowing module internals.
    """

    model_name: str
    audio_path: str
    metadata: Dict = field(default_factory=dict)
    # Module-specific payloads — populated by each model's runner
    phoneme_segments: Optional[List[PhonemeSegment]] = None
    pitch_frames: Optional[List[PitchFrame]] = None
    note_events: Optional[List[NoteEvent]] = None

    def to_dict(self) -> Dict:
        out: Dict = {
            "model_name": self.model_name,
            "audio_path": self.audio_path,
            "metadata": self.metadata,
        }
        if self.phoneme_segments is not None:
            out["phoneme_segments"] = [s.to_dict() for s in self.phoneme_segments]
        if self.pitch_frames is not None:
            out["pitch_frames"] = [f.to_dict() for f in self.pitch_frames]
        if self.note_events is not None:
            out["note_events"] = [n.to_dict() for n in self.note_events]
        return out


# ---------------------------------------------------------------------------
# Phase 2 — frame-aligned feature containers
# ---------------------------------------------------------------------------

@dataclass
class MelSpectrogramFeatures:
    """Log-mel spectrogram with its associated timestamp array."""

    spectrogram: Any    # np.ndarray, shape (n_mels, n_frames)
    timestamps: Any     # np.ndarray, shape (n_frames,), seconds
    hop_length: int = 160
    sample_rate: int = 16000
    n_mels: int = 80

    @property
    def n_frames(self) -> int:
        return self.spectrogram.shape[1] if hasattr(self.spectrogram, "shape") else 0

    def to_dict(self) -> Dict:
        return {
            "hop_length": self.hop_length,
            "sample_rate": self.sample_rate,
            "n_mels": self.n_mels,
            "n_frames": self.n_frames,
        }


@dataclass
class FrameAlignedFeatures:
    """
    Multi-stream feature tensor aligned to ONE canonical timestamp grid.

    All arrays have the same first dimension (n_frames) so they can be
    stacked or zipped frame-by-frame without any additional alignment.

    The canonical timeline uses HOP_LENGTH=160 at 16 kHz (10 ms frames).
    Per-model outputs at other frame rates are resampled via
    fusion/alignment.py before being stored here.
    """

    timestamps: Any                        # np.ndarray (n_frames,) seconds
    f0: Optional[Any] = None              # np.ndarray (n_frames,) Hz; 0 = unvoiced
    voiced: Optional[Any] = None          # np.ndarray bool (n_frames,)
    onset_probs: Optional[Any] = None     # np.ndarray (n_frames,) ∈ [0, 1]
    offset_probs: Optional[Any] = None    # np.ndarray (n_frames,) ∈ [0, 1]
    phoneme_labels: Optional[List[str]] = None  # length n_frames
    hop_length: int = 160
    sample_rate: int = 16000

    @property
    def n_frames(self) -> int:
        arr = self.timestamps
        return len(arr) if arr is not None else 0

    def to_dict(self) -> Dict:
        out: Dict = {
            "hop_length": self.hop_length,
            "sample_rate": self.sample_rate,
            "n_frames": self.n_frames,
        }
        if self.phoneme_labels is not None:
            out["phoneme_labels"] = self.phoneme_labels
        return out


@dataclass
class TimestampedFeatureSequence:
    """
    Generic container for any named feature aligned to a timestamp axis.

    Useful for passing single-stream outputs between processing stages
    without committing to a specific feature set.
    """

    timestamps: Any    # np.ndarray (N,) seconds
    values: Any        # np.ndarray (N,) or (N, D)
    feature_name: str = ""
    hop_length: int = 160
    sample_rate: int = 16000

    @property
    def n_frames(self) -> int:
        return len(self.timestamps) if self.timestamps is not None else 0

    def to_dict(self) -> Dict:
        return {
            "feature_name": self.feature_name,
            "hop_length": self.hop_length,
            "sample_rate": self.sample_rate,
            "n_frames": self.n_frames,
        }


# ---------------------------------------------------------------------------
# Phase 4 — Fused performance representation (canonical downstream object)
# ---------------------------------------------------------------------------

@dataclass
class FusedPerformanceRepresentation:
    """
    Canonical downstream representation combining all fusion outputs.

    Produced by inference/pipeline.py when fusion.enabled=true.
    Consumed by the scoring and feedback layers (Phase 5+).

    Invariants maintained by fusion/validation.py:
      - note_events sorted by onset_time
      - lyric_events sorted by start_time
      - word_events sorted by start_time
      - phrase_events sorted by start_time
      - voiced_regions sorted by start_time, non-overlapping
      - all event start/end times within [0, duration_s]
      - timestamps, f0, voiced share the same length (n_frames)
    """

    audio_path: str
    duration_s: float
    sample_rate: int = 16000
    hop_length: int = 160

    # Canonical-grid arrays (all length n_frames = 100fps)
    timestamps: Optional[Any] = None   # np.ndarray (n_frames,) seconds
    f0: Optional[Any] = None           # np.ndarray (n_frames,) Hz; 0 = unvoiced
    voiced: Optional[Any] = None       # np.ndarray bool (n_frames,)

    # Structured events
    note_events: List[NoteEvent] = field(default_factory=list)
    lyric_events: List[LyricEvent] = field(default_factory=list)
    word_events: List[WordEvent] = field(default_factory=list)
    phrase_events: List[PhraseEvent] = field(default_factory=list)
    voiced_regions: List[TemporalRegion] = field(default_factory=list)

    # Raw phoneme segments (preserved for downstream use)
    phoneme_segments: Optional[List[PhonemeSegment]] = None

    # Metadata
    alignment_metadata: Dict = field(default_factory=dict)
    inference_metadata: Dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.timestamps) if self.timestamps is not None else 0

    @property
    def n_notes(self) -> int:
        return len(self.note_events)

    def to_dict(self) -> Dict:
        out: Dict = {
            "audio_path": self.audio_path,
            "duration_s": self.duration_s,
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "n_frames": self.n_frames,
            "n_notes": self.n_notes,
            "n_words": len(self.word_events),
            "n_phrases": len(self.phrase_events),
            "n_voiced_regions": len(self.voiced_regions),
            "alignment_metadata": self.alignment_metadata,
            "inference_metadata": self.inference_metadata,
        }
        if self.note_events:
            out["note_events"] = [n.to_dict() for n in self.note_events]
        if self.lyric_events:
            out["lyric_events"] = [le.to_dict() for le in self.lyric_events]
        if self.word_events:
            out["word_events"] = [w.to_dict() for w in self.word_events]
        if self.phrase_events:
            out["phrase_events"] = [p.to_dict() for p in self.phrase_events]
        if self.voiced_regions:
            out["voiced_regions"] = [r.to_dict() for r in self.voiced_regions]
        if self.phoneme_segments is not None:
            out["phoneme_segments"] = [s.to_dict() for s in self.phoneme_segments]
        return out


# ---------------------------------------------------------------------------
# Phase 3 — unified pipeline result
# ---------------------------------------------------------------------------

@dataclass
class UnifiedInferenceResult:
    """
    Aggregated output of the unified inference pipeline.

    Contains raw per-model outputs (preserved at each model's native frame
    rate) and a temporally aligned view on the canonical 10-ms grid.

    Fields marked Optional are None when the corresponding module is disabled
    via the pipeline config (enable_phoneme, enable_pitch, enable_onset_offset).

    Phase 4 adds the optional fused field, populated when fusion.enabled=true
    in the pipeline config.
    """

    # Core metadata
    audio_path: str
    sample_rate: int = 16000
    hop_length: int = 160          # canonical hop
    duration_s: float = 0.0

    # Raw phoneme outputs (native: time-continuous segments, ~50fps encoder)
    phoneme_segments: Optional[List[PhonemeSegment]] = None

    # Raw pitch outputs (native: 100fps, hop=160)
    pitch_timestamps: Optional[Any] = None   # np.ndarray (N_p,)
    f0: Optional[Any] = None                 # np.ndarray (N_p,) Hz
    voiced: Optional[Any] = None             # np.ndarray bool (N_p,)

    # Raw onset/offset outputs (native: ~62.5fps, hop=256)
    onset_timestamps: Optional[Any] = None   # np.ndarray (N_o,)
    onset_probs: Optional[Any] = None        # np.ndarray (N_o,) ∈ [0,1]
    offset_probs: Optional[Any] = None       # np.ndarray (N_o,) ∈ [0,1]

    # Detected note boundaries (from peak-picking on onset/offset probs)
    note_events: Optional[List[NoteEvent]] = None

    # Temporally aligned view on the canonical 100fps grid
    aligned: Optional[FrameAlignedFeatures] = None

    # Phase 4: fused performance representation (populated when fusion enabled)
    fused: Optional[FusedPerformanceRepresentation] = None

    # Phase 5: parsed reference + alignment result
    reference: Optional[ReferencePerformanceRepresentation] = None
    alignment: Optional[AlignmentResult] = None

    # Phase 6: computed metrics
    metrics: Optional[PerformanceMetricsReport] = None

    # Phase 7: scoring and interpretation
    scores: Optional[PerformanceScoreReport] = None
    interpretation: Optional[InterpretationSummary] = None

    # Pipeline metadata (timing, device, enabled modules, etc.)
    metadata: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------

    def has_pitch(self) -> bool:
        return self.f0 is not None

    def has_phonemes(self) -> bool:
        return self.phoneme_segments is not None

    def has_onset_offset(self) -> bool:
        return self.onset_probs is not None

    def has_fused(self) -> bool:
        return self.fused is not None

    def is_complete(self) -> bool:
        """True if all three model streams are present."""
        return self.has_pitch() and self.has_phonemes() and self.has_onset_offset()

    def to_dict(self) -> Dict:
        out: Dict = {
            "audio_path": self.audio_path,
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "duration_s": self.duration_s,
            "metadata": self.metadata,
        }
        if self.phoneme_segments is not None:
            out["phoneme_segments"] = [s.to_dict() for s in self.phoneme_segments]
        if self.note_events is not None:
            out["note_events"] = [n.to_dict() for n in self.note_events]
        if self.f0 is not None:
            out["n_pitch_frames"] = len(self.f0)
        if self.onset_probs is not None:
            out["n_onset_frames"] = len(self.onset_probs)
        if self.aligned is not None:
            out["n_canonical_frames"] = self.aligned.n_frames
        if self.fused is not None:
            out["fused"] = self.fused.to_dict()
        if self.reference is not None:
            out["reference"] = self.reference.to_dict()
        if self.alignment is not None:
            out["alignment"] = self.alignment.to_dict()
        if self.metrics is not None:
            out["metrics"] = self.metrics.to_dict()
        if self.scores is not None:
            out["scores"] = self.scores.to_dict()
        if self.interpretation is not None:
            out["interpretation"] = self.interpretation.to_dict()
        return out


# ---------------------------------------------------------------------------
# Phase 5 — Ground-truth / reference data structures
# ---------------------------------------------------------------------------

@dataclass
class ReferenceNote:
    """
    A single note from a reference score (MusicXML) or ground-truth annotation.

    Timestamps are always in seconds (converted from beats/measures using tempo
    by the parser). All pitch metadata is Optional; rests have pitch_midi=None.
    Tied notes are represented as a single merged note with the combined duration.
    """

    onset_time: float
    offset_time: float
    duration: Optional[float] = None      # offset_time - onset_time (computed in __post_init__)

    # Pitch
    pitch_midi: Optional[float] = None    # MIDI note number (0–127); None for rests
    pitch_hz: Optional[float] = None      # Hz; derived from MIDI via 440·2^((midi-69)/12)
    pitch_name: Optional[str] = None      # e.g. "A4", "C#5"

    # Lyric / text
    lyric: Optional[str] = None           # syllable/word attached to this note in the score

    # Score position
    measure: Optional[int] = None         # 1-indexed measure number
    beat: Optional[float] = None          # beat within measure (1.0 = first beat)
    duration_beats: Optional[float] = None  # note duration in quarter-note beats

    # Flags
    is_rest: bool = False
    is_tied: bool = False                 # True when this is a continuation of a tied note

    # Identity
    note_idx: Optional[int] = None

    def __post_init__(self) -> None:
        if self.duration is None:
            self.duration = round(self.offset_time - self.onset_time, 6)
        if self.pitch_midi is not None and self.pitch_hz is None:
            import math
            self.pitch_hz = round(440.0 * (2.0 ** ((self.pitch_midi - 69.0) / 12.0)), 4)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ReferencePhoneme:
    """
    A phoneme annotation entry from a Praat TextGrid file.

    Timing is in seconds (TextGrid stores absolute timestamps).
    word_idx links back to the parent ReferenceWord after parsing.
    """

    phoneme: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    word_idx: Optional[int] = None
    phoneme_idx: Optional[int] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ReferenceWord:
    """
    A word annotation entry from a Praat TextGrid file.

    phoneme_indices lists the indices into the ReferencePhoneme list that
    fall within this word's time span. Back-annotated by the parser.
    """

    text: str
    start_time: float
    end_time: float
    phoneme_indices: List[int] = field(default_factory=list)
    confidence: float = 1.0
    word_idx: Optional[int] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ReferencePhrase:
    """
    A musical phrase inferred from the reference score.

    Constructed by grouping consecutive reference notes whose inter-note
    gap is below a configurable threshold, mirroring the prediction-side
    PhraseEvent construction in fusion/event_alignment.py.
    """

    start_time: float
    end_time: float
    note_indices: List[int] = field(default_factory=list)
    word_indices: List[int] = field(default_factory=list)
    phrase_idx: int = 0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def n_notes(self) -> int:
        return len(self.note_indices)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ReferencePerformanceRepresentation:
    """
    Canonical reference object combining MusicXML score data and TextGrid
    phoneme/word annotations.

    Produced by reference/reference_builder.py.
    Consumed by alignment/reference_alignment.py to compare against
    FusedPerformanceRepresentation.

    Invariants maintained by reference/validation.py:
      - notes sorted by onset_time
      - phonemes sorted by start_time
      - words sorted by start_time
      - phrases sorted by start_time
      - no negative timestamps
      - tied notes merged (single contiguous note event)
    """

    source_path: str                       # MusicXML path (or TextGrid if no XML)
    duration_s: float

    # Score-level metadata
    tempo_bpm: Optional[float] = None
    time_signature: Optional[Tuple[int, int]] = None   # (numerator, denominator)
    key_signature: Optional[str] = None                # e.g. "C major", "A minor"

    # Structured events
    notes: List[ReferenceNote] = field(default_factory=list)
    phonemes: List[ReferencePhoneme] = field(default_factory=list)
    words: List[ReferenceWord] = field(default_factory=list)
    phrases: List[ReferencePhrase] = field(default_factory=list)

    # Parser metadata
    metadata: Dict = field(default_factory=dict)

    @property
    def n_notes(self) -> int:
        return len(self.notes)

    @property
    def n_phonemes(self) -> int:
        return len(self.phonemes)

    def to_dict(self) -> Dict:
        out: Dict = {
            "source_path": self.source_path,
            "duration_s": self.duration_s,
            "tempo_bpm": self.tempo_bpm,
            "time_signature": list(self.time_signature) if self.time_signature else None,
            "key_signature": self.key_signature,
            "n_notes": self.n_notes,
            "n_phonemes": self.n_phonemes,
            "n_words": len(self.words),
            "n_phrases": len(self.phrases),
            "metadata": self.metadata,
        }
        if self.notes:
            out["notes"] = [n.to_dict() for n in self.notes]
        if self.phonemes:
            out["phonemes"] = [p.to_dict() for p in self.phonemes]
        if self.words:
            out["words"] = [w.to_dict() for w in self.words]
        if self.phrases:
            out["phrases"] = [p.to_dict() for p in self.phrases]
        return out


# ---------------------------------------------------------------------------
# Phase 5 — Alignment match records
# ---------------------------------------------------------------------------

@dataclass
class NoteAlignmentMatch:
    """One predicted-note ↔ reference-note correspondence."""

    pred_idx: int
    ref_idx: int
    overlap_s: float                    # temporal overlap in seconds
    overlap_fraction: float             # overlap / duration of the shorter note
    onset_deviation_s: float            # predicted_onset - reference_onset (positive = late)
    offset_deviation_s: float           # predicted_offset - reference_offset
    pitch_deviation_cents: Optional[float] = None   # positive = sharp

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PhonemeAlignmentMatch:
    """One predicted LyricEvent ↔ reference ReferencePhoneme correspondence."""

    pred_idx: int
    ref_idx: int
    overlap_s: float
    overlap_fraction: float
    onset_deviation_s: float
    label_match: bool = False           # True if phoneme strings match exactly

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class WordAlignmentMatch:
    """One predicted WordEvent ↔ reference ReferenceWord correspondence."""

    pred_idx: int
    ref_idx: int
    overlap_s: float
    overlap_fraction: float
    onset_deviation_s: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AlignmentResult:
    """
    Full prediction ↔ reference alignment result.

    Produced by alignment/reference_alignment.align_performance().
    Consumed by the scoring layer (Phase 6+) and visualization.

    unmatched_pred_* and unmatched_ref_* contain indices of events with
    no correspondence found at the configured alignment thresholds.
    """

    predicted_audio_path: str
    reference_source_path: str

    note_matches: List[NoteAlignmentMatch] = field(default_factory=list)
    phoneme_matches: List[PhonemeAlignmentMatch] = field(default_factory=list)
    word_matches: List[WordAlignmentMatch] = field(default_factory=list)

    unmatched_pred_notes: List[int] = field(default_factory=list)
    unmatched_ref_notes: List[int] = field(default_factory=list)
    unmatched_pred_phonemes: List[int] = field(default_factory=list)
    unmatched_ref_phonemes: List[int] = field(default_factory=list)
    unmatched_pred_words: List[int] = field(default_factory=list)
    unmatched_ref_words: List[int] = field(default_factory=list)

    alignment_metadata: Dict = field(default_factory=dict)

    @property
    def n_note_matches(self) -> int:
        return len(self.note_matches)

    @property
    def n_phoneme_matches(self) -> int:
        return len(self.phoneme_matches)

    @property
    def note_precision(self) -> Optional[float]:
        """Fraction of predicted notes that were matched."""
        total = self.n_note_matches + len(self.unmatched_pred_notes)
        return self.n_note_matches / total if total > 0 else None

    @property
    def note_recall(self) -> Optional[float]:
        """Fraction of reference notes that were matched."""
        total = self.n_note_matches + len(self.unmatched_ref_notes)
        return self.n_note_matches / total if total > 0 else None

    def to_dict(self) -> Dict:
        return {
            "predicted_audio_path": self.predicted_audio_path,
            "reference_source_path": self.reference_source_path,
            "n_note_matches": self.n_note_matches,
            "n_phoneme_matches": self.n_phoneme_matches,
            "note_precision": self.note_precision,
            "note_recall": self.note_recall,
            "unmatched_pred_notes": self.unmatched_pred_notes,
            "unmatched_ref_notes": self.unmatched_ref_notes,
            "unmatched_pred_phonemes": self.unmatched_pred_phonemes,
            "unmatched_ref_phonemes": self.unmatched_ref_phonemes,
            "unmatched_pred_words": self.unmatched_pred_words,
            "unmatched_ref_words": self.unmatched_ref_words,
            "note_matches": [m.to_dict() for m in self.note_matches],
            "phoneme_matches": [m.to_dict() for m in self.phoneme_matches],
            "word_matches": [m.to_dict() for m in self.word_matches],
            "alignment_metadata": self.alignment_metadata,
        }


# ---------------------------------------------------------------------------
# Phase 6 — Metric computation data structures
# ---------------------------------------------------------------------------

@dataclass
class MetricBreakdown:
    """Per-event metric detail (note, phoneme, or word level)."""

    event_idx: int
    value: Optional[float]
    label: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PitchMetrics:
    """Aggregate and per-note pitch evaluation metrics."""

    pitch_accuracy: Optional[float] = None          # fraction of notes within tolerance
    pitch_rmse_cents: Optional[float] = None        # RMSE of pitch deviation in cents
    mace_cents: Optional[float] = None              # Mean Absolute Cent Error
    note_pitch_accuracy: Optional[float] = None     # alias for pitch_accuracy
    mean_pitch_deviation_cents: Optional[float] = None  # signed mean (positive = sharp)
    n_evaluated: int = 0                            # notes with valid pitch deviation
    tolerance_cents: float = 50.0
    per_note: List[MetricBreakdown] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "pitch_accuracy": self.pitch_accuracy,
            "pitch_rmse_cents": self.pitch_rmse_cents,
            "mace_cents": self.mace_cents,
            "note_pitch_accuracy": self.note_pitch_accuracy,
            "mean_pitch_deviation_cents": self.mean_pitch_deviation_cents,
            "n_evaluated": self.n_evaluated,
            "tolerance_cents": self.tolerance_cents,
            "per_note": [b.to_dict() for b in self.per_note],
            "metadata": self.metadata,
        }


@dataclass
class TimingMetrics:
    """Aggregate and per-note timing evaluation metrics."""

    mean_onset_error_ms: Optional[float] = None        # signed mean (positive = late)
    std_onset_error_ms: Optional[float] = None
    mean_abs_onset_error_ms: Optional[float] = None
    median_onset_error_ms: Optional[float] = None
    mean_offset_error_ms: Optional[float] = None
    mean_abs_offset_error_ms: Optional[float] = None
    timing_accuracy: Optional[float] = None            # fraction within tolerance
    ioi_mae_ms: Optional[float] = None                 # inter-onset interval MAE
    n_evaluated: int = 0
    tolerance_ms: float = 50.0
    per_note: List[MetricBreakdown] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "mean_onset_error_ms": self.mean_onset_error_ms,
            "std_onset_error_ms": self.std_onset_error_ms,
            "mean_abs_onset_error_ms": self.mean_abs_onset_error_ms,
            "median_onset_error_ms": self.median_onset_error_ms,
            "mean_offset_error_ms": self.mean_offset_error_ms,
            "mean_abs_offset_error_ms": self.mean_abs_offset_error_ms,
            "timing_accuracy": self.timing_accuracy,
            "ioi_mae_ms": self.ioi_mae_ms,
            "n_evaluated": self.n_evaluated,
            "tolerance_ms": self.tolerance_ms,
            "per_note": [b.to_dict() for b in self.per_note],
            "metadata": self.metadata,
        }


@dataclass
class DurationMetrics:
    """Aggregate and per-note duration evaluation metrics."""

    mean_duration_error_s: Optional[float] = None      # signed mean (positive = too long)
    mean_abs_duration_error_s: Optional[float] = None
    std_duration_error_s: Optional[float] = None
    mean_duration_ratio: Optional[float] = None        # pred_dur / ref_dur (1.0 = perfect)
    mean_relative_duration_error: Optional[float] = None  # mean |error| / ref_dur
    n_evaluated: int = 0
    per_note: List[MetricBreakdown] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "mean_duration_error_s": self.mean_duration_error_s,
            "mean_abs_duration_error_s": self.mean_abs_duration_error_s,
            "std_duration_error_s": self.std_duration_error_s,
            "mean_duration_ratio": self.mean_duration_ratio,
            "mean_relative_duration_error": self.mean_relative_duration_error,
            "n_evaluated": self.n_evaluated,
            "per_note": [b.to_dict() for b in self.per_note],
            "metadata": self.metadata,
        }


@dataclass
class LyricMetrics:
    """Aggregate phoneme and word-level lyric timing metrics."""

    mean_phoneme_boundary_error_ms: Optional[float] = None   # signed mean
    mean_abs_phoneme_boundary_error_ms: Optional[float] = None
    std_phoneme_boundary_error_ms: Optional[float] = None
    phoneme_overlap_accuracy: Optional[float] = None   # fraction with overlap_fraction >= 0.5
    word_alignment_accuracy: Optional[float] = None    # fraction of ref words matched
    label_match_rate: Optional[float] = None           # fraction of phoneme pairs where labels match
    n_phoneme_matches: int = 0
    n_word_matches: int = 0
    tolerance_ms: float = 30.0
    per_phoneme: List[MetricBreakdown] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "mean_phoneme_boundary_error_ms": self.mean_phoneme_boundary_error_ms,
            "mean_abs_phoneme_boundary_error_ms": self.mean_abs_phoneme_boundary_error_ms,
            "std_phoneme_boundary_error_ms": self.std_phoneme_boundary_error_ms,
            "phoneme_overlap_accuracy": self.phoneme_overlap_accuracy,
            "word_alignment_accuracy": self.word_alignment_accuracy,
            "label_match_rate": self.label_match_rate,
            "n_phoneme_matches": self.n_phoneme_matches,
            "n_word_matches": self.n_word_matches,
            "tolerance_ms": self.tolerance_ms,
            "per_phoneme": [b.to_dict() for b in self.per_phoneme],
            "metadata": self.metadata,
        }


@dataclass
class PerformanceMetricsReport:
    """
    Full metric computation report for one performance evaluation.

    Produced by metrics/reporting.build_metrics_report().
    Consumed by the scoring layer (Phase 7+) and visualization.

    All sub-reports are Optional; they are None when the corresponding
    alignment data was not available at computation time.
    """

    audio_path: str
    reference_source_path: str

    # Sub-reports per metric category
    pitch: Optional[PitchMetrics] = None
    timing: Optional[TimingMetrics] = None
    duration: Optional[DurationMetrics] = None
    lyric: Optional[LyricMetrics] = None

    # Note-level summary (derived from AlignmentResult)
    note_precision: Optional[float] = None
    note_recall: Optional[float] = None
    n_note_matches: int = 0
    n_reference_notes: int = 0
    n_predicted_notes: int = 0

    # Computation metadata (config, timing, data availability)
    computation_metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        out: Dict = {
            "audio_path": self.audio_path,
            "reference_source_path": self.reference_source_path,
            "note_precision": self.note_precision,
            "note_recall": self.note_recall,
            "n_note_matches": self.n_note_matches,
            "n_reference_notes": self.n_reference_notes,
            "n_predicted_notes": self.n_predicted_notes,
            "computation_metadata": self.computation_metadata,
        }
        if self.pitch is not None:
            out["pitch"] = self.pitch.to_dict()
        if self.timing is not None:
            out["timing"] = self.timing.to_dict()
        if self.duration is not None:
            out["duration"] = self.duration.to_dict()
        if self.lyric is not None:
            out["lyric"] = self.lyric.to_dict()
        return out


# ---------------------------------------------------------------------------
# Phase 7 — Scoring engine data structures
# ---------------------------------------------------------------------------

@dataclass
class ScoreBreakdown:
    """
    One scored component within a CategoryScore.

    Produced by individual scoring functions (compute_intonation_score,
    compute_rhythm_stability_score, etc.) and collected into CategoryScore.components.
    """

    component: str                      # e.g. "intonation", "accuracy", "stability"
    raw_value: Optional[float]          # untransformed metric (cents, ms, fraction)
    score: float                        # normalized score ∈ [0.0, 100.0]
    weight: float = 1.0                 # weight used in category aggregation
    confidence: Optional[float] = None  # ∈ [0.0, 1.0]; None = not determined
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "component": self.component,
            "raw_value": self.raw_value,
            "score": self.score,
            "weight": self.weight,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class CategoryScore:
    """
    Aggregate score for one evaluation category (pitch, timing, duration, lyric).

    Produced by scoring/pitch_scoring.compute_pitch_score(), etc.
    Consumed by scoring/performance_scoring.build_performance_score_report().
    """

    category: str                       # "pitch" | "timing" | "duration" | "lyric"
    score: float                        # weighted aggregate ∈ [0.0, 100.0]
    confidence: Optional[float] = None  # ∈ [0.0, 1.0]; confidence in the score
    components: List[ScoreBreakdown] = field(default_factory=list)
    n_evaluated: int = 0                # number of events that contributed
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "category": self.category,
            "score": self.score,
            "confidence": self.confidence,
            "n_evaluated": self.n_evaluated,
            "components": [c.to_dict() for c in self.components],
            "metadata": self.metadata,
        }


@dataclass
class PerformanceScoreReport:
    """
    Full scoring report for one performance evaluation.

    Produced by scoring/performance_scoring.build_performance_score_report().
    Consumed by scoring/interpretation.build_interpretation_summary() and
    visualization/scoring_viz.py.

    All CategoryScore fields are Optional; they are None when the corresponding
    metrics sub-report was not available.
    """

    audio_path: str
    reference_source_path: str

    # Category scores
    pitch_score: Optional[CategoryScore] = None
    timing_score: Optional[CategoryScore] = None
    duration_score: Optional[CategoryScore] = None
    lyric_score: Optional[CategoryScore] = None

    # Aggregated overall score ∈ [0.0, 100.0]; None when no categories available
    overall_score: Optional[float] = None

    # Weights actually used in overall aggregation (category → effective weight)
    weights_used: Dict = field(default_factory=dict)

    # Scoring metadata (config, timing, etc.)
    score_metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        out: Dict = {
            "audio_path": self.audio_path,
            "reference_source_path": self.reference_source_path,
            "overall_score": self.overall_score,
            "weights_used": self.weights_used,
            "score_metadata": self.score_metadata,
        }
        for key, val in [
            ("pitch_score", self.pitch_score),
            ("timing_score", self.timing_score),
            ("duration_score", self.duration_score),
            ("lyric_score", self.lyric_score),
        ]:
            out[key] = val.to_dict() if val is not None else None
        return out


@dataclass
class InterpretationSummary:
    """
    Deterministic rule-based interpretation of a PerformanceScoreReport.

    Produced by scoring/interpretation.build_interpretation_summary().
    All interpretation is threshold-based with no freeform generation or LLM usage.

    overall_level is one of: "excellent" | "good" | "fair" | "needs_work".
    category_levels maps each evaluated category to the same level vocabulary.
    """

    audio_path: str
    overall_level: str                          # "excellent" | "good" | "fair" | "needs_work"

    strengths: List[str] = field(default_factory=list)   # positive rule-triggered messages
    weaknesses: List[str] = field(default_factory=list)  # improvement rule-triggered messages

    category_levels: Dict[str, str] = field(default_factory=dict)  # category → level

    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "audio_path": self.audio_path,
            "overall_level": self.overall_level,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "category_levels": self.category_levels,
            "metadata": self.metadata,
        }
