"""
Phoneme Boundary Detection Module for Singing Voice Evaluation

A production-grade pipeline for extracting phoneme boundaries from audio using
pretrained Wav2Vec2 models with CTC alignment. Designed for high-precision
phoneme timing in singing voice analysis.
"""

# ============================================================================
# ENVIRONMENT SETUP — must precede any import that touches espeak/phonemizer
# ============================================================================
import os

os.environ.setdefault("PHONEMIZER_ESPEAK_PATH", r"C:\Program Files\eSpeak NG\espeak-ng.exe")
os.environ.setdefault("ESPEAKNG_PATH", r"C:\Program Files\eSpeak NG")
os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", r"C:\Program Files\eSpeak NG\libespeak-ng.dll")
_espeak_dir = r"C:\Program Files\eSpeak NG"
if _espeak_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] += f";{_espeak_dir}"

# ============================================================================
# STANDARD IMPORTS
# ============================================================================
import json
import logging
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import (
    AutoModelForCTC,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2PhonemeCTCTokenizer,
    Wav2Vec2Processor,
)

# ============================================================================
# LOGGING — configure before first getLogger call
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# CONFIGURATION & DATA CLASSES
# ============================================================================

@dataclass
class PhonemeSegment:
    """Phoneme boundary with timing and confidence."""
    phoneme: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    frame_start: int = 0
    frame_end: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PhonemeBoundaryConfig:
    """Runtime configuration for the phoneme boundary pipeline."""
    model_name: str = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    sample_rate: int = 16000
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    batch_size: int = 1
    use_fp16: bool = False
    collapse_repeated_tokens: bool = True
    remove_blank_tokens: bool = True
    blank_token_id: int = 0
    # Post-processing A — long-segment splitting (rarely triggers for singing,
    # kept for completeness when segments are unusually long).
    max_segment_ms: float = 300.0   # split segments longer than this (0 = disabled)
    min_split_prob: float = 0.25    # min alt-token probability to justify a split
    # Post-processing B — blank-region phoneme recovery.
    # In singing, brief consonants between sustained vowels never win the CTC
    # argmax (blank wins instead) yet still register a meaningful probability
    # bump in the logits.  Scanning those blank frames and inserting a segment
    # at the probability peak recovers the missed boundaries.
    blank_region_scan: bool = True   # enable blank-region scanning
    min_phoneme_prob: float = 0.05   # min non-blank prob to insert a segment
    min_gap_frames: int = 2          # skip gaps shorter than this (noise guard; 2 frames = 40 ms)

    def __post_init__(self):
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        elif not isinstance(self.device, torch.device):
            raise TypeError("device must be a torch.device or a string")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but not available")


# ============================================================================
# AUDIO LOADING & PREPROCESSING
# ============================================================================

def load_audio(
    filepath: str,
    target_sr: int = 16000,
    mono: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Load and normalise an audio file to a 1-D float32 CPU tensor.

    Falls back from torchaudio to soundfile when the primary loader fails
    (e.g. missing FFmpeg / TorchCodec on Windows).

    Returns:
        (waveform [seq], sample_rate)
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg"}:
        raise ValueError(f"Unsupported audio format: {path.suffix}")

    try:
        waveform, sr = torchaudio.load(str(path))
        logger.info(f"Loaded (torchaudio): {path.name} | SR={sr} | shape={waveform.shape}")
    except Exception as e:
        logger.warning(f"torchaudio failed, falling back to soundfile: {e}")
        try:
            data, sr = sf.read(str(path), always_2d=True)
            waveform = torch.from_numpy(data.T.astype(np.float32))
            logger.info(f"Loaded (soundfile): {path.name} | SR={sr} | shape={waveform.shape}")
        except Exception as e2:
            raise RuntimeError(f"Failed to load audio: {e2}") from e2

    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr != target_sr:
        waveform = T.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
        sr = target_sr

    max_val = torch.abs(waveform).max()
    if max_val > 0:
        waveform = waveform / max_val

    return waveform.squeeze(0), sr  # 1-D, CPU


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(config: PhonemeBoundaryConfig) -> Tuple[AutoModelForCTC, Wav2Vec2Processor]:
    """
    Download / cache and initialise Wav2Vec2Phoneme model + processor.

    Uses Wav2Vec2PhonemeCTCTokenizer explicitly so that IPA phoneme tokens
    stored in added_tokens.json are included in the vocabulary, fixing the
    <unk> label issue caused by get_vocab() missing those tokens.

    Returns:
        (model on config.device in eval mode, Wav2Vec2Processor)
    """
    try:
        logger.info(f"Loading model: {config.model_name} → {config.device}")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.model_name)
        tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(config.model_name)
        processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor, tokenizer=tokenizer
        )
        model = AutoModelForCTC.from_pretrained(config.model_name)
        model = model.to(config.device).eval()
        logger.info("Model loaded successfully")
        return model, processor
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{config.model_name}': {e}") from e


# ============================================================================
# VOCABULARY MAPPING
# ============================================================================

def build_id2phoneme(model: AutoModelForCTC, processor: Wav2Vec2Processor) -> Dict[int, str]:
    """
    Build token-ID → phoneme-string mapping.

    Priority:
    1. processor.tokenizer._added_tokens_decoder — AddedToken objects loaded
       from added_tokens.json by Wav2Vec2PhonemeCTCTokenizer; this is the only
       source that reliably contains all IPA phoneme tokens.
    2. processor.tokenizer.get_vocab() — fills in base special tokens (<pad> etc.)
       not covered by _added_tokens_decoder.
    3. model.config.id2label — last resort if both tokenizer sources are empty.
    """
    mapping: Dict[int, str] = {}

    # Source 1: added_tokens_decoder — IPA phonemes live here after
    # Wav2Vec2PhonemeCTCTokenizer.from_pretrained() loads added_tokens.json.
    # model.config.id2label is often absent or only covers special tokens for
    # this checkpoint, so it cannot be used as the primary source.
    for tok_id, added_tok in processor.tokenizer._added_tokens_decoder.items():
        mapping[int(tok_id)] = str(added_tok)

    # Source 2: base vocab (special tokens not already covered above)
    for tok_str, tok_id in processor.tokenizer.get_vocab().items():
        if int(tok_id) not in mapping:
            mapping[int(tok_id)] = tok_str

    # Source 3: id2label fallback when both tokenizer sources are empty
    if not mapping:
        id2label = getattr(model.config, "id2label", None)
        if id2label:
            mapping = {int(k): v for k, v in id2label.items()}
            logger.warning(
                f"id2phoneme fell back to model.config.id2label ({len(mapping)} tokens)"
            )

    if not mapping:
        logger.error(
            "id2phoneme mapping is empty — all tokens will decode as <unk>. "
            "Verify that Wav2Vec2PhonemeCTCTokenizer was used (not AutoProcessor)."
        )

    logger.debug(f"id2phoneme: {len(mapping)} / {model.config.vocab_size} tokens")
    return mapping


def compute_frame_duration(model: AutoModelForCTC, sample_rate: int) -> float:
    """
    Derive seconds-per-encoder-frame from conv_stride in the model config.

    Wav2Vec2 stacks several strided convolutions; total stride = product of
    all per-layer strides.  Falls back to the canonical 20 ms if the config
    attribute is absent.
    """
    if hasattr(model.config, "conv_stride"):
        total_stride = 1
        for s in model.config.conv_stride:
            total_stride *= s
        frame_dur = total_stride / sample_rate
        logger.debug(f"Frame duration: {frame_dur * 1000:.2f} ms (from conv_stride)")
        return frame_dur

    logger.warning("conv_stride missing from model config; using fallback 20 ms")
    return 0.02


# ============================================================================
# INFERENCE
# ============================================================================

def run_inference(
    audio: torch.Tensor,
    model: AutoModelForCTC,
    processor: Wav2Vec2Processor,
    config: PhonemeBoundaryConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess audio and run a forward pass.

    audio must be a 1-D CPU float32 tensor (as returned by load_audio).

    Returns:
        logits [1, seq_len, vocab_size] on config.device
        input_values [1, audio_len] on config.device
    """
    audio_np = audio.cpu().numpy()  # explicit .cpu() guards against accidental GPU tensors

    with torch.no_grad():
        inputs = processor(
            audio_np,
            sampling_rate=config.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(config.device)
        logits = model(input_values).logits  # [1, T, V]

    logger.info(f"Logits: shape={logits.shape} device={logits.device}")
    return logits, input_values


# ============================================================================
# CTC ALIGNMENT
# ============================================================================

def ctc_align(
    pred_ids: np.ndarray,
    collapse_repeated: bool = True,
    remove_blanks: bool = True,
    blank_id: int = 0,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Convert frame-level CTC argmax predictions into phoneme segments.

    Each output segment has an inclusive frame range [start_frame, end_frame].

    When collapse_repeated=True (standard CTC):
        - Consecutive identical tokens are merged into one segment whose
          end_frame advances with each repetition.
    When collapse_repeated=False:
        - Every frame becomes its own segment (useful for analysis).

    Bug fixed: the original code mutated frame_ranges[-1][1] even when not
    collapsing, creating overlapping ranges.

    Args:
        pred_ids:          1-D array of argmax token IDs, one per encoder frame
        collapse_repeated: Merge consecutive identical non-blank tokens
        remove_blanks:     Drop blank token frames from output
        blank_id:          CTC blank token ID (0 for HuggingFace Wav2Vec2)

    Returns:
        aligned_tokens:  List of token IDs (one per segment)
        frame_ranges:    List of (start_frame, end_frame) inclusive tuples
    """
    if len(pred_ids) == 0:
        return [], []

    aligned_tokens: List[int] = []
    frame_ranges: List[List[int]] = []

    for i, raw_id in enumerate(pred_ids):
        token_id = int(raw_id)

        if remove_blanks and token_id == blank_id:
            continue

        if aligned_tokens and token_id == aligned_tokens[-1]:
            if collapse_repeated:
                frame_ranges[-1][1] = i  # extend current segment
                continue
            # Not collapsing: start a fresh segment for this frame
            frame_ranges.append([i, i])
            aligned_tokens.append(token_id)
        else:
            frame_ranges.append([i, i])
            aligned_tokens.append(token_id)

    logger.debug(f"CTC: {len(pred_ids)} frames → {len(aligned_tokens)} tokens")
    return aligned_tokens, [tuple(r) for r in frame_ranges]


# ============================================================================
# BOUNDARY EXTRACTION
# ============================================================================

def extract_phoneme_boundaries(
    logits: torch.Tensor,
    model: AutoModelForCTC,        # FIX: was missing; needed for id2label + conv_stride
    processor: Wav2Vec2Processor,
    config: PhonemeBoundaryConfig,
    input_length: Optional[int] = None,
) -> Tuple[List[str], List[Tuple[float, float]], List[Tuple[int, int]], List[float]]:
    """
    Decode CTC logits into timed phoneme segments.

    Returns:
        phonemes:        Decoded phoneme strings
        time_boundaries: (start_s, end_s) per phoneme
        frame_ranges:    (start_frame, end_frame) per phoneme (inclusive)
        confidences:     Mean softmax probability per segment
    """
    logits_cpu = logits.cpu()  # [1, T, V]

    # Greedy decode — use index [0] (not squeeze) to safely remove batch dim
    # even if T == 1, avoiding accidental scalar collapse.
    pred_ids: np.ndarray = torch.argmax(logits_cpu, dim=-1)[0].numpy()  # [T]

    logger.debug(f"pred_ids shape={pred_ids.shape}, unique={np.unique(pred_ids)}")

    aligned_tokens, frame_ranges = ctc_align(
        pred_ids,
        collapse_repeated=config.collapse_repeated_tokens,
        remove_blanks=config.remove_blank_tokens,
        blank_id=config.blank_token_id,
    )

    if not aligned_tokens:
        logger.warning("No phonemes detected after CTC alignment — check audio or model")
        return [], [], [], []

    id2phoneme = build_id2phoneme(model, processor)

    phonemes: List[str] = []
    for token_id in aligned_tokens:
        phoneme = id2phoneme.get(token_id)
        if phoneme is None:
            logger.warning(f"Unknown token ID {token_id} — substituting <unk>")
            phoneme = "<unk>"
        phonemes.append(phoneme)

    # Per-segment confidence: mean softmax prob of the predicted token
    probs = torch.softmax(logits_cpu[0], dim=-1).numpy()  # [T, V]
    confidences: List[float] = []
    for (fs, fe), tid in zip(frame_ranges, aligned_tokens):
        seg_probs = probs[fs : fe + 1, tid]
        confidences.append(float(seg_probs.mean()))

    # Time conversion using model-derived frame duration (not hardcoded)
    frame_dur = compute_frame_duration(model, config.sample_rate)
    time_boundaries: List[Tuple[float, float]] = [
        (fs * frame_dur, (fe + 1) * frame_dur)
        for fs, fe in frame_ranges
    ]

    logger.debug(
        f"Decoded {len(phonemes)} phonemes | frame_dur={frame_dur * 1000:.2f} ms"
    )
    return phonemes, time_boundaries, list(frame_ranges), confidences, probs, id2phoneme


# ============================================================================
# SEGMENT CONSTRUCTION
# ============================================================================

def create_phoneme_segments(
    phonemes: List[str],
    time_boundaries: List[Tuple[float, float]],
    frame_ranges: Optional[List[Tuple[int, int]]] = None,
    confidences: Optional[List[float]] = None,
) -> List[PhonemeSegment]:
    """
    Zip decoded phoneme data into PhonemeSegment objects.
    """
    if len(phonemes) != len(time_boundaries):
        raise ValueError(
            f"Length mismatch: phonemes={len(phonemes)}, "
            f"boundaries={len(time_boundaries)}"
        )

    segments: List[PhonemeSegment] = []
    for i, (ph, (t0, t1)) in enumerate(zip(phonemes, time_boundaries)):
        segments.append(PhonemeSegment(
            phoneme=ph,
            start_time=round(t0, 4),
            end_time=round(t1, 4),
            confidence=round(confidences[i], 4) if confidences else 1.0,
            frame_start=frame_ranges[i][0] if frame_ranges else 0,
            frame_end=frame_ranges[i][1] if frame_ranges else 0,
        ))
    return segments


def group_by_words(
    segments: List[PhonemeSegment],
    word_separator: str = "|",
) -> List[Dict]:
    """
    Group consecutive phoneme segments into words using a separator token.
    """
    words: List[Dict] = []
    current: Dict = {"phonemes": [], "start_time": None, "end_time": None}

    for seg in segments:
        if seg.phoneme == word_separator:
            if current["phonemes"]:
                words.append(current)
            current = {"phonemes": [], "start_time": None, "end_time": None}
        else:
            current["phonemes"].append(seg.phoneme)
            if current["start_time"] is None:
                current["start_time"] = seg.start_time
            current["end_time"] = seg.end_time

    if current["phonemes"]:
        words.append(current)

    return words


# ============================================================================
# SEGMENT SPLITTING
# ============================================================================

def split_long_segments(
    segments: List[PhonemeSegment],
    probs: np.ndarray,
    id2phoneme: Dict[int, str],
    frame_dur: float,
    max_segment_ms: float,
    min_split_prob: float,
) -> List[PhonemeSegment]:
    """
    Post-processing pass that splits over-long CTC segments at their best
    internal phoneme-transition candidate.

    Why this is needed:
        In singing, vowels are held for 300–800 ms.  The CTC model was trained
        on speech where vowels last ~100 ms.  When it encounters a long vowel
        followed by a consonant (e.g. "let"), it often keeps the argmax on the
        vowel for the entire span, merging two phonemes into one CTC segment.
        This causes systematic under-prediction (predictions < references).

    Algorithm per long segment [fs, fe]:
        1. Build an "alternative probability" trace: for each frame, find the
           highest-probability token that is neither blank (0) nor the segment's
           own winning token.
        2. Search only interior frames (1 … n-2) to avoid zero-length splits.
        3. If the peak alternative probability exceeds min_split_prob, split
           there and label the tail with the alternative token's phoneme.
        4. Recurse on each half in case a sub-segment is still too long.
    """
    max_frames = max(2, int(max_segment_ms / (frame_dur * 1000)))
    result: List[PhonemeSegment] = []

    for seg in segments:
        n_frames = seg.frame_end - seg.frame_start + 1
        if n_frames <= max_frames:
            result.append(seg)
            continue

        seg_probs = probs[seg.frame_start : seg.frame_end + 1]  # [n, V]

        # Identify winning token: most frequent non-blank argmax in the segment
        frame_argmax = np.argmax(seg_probs, axis=1)             # [n]
        non_blank = frame_argmax[frame_argmax != 0]
        if len(non_blank) == 0:
            result.append(seg)
            continue
        winning_id = int(np.bincount(non_blank).argmax())

        # Build alternative-probability trace (mask blank and winning token)
        alt_probs = seg_probs.copy()
        alt_probs[:, 0] = 0
        alt_probs[:, winning_id] = 0
        max_alt_per_frame = alt_probs.max(axis=1)  # [n]

        # Only consider interior frames so neither sub-segment is empty
        interior = max_alt_per_frame[1:-1]
        if len(interior) == 0:
            result.append(seg)
            continue

        best_local = int(np.argmax(interior)) + 1  # offset for the sliced interior
        best_prob = float(max_alt_per_frame[best_local])

        if best_prob < min_split_prob:
            result.append(seg)
            continue

        split_token_id = int(np.argmax(alt_probs[best_local]))
        split_phoneme = id2phoneme.get(split_token_id, seg.phoneme)
        abs_split = seg.frame_start + best_local

        seg1 = PhonemeSegment(
            phoneme=seg.phoneme,
            start_time=seg.start_time,
            end_time=round(abs_split * frame_dur, 4),
            confidence=round(float(seg_probs[:best_local, winning_id].mean()), 4),
            frame_start=seg.frame_start,
            frame_end=abs_split - 1,
        )
        seg2 = PhonemeSegment(
            phoneme=split_phoneme,
            start_time=round(abs_split * frame_dur, 4),
            end_time=seg.end_time,
            confidence=round(float(seg_probs[best_local:, split_token_id].mean()), 4),
            frame_start=abs_split,
            frame_end=seg.frame_end,
        )

        # Recurse: each half may itself still be too long
        result.extend(split_long_segments(
            [seg1, seg2], probs, id2phoneme, frame_dur, max_segment_ms, min_split_prob
        ))

    return result


# ============================================================================
# BLANK-REGION PHONEME RECOVERY
# ============================================================================

def insert_blank_region_phonemes(
    segments: List[PhonemeSegment],
    probs: np.ndarray,
    id2phoneme: Dict[int, str],
    frame_dur: float,
    min_phoneme_prob: float,
    min_gap_frames: int,
) -> List[PhonemeSegment]:
    """
    Recover phonemes suppressed by CTC blank dominance.

    Why blank regions contain missed boundaries:
        Wav2Vec2 was trained on speech where consonants last ~50–100 ms and
        are strong enough to win the frame-level argmax.  In singing, the same
        consonants are brief and sandwiched between long sustained vowels; the
        blank token often wins the argmax even while the consonant's softmax
        probability is clearly elevated (0.10–0.25).  Scanning the blank frames
        between detected segments and inserting a new segment at the probability
        peak recovers these timing boundaries.

    Algorithm per inter-segment gap [gap_start, gap_end]:
        1. If gap is shorter than min_gap_frames, skip (transition noise).
        2. Compute per-frame max non-blank probability.
        3. At the frame where this peaks, record the winning non-blank token.
        4. If the peak exceeds min_phoneme_prob, insert a 1-frame segment there.
        5. Merge all insertions back into the sorted segment list.
    """
    if len(segments) < 2:
        return segments

    insertions: List[PhonemeSegment] = []

    for i in range(len(segments) - 1):
        gap_start = segments[i].frame_end + 1
        gap_end = segments[i + 1].frame_start - 1
        gap_len = gap_end - gap_start + 1

        if gap_len < min_gap_frames:
            continue

        gap_probs = probs[gap_start : gap_end + 1, :]  # [gap_len, V]
        non_blank = gap_probs[:, 1:]                    # exclude blank (col 0)
        peak_per_frame = non_blank.max(axis=1)          # [gap_len]

        best_local = int(np.argmax(peak_per_frame))
        peak_val = float(peak_per_frame[best_local])

        if peak_val < min_phoneme_prob:
            continue

        # +1 offsets back to the full vocab indexing (we dropped col 0 above)
        best_token_id = int(np.argmax(non_blank[best_local])) + 1
        phoneme = id2phoneme.get(best_token_id, "<unk>")
        abs_frame = gap_start + best_local

        insertions.append(PhonemeSegment(
            phoneme=phoneme,
            start_time=round(abs_frame * frame_dur, 4),
            end_time=round((abs_frame + 1) * frame_dur, 4),
            confidence=round(peak_val, 4),
            frame_start=abs_frame,
            frame_end=abs_frame,
        ))

    if not insertions:
        return segments

    merged = list(segments) + insertions
    merged.sort(key=lambda s: s.start_time)
    logger.debug(f"Blank-region scan: inserted {len(insertions)} segment(s)")
    return merged


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_phoneme_boundaries(
    audio: torch.Tensor,
    segments: List[PhonemeSegment],
    sample_rate: int = 16000,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 6),
) -> None:
    """
    Waveform + colour-coded phoneme segment timeline.

    Saves to save_path if given, otherwise calls plt.show().
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping visualization")
        return

    audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
    time_axis = np.arange(len(audio_np)) / sample_rate

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ax1.plot(time_axis, audio_np, linewidth=0.5, color="steelblue", alpha=0.7)
    ax1.set_ylabel("Amplitude", fontsize=10)
    ax1.set_title("Waveform with Phoneme Boundaries", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Preserve insertion order for colour consistency
    unique_phonemes = list(dict.fromkeys(s.phoneme for s in segments))
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(unique_phonemes), 1)))
    color_map = {p: palette[i] for i, p in enumerate(unique_phonemes)}

    for seg in segments:
        color = color_map[seg.phoneme]
        ax2.axvspan(seg.start_time, seg.end_time, alpha=0.3, color=color)
        mid = (seg.start_time + seg.end_time) / 2
        ax2.text(mid, 0.5, seg.phoneme, ha="center", va="center",
                 fontsize=9, fontweight="bold")

    ax2.set_xlabel("Time (seconds)", fontsize=10)
    ax2.set_ylabel("Phonemes", fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


# ============================================================================
# EVALUATION
# ============================================================================

def compute_boundary_metrics(
    predicted: List[PhonemeSegment],
    reference: List[PhonemeSegment],
    tolerance_ms: float = 50.0,
) -> Dict[str, float]:
    """
    Compute precision / recall / F1 / MAE against reference boundaries.

    Boundaries are the start times of all segments plus the end time of the
    final segment.  A predicted boundary is a match if it falls within
    tolerance_ms of any reference boundary.
    """
    if not predicted or not reference:
        logger.warning("Empty prediction or reference list")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mae_ms": 0.0}

    tol = tolerance_ms / 1000.0
    pred_b = [s.start_time for s in predicted] + [predicted[-1].end_time]
    ref_b = [s.start_time for s in reference] + [reference[-1].end_time]

    matches = 0
    total_error = 0.0
    for pt in pred_b:
        nearest = min(abs(pt - rt) for rt in ref_b)
        if nearest <= tol:
            matches += 1
            total_error += nearest

    precision = matches / len(pred_b) if pred_b else 0.0
    recall = matches / len(ref_b) if ref_b else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    mae_ms = (total_error / matches * 1000) if matches > 0 else float("inf")

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mae_ms": round(mae_ms, 2),
        "matches": matches,
        "total_boundaries": len(pred_b),
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def extract_phoneme_boundaries_from_audio(
    audio_filepath: str,
    config: Optional[PhonemeBoundaryConfig] = None,
    return_segments: bool = True,
    word_grouping: bool = False,
) -> Dict:
    """
    End-to-end pipeline: audio file → structured phoneme boundary output.

    Returns dict with keys:
        phonemes, boundaries, metadata,
        segments (if return_segments=True),
        words    (if word_grouping=True)
    """
    if config is None:
        config = PhonemeBoundaryConfig()

    logger.info("=" * 70)
    logger.info("PHONEME BOUNDARY EXTRACTION PIPELINE")
    logger.info("=" * 70)

    try:
        logger.info("[1/5] Loading audio...")
        audio, sr = load_audio(audio_filepath, target_sr=config.sample_rate)
        duration = audio.shape[0] / config.sample_rate
        logger.info(f"Duration: {duration:.2f}s")

        logger.info("[2/5] Loading model...")
        model, processor = load_model(config)

        logger.info("[3/5] Running inference...")
        logits, input_values = run_inference(audio, model, processor, config)

        logger.info("[4/5] Extracting phoneme boundaries...")
        phonemes, time_boundaries, frame_ranges, confidences, probs, id2phoneme = (
            extract_phoneme_boundaries(logits, model, processor, config, input_values.shape[1])
        )

        logger.info("[5/5] Building output...")
        segments = create_phoneme_segments(
            phonemes, time_boundaries, frame_ranges, confidences
        )

        frame_dur = compute_frame_duration(model, config.sample_rate)

        if config.max_segment_ms > 0 and segments:
            n_before = len(segments)
            segments = split_long_segments(
                segments, probs, id2phoneme, frame_dur,
                config.max_segment_ms, config.min_split_prob,
            )
            n_split = len(segments) - n_before
            if n_split:
                logger.info(f"Segment splitting: {n_before} → {len(segments)} (+{n_split})")

        if config.blank_region_scan and segments:
            n_before = len(segments)
            segments = insert_blank_region_phonemes(
                segments, probs, id2phoneme, frame_dur,
                config.min_phoneme_prob, config.min_gap_frames,
            )
            n_inserted = len(segments) - n_before
            if n_inserted:
                logger.info(f"Blank-region scan: {n_before} → {len(segments)} (+{n_inserted})")

        if segments:
            logger.info(
                f"Extracted {len(segments)} phonemes | "
                f"span: 0.00s – {segments[-1].end_time:.2f}s"
            )
        else:
            logger.warning("Pipeline produced zero phoneme segments")

        output: Dict = {
            "phonemes": phonemes,
            "boundaries": time_boundaries,
            "metadata": {
                "audio_file": Path(audio_filepath).name,
                "duration_s": round(duration, 2),
                "num_phonemes": len(segments),
                "model": config.model_name,
                "device": str(config.device),  # FIX: torch.device is not JSON-serialisable
            },
        }

        if return_segments:
            output["segments"] = segments
        if word_grouping:
            output["words"] = group_by_words(segments)

        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 70)
        return output

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    """
    CLI entry-point.

    Usage:
        python phoneme_model.py <audio.wav> [--output out.json] [--plot] [--words]
                                            [--reference ref.json] [--device cpu|cuda]
    """
    import argparse
    import sys

    # IPA symbols can't encode in the default Windows CP1252 console.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Extract phoneme boundaries from audio using Wav2Vec2 + CTC"
    )
    parser.add_argument("audio_file", help="Path to audio file (.wav/.flac/.mp3/.ogg)")
    parser.add_argument("--output", "-o", help="Save results to JSON")
    parser.add_argument("--plot", "-p", action="store_true", help="Visualise boundaries")
    parser.add_argument("--words", "-w", action="store_true", help="Group phonemes into words")
    parser.add_argument("--reference", "-r", help="Reference JSON for evaluation metrics")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    config = PhonemeBoundaryConfig(device=args.device)
    result = extract_phoneme_boundaries_from_audio(
        args.audio_file,
        config=config,
        return_segments=True,
        word_grouping=args.words,
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    preview = result["phonemes"][:20]
    suffix = "..." if len(result["phonemes"]) > 20 else ""
    print(f"\nPhoneme sequence (first 20): {' '.join(preview)}{suffix}")

    if result.get("segments"):
        print("\nFirst 5 segments:")
        for seg in result["segments"][:5]:
            print(
                f"  {seg.phoneme:<8s}  "
                f"{seg.start_time:.3f}s – {seg.end_time:.3f}s  "
                f"conf={seg.confidence:.3f}"
            )

    if args.output:
        output_data = {
            "phonemes": result["phonemes"],
            "boundaries": [[s, e] for s, e in result["boundaries"]],
            "segments": [seg.to_dict() for seg in result.get("segments", [])],
            "metadata": result["metadata"],
        }
        if "words" in result:
            output_data["words"] = result["words"]

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Output saved to {args.output}")

    if args.plot:
        audio, _ = load_audio(args.audio_file)
        segments = result.get("segments", [])
        if segments:
            plot_path = (
                Path(args.output).stem + "_plot.png" if args.output else "phoneme_plot.png"
            )
            plot_phoneme_boundaries(audio, segments, save_path=str(plot_path))

    if args.reference:
        with open(args.reference, "r") as f:
            ref_data = json.load(f)

        ref_segments = [PhonemeSegment(**seg) for seg in ref_data.get("segments", [])]
        pred_segments = result.get("segments", [])
        metrics = compute_boundary_metrics(pred_segments, ref_segments)

        print("\n" + "=" * 70)
        print("EVALUATION METRICS")
        print("=" * 70)
        for key, val in metrics.items():
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
