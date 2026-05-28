"""
Phoneme Boundary Detection Module for Singing Voice Evaluation.

Production-grade pipeline: Wav2Vec2 + CTC alignment → timed phoneme segments.

Changes from the original Phoneme Model/phoneme_model.py:
  - load_audio() replaced by utils.audio.load_audio_torch (shared utility)
  - Device handling delegates to utils.device.get_torch_device
  - Logging setup uses utils.logging_utils
  - PhonemeInferenceModel wrapper added (implements models.base.BaseInferenceModel)
  - All prediction logic is UNCHANGED
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
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForCTC,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2PhonemeCTCTokenizer,
    Wav2Vec2Processor,
)

from models.base import BaseInferenceModel
from utils.audio import load_audio_torch
from utils.device import get_torch_device
from utils.logging_utils import get_logger

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# CONFIGURATION & DATA CLASSES
# ============================================================================

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
    max_segment_ms: float = 300.0
    min_split_prob: float = 0.25
    blank_region_scan: bool = True
    min_phoneme_prob: float = 0.05
    min_gap_frames: int = 2

    def __post_init__(self):
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        elif not isinstance(self.device, torch.device):
            raise TypeError("device must be a torch.device or a string")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but not available")

    @classmethod
    def from_yaml(cls, cfg: Dict[str, Any]) -> "PhonemeBoundaryConfig":
        """Build from a loaded configs/phoneme.yaml dict."""
        model_cfg = cfg.get("model", {})
        audio_cfg = cfg.get("audio", {})
        ctc_cfg = cfg.get("ctc", {})
        pp_cfg = cfg.get("post_processing", {})
        dev_pref = cfg.get("device", {}).get("preference", "auto")

        return cls(
            model_name=model_cfg.get("name", cls.model_name),
            sample_rate=audio_cfg.get("sample_rate", cls.sample_rate),
            device=get_torch_device(dev_pref),
            batch_size=model_cfg.get("batch_size", cls.batch_size),
            use_fp16=model_cfg.get("use_fp16", cls.use_fp16),
            collapse_repeated_tokens=ctc_cfg.get("collapse_repeated_tokens", cls.collapse_repeated_tokens),
            remove_blank_tokens=ctc_cfg.get("remove_blank_tokens", cls.remove_blank_tokens),
            blank_token_id=ctc_cfg.get("blank_token_id", cls.blank_token_id),
            max_segment_ms=pp_cfg.get("max_segment_ms", cls.max_segment_ms),
            min_split_prob=pp_cfg.get("min_split_prob", cls.min_split_prob),
            blank_region_scan=pp_cfg.get("blank_region_scan", cls.blank_region_scan),
            min_phoneme_prob=pp_cfg.get("min_phoneme_prob", cls.min_phoneme_prob),
            min_gap_frames=pp_cfg.get("min_gap_frames", cls.min_gap_frames),
        )


# PhonemeSegment is re-exported from utils.types for shared use.
# The local alias preserves backward compatibility with the original file.
from utils.types import PhonemeSegment  # noqa: E402


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(config: PhonemeBoundaryConfig) -> Tuple[AutoModelForCTC, Wav2Vec2Processor]:
    """Download / cache and initialize Wav2Vec2Phoneme model + processor."""
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
    """Build token-ID → phoneme-string mapping from the processor's tokenizer."""
    mapping: Dict[int, str] = {}

    for tok_id, added_tok in processor.tokenizer._added_tokens_decoder.items():
        mapping[int(tok_id)] = str(added_tok)

    for tok_str, tok_id in processor.tokenizer.get_vocab().items():
        if int(tok_id) not in mapping:
            mapping[int(tok_id)] = tok_str

    if not mapping:
        id2label = getattr(model.config, "id2label", None)
        if id2label:
            mapping = {int(k): v for k, v in id2label.items()}
            logger.warning(f"id2phoneme fell back to model.config.id2label ({len(mapping)} tokens)")

    if not mapping:
        logger.error("id2phoneme mapping is empty — all tokens will decode as <unk>.")

    logger.debug(f"id2phoneme: {len(mapping)} / {model.config.vocab_size} tokens")
    return mapping


def compute_frame_duration(model: AutoModelForCTC, sample_rate: int) -> float:
    """Derive seconds-per-encoder-frame from model conv_stride config."""
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
    """Preprocess audio and run a forward pass. audio must be 1-D CPU float32."""
    audio_np = audio.cpu().numpy()

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
    """Convert frame-level CTC argmax predictions into phoneme segments."""
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
                frame_ranges[-1][1] = i
                continue
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
    model: AutoModelForCTC,
    processor: Wav2Vec2Processor,
    config: PhonemeBoundaryConfig,
    input_length: Optional[int] = None,
) -> Tuple[List[str], List[Tuple[float, float]], List[Tuple[int, int]], List[float], np.ndarray, Dict[int, str]]:
    """Decode CTC logits into timed phoneme segments."""
    logits_cpu = logits.cpu()
    pred_ids: np.ndarray = torch.argmax(logits_cpu, dim=-1)[0].numpy()

    logger.debug(f"pred_ids shape={pred_ids.shape}, unique={np.unique(pred_ids)}")

    aligned_tokens, frame_ranges = ctc_align(
        pred_ids,
        collapse_repeated=config.collapse_repeated_tokens,
        remove_blanks=config.remove_blank_tokens,
        blank_id=config.blank_token_id,
    )

    if not aligned_tokens:
        logger.warning("No phonemes detected after CTC alignment — check audio or model")
        return [], [], [], [], np.array([]), {}

    id2phoneme = build_id2phoneme(model, processor)

    phonemes: List[str] = []
    for token_id in aligned_tokens:
        phoneme = id2phoneme.get(token_id)
        if phoneme is None:
            logger.warning(f"Unknown token ID {token_id} — substituting <unk>")
            phoneme = "<unk>"
        phonemes.append(phoneme)

    probs = torch.softmax(logits_cpu[0], dim=-1).numpy()  # [T, V]
    confidences: List[float] = []
    for (fs, fe), tid in zip(frame_ranges, aligned_tokens):
        seg_probs = probs[fs : fe + 1, tid]
        confidences.append(float(seg_probs.mean()))

    frame_dur = compute_frame_duration(model, config.sample_rate)
    time_boundaries: List[Tuple[float, float]] = [
        (fs * frame_dur, (fe + 1) * frame_dur)
        for fs, fe in frame_ranges
    ]

    logger.debug(f"Decoded {len(phonemes)} phonemes | frame_dur={frame_dur * 1000:.2f} ms")
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
    """Zip decoded phoneme data into PhonemeSegment objects."""
    if len(phonemes) != len(time_boundaries):
        raise ValueError(
            f"Length mismatch: phonemes={len(phonemes)}, boundaries={len(time_boundaries)}"
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
    """Group consecutive phoneme segments into words using a separator token."""
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
    """Post-processing: split over-long CTC segments at internal transition candidates."""
    max_frames = max(2, int(max_segment_ms / (frame_dur * 1000)))
    result: List[PhonemeSegment] = []

    for seg in segments:
        n_frames = seg.frame_end - seg.frame_start + 1
        if n_frames <= max_frames:
            result.append(seg)
            continue

        seg_probs = probs[seg.frame_start : seg.frame_end + 1]
        frame_argmax = np.argmax(seg_probs, axis=1)
        non_blank = frame_argmax[frame_argmax != 0]
        if len(non_blank) == 0:
            result.append(seg)
            continue
        winning_id = int(np.bincount(non_blank).argmax())

        alt_probs = seg_probs.copy()
        alt_probs[:, 0] = 0
        alt_probs[:, winning_id] = 0
        max_alt_per_frame = alt_probs.max(axis=1)

        interior = max_alt_per_frame[1:-1]
        if len(interior) == 0:
            result.append(seg)
            continue

        best_local = int(np.argmax(interior)) + 1
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
    """Recover phonemes suppressed by CTC blank dominance in singing."""
    if len(segments) < 2:
        return segments

    insertions: List[PhonemeSegment] = []

    for i in range(len(segments) - 1):
        gap_start = segments[i].frame_end + 1
        gap_end = segments[i + 1].frame_start - 1
        gap_len = gap_end - gap_start + 1

        if gap_len < min_gap_frames:
            continue

        gap_probs = probs[gap_start : gap_end + 1, :]
        non_blank = gap_probs[:, 1:]
        peak_per_frame = non_blank.max(axis=1)

        best_local = int(np.argmax(peak_per_frame))
        peak_val = float(peak_per_frame[best_local])

        if peak_val < min_phoneme_prob:
            continue

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
# METRICS
# ============================================================================

def compute_boundary_metrics(
    predicted: List[PhonemeSegment],
    reference: List[PhonemeSegment],
    tolerance_ms: float = 50.0,
) -> Dict[str, float]:
    """Precision / recall / F1 / MAE against reference boundaries."""
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
        if (precision + recall) > 0 else 0.0
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
# MAIN PIPELINE FUNCTION
# ============================================================================

def extract_phoneme_boundaries_from_audio(
    audio_filepath: str,
    config: Optional[PhonemeBoundaryConfig] = None,
    return_segments: bool = True,
    word_grouping: bool = False,
) -> Dict:
    """
    End-to-end pipeline: audio file → structured phoneme boundary output.

    Returns dict with keys: phonemes, boundaries, metadata,
    segments (if return_segments=True), words (if word_grouping=True).
    """
    if config is None:
        config = PhonemeBoundaryConfig()

    logger.info("=" * 70)
    logger.info("PHONEME BOUNDARY EXTRACTION PIPELINE")
    logger.info("=" * 70)

    try:
        logger.info("[1/5] Loading audio...")
        # Uses shared utils.audio loader; returns 1-D CPU torch.Tensor
        audio, sr = load_audio_torch(audio_filepath, target_sr=config.sample_rate)
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
                "device": str(config.device),
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
# BaseInferenceModel WRAPPER
# ============================================================================

class PhonemeInferenceModel(BaseInferenceModel):
    """
    BaseInferenceModel implementation for the phoneme boundary detector.

    Example:
        model = PhonemeInferenceModel()
        result = model.run("singing.wav")
        for seg in result["segments"]:
            print(seg.phoneme, seg.start_time, seg.end_time)
    """

    def __init__(self, config: Optional[PhonemeBoundaryConfig] = None) -> None:
        super().__init__()
        self.phoneme_config = config or PhonemeBoundaryConfig()
        self._model = None
        self._processor = None

    def load_model(self) -> None:
        self._model, self._processor = load_model(self.phoneme_config)
        self._is_loaded = True

    def predict(self, audio) -> Dict:
        if not self._is_loaded:
            self.load_model()

        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio).float()

        logits, input_values = run_inference(
            audio, self._model, self._processor, self.phoneme_config
        )
        phonemes, time_boundaries, frame_ranges, confidences, probs, id2phoneme = (
            extract_phoneme_boundaries(
                logits, self._model, self._processor, self.phoneme_config,
                input_values.shape[1],
            )
        )
        segments = create_phoneme_segments(phonemes, time_boundaries, frame_ranges, confidences)

        frame_dur = compute_frame_duration(self._model, self.phoneme_config.sample_rate)
        if self.phoneme_config.max_segment_ms > 0 and segments:
            segments = split_long_segments(
                segments, probs, id2phoneme, frame_dur,
                self.phoneme_config.max_segment_ms, self.phoneme_config.min_split_prob,
            )
        if self.phoneme_config.blank_region_scan and segments:
            segments = insert_blank_region_phonemes(
                segments, probs, id2phoneme, frame_dur,
                self.phoneme_config.min_phoneme_prob, self.phoneme_config.min_gap_frames,
            )

        return {
            "phonemes": phonemes,
            "boundaries": time_boundaries,
            "segments": segments,
            "metadata": {"model": self.phoneme_config.model_name},
        }

    def run(self, audio_path) -> Dict:
        return extract_phoneme_boundaries_from_audio(
            str(audio_path), config=self.phoneme_config,
            return_segments=True, word_grouping=False,
        )
