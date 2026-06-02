"""
models/pitch/fusion.py - Voiced/unvoiced fusion and pitch contour cleaning.

Combines the binary VAD mask with raw pitch predictions to produce a clean,
smooth F0 contour suitable for frame-level and note-level scoring.

Pipeline:
  1. Intersect pitch-model voiced flags with VAD mask
  2. Zero / NaN unvoiced frames
  3. Interpolate short unvoiced gaps
  4. Median filter voiced frames
  5. Optional Gaussian smoothing

Unchanged from the original Pitch Model w VAD/fusion.py.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from scipy.ndimage import median_filter  # type: ignore

logger = logging.getLogger(__name__)

UnvoicedFill = Literal["zero", "nan"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FusionConfig:
    """Tunable parameters for the VAD–pitch fusion and contour cleaning step."""

    unvoiced_fill: UnvoicedFill = "zero"
    median_filter_size: int = 5
    max_gap_fill_frames: int = 10
    smoothing_sigma: float = 1.0

    @classmethod
    def from_yaml(cls, cfg: Dict) -> "FusionConfig":
        f = cfg.get("fusion", {})
        return cls(
            unvoiced_fill=f.get("unvoiced_fill", "zero"),
            median_filter_size=f.get("median_filter_size", 5),
            max_gap_fill_frames=f.get("max_gap_fill_frames", 10),
            smoothing_sigma=f.get("smoothing_sigma", 1.0),
        )


# ---------------------------------------------------------------------------
# Individual operations
# ---------------------------------------------------------------------------

def apply_vad_mask(
    f0: np.ndarray,
    voiced_mask: np.ndarray,
    fill: UnvoicedFill = "zero",
) -> np.ndarray:
    """Zero or NaN frames that the VAD marks as unvoiced."""
    f0_out = f0.copy().astype(np.float32)
    f0_out[~voiced_mask] = 0.0 if fill == "zero" else np.nan
    return f0_out


def median_filter_pitch(
    f0: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """Apply a median filter to voiced frames only."""
    if kernel_size <= 1:
        return f0.copy()

    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    voiced = (f0 > 0) & ~np.isnan(f0)
    if not np.any(voiced):
        return f0.copy()

    filtered_full = median_filter(f0, size=k, mode="reflect")
    f0_out = f0.copy()
    f0_out[voiced] = filtered_full[voiced]
    return f0_out


def interpolate_short_gaps(
    f0: np.ndarray,
    voiced_mask: np.ndarray,
    max_gap_frames: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill short silent gaps between voiced regions using linear interpolation."""
    if max_gap_frames <= 0:
        return f0.copy(), voiced_mask.copy()

    f0_out = f0.copy().astype(np.float32)
    mask_out = voiced_mask.copy()
    T = len(f0)
    i = 0

    while i < T:
        if mask_out[i]:
            i += 1
            continue

        j = i
        while j < T and not mask_out[j]:
            j += 1

        gap_len = j - i

        if gap_len <= max_gap_frames and i > 0 and j < T:
            left_f0 = f0_out[i - 1]
            right_f0 = f0_out[j]

            if left_f0 > 0 and right_f0 > 0:
                fill_vals = np.linspace(left_f0, right_f0, gap_len + 2)[1:-1]
                f0_out[i:j] = fill_vals
                mask_out[i:j] = True
                logger.debug(
                    f"[Fusion] Filled {gap_len}-frame gap at [{i}:{j}] "
                    f"({left_f0:.1f}→{right_f0:.1f} Hz)"
                )

        i = j

    return f0_out, mask_out


def smooth_pitch_contour(
    f0: np.ndarray,
    voiced_mask: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Apply Gaussian smoothing to voiced regions of the pitch contour."""
    if sigma <= 0.0:
        return f0.copy()

    try:
        from scipy.ndimage import gaussian_filter1d  # type: ignore
    except ImportError:
        logger.warning("[Fusion] scipy not available — Gaussian smoothing skipped.")
        return f0.copy()

    voiced = voiced_mask & (f0 > 0)
    if not np.any(voiced):
        return f0.copy()

    smoothed_full = gaussian_filter1d(f0.astype(np.float64), sigma=sigma)
    f0_out = f0.copy()
    f0_out[voiced] = smoothed_full[voiced].astype(np.float32)
    return f0_out


# ---------------------------------------------------------------------------
# Full fusion pipeline
# ---------------------------------------------------------------------------

def fuse_vad_and_pitch(
    f0_raw: np.ndarray,
    pitch_voiced: np.ndarray,
    vad_mask_aligned: np.ndarray,
    config: Optional[FusionConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the complete VAD–pitch fusion and contour cleaning pipeline.

    Single entry point called by the pitch inference pipeline after alignment.
    Produces the final F0 array and voiced mask written to pitch_data.json.

    Args:
        f0_raw:           Raw F0 from the pitch model, shape (T,) Hz.
        pitch_voiced:     Boolean voiced flags from pitch model, shape (T,).
        vad_mask_aligned: VAD mask resampled to pitch frame rate, shape (T,).
        config:           FusionConfig. Uses defaults if None.

    Returns:
        f0_clean:     Cleaned F0 array, shape (T,). Unvoiced = 0.0 (or NaN).
        voiced_final: Final voiced boolean mask, shape (T,).
    """
    cfg = config or FusionConfig()

    voiced_final = pitch_voiced & vad_mask_aligned
    logger.info(
        f"[Fusion] Voiced frames → "
        f"pitch model: {int(np.sum(pitch_voiced))} | "
        f"VAD: {int(np.sum(vad_mask_aligned))} | "
        f"intersection: {int(np.sum(voiced_final))}"
    )

    f0_masked = apply_vad_mask(f0_raw, voiced_final, fill=cfg.unvoiced_fill)
    f0_filled, voiced_final = interpolate_short_gaps(
        f0_masked, voiced_final, max_gap_frames=cfg.max_gap_fill_frames
    )
    f0_filtered = median_filter_pitch(f0_filled, kernel_size=cfg.median_filter_size)
    f0_clean = smooth_pitch_contour(f0_filtered, voiced_final, sigma=cfg.smoothing_sigma)

    logger.info(f"[Fusion] Done — {int(np.sum(voiced_final))} voiced frames after cleaning")
    return f0_clean, voiced_final
