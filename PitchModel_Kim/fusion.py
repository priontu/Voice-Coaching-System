"""
fusion.py - Voiced/unvoiced fusion and pitch contour cleaning.

Combines the binary VAD mask with raw pitch predictions to produce a clean,
smooth F0 contour suitable for frame-level and note-level scoring.

Pipeline (in order):
  1. Intersect pitch-model voiced flags with the VAD mask
  2. Zero or NaN unvoiced frames
  3. Interpolate short unvoiced gaps (micro-pauses between sustained notes)
  4. Median filter to remove pitch outliers
  5. Optional Gaussian smoothing for a final smooth contour
"""

import logging
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

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
    """
    How to represent unvoiced frames after masking.
      "zero" : f0[t] = 0.0   — compatible with pitch_score.py's voiced_mask=(f0>0)
      "nan"  : f0[t] = NaN   — useful for plotting gaps in matplotlib
    Use "zero" for everything that feeds into pitch_score.py.
    """

    median_filter_size: int = 5
    """
    Kernel size for voiced-only median filtering.
    Removes impulsive pitch errors (outlier frames) without blurring transitions.
    Set to 0 or 1 to disable.
    """

    max_gap_fill_frames: int = 10
    """
    Maximum number of consecutive unvoiced frames to fill by linear
    interpolation. This bridges short micro-pauses within a sustained note
    that VAD mistakenly marks as unvoiced.
    Set to 0 to disable gap filling.
    """

    smoothing_sigma: float = 1.0
    """
    Gaussian smoothing applied to voiced regions after median filtering.
    Standard deviation in frames. Set to 0.0 to disable.
    """


# ---------------------------------------------------------------------------
# Individual operations
# ---------------------------------------------------------------------------

def apply_vad_mask(
    f0: np.ndarray,
    voiced_mask: np.ndarray,
    fill: UnvoicedFill = "zero",
) -> np.ndarray:
    """
    Zero or NaN frames that the VAD marks as unvoiced.

    Args:
        f0: Raw F0 predictions in Hz, shape (T,).
        voiced_mask: Boolean mask — True = voiced, shape (T,).
        fill: "zero" sets unvoiced to 0.0; "nan" sets to NaN.

    Returns:
        Masked F0 array, shape (T,).
    """
    f0_out = f0.copy().astype(np.float32)
    f0_out[~voiced_mask] = 0.0 if fill == "zero" else np.nan
    return f0_out


def median_filter_pitch(
    f0: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Apply a median filter to voiced frames only, leaving unvoiced frames at 0.

    Operating only on voiced samples prevents the filter from pulling voiced
    estimates toward zero at voiced/unvoiced boundaries.

    Args:
        f0: Pitch in Hz, shape (T,). Unvoiced frames should be 0.0 or NaN.
        kernel_size: Median kernel width. Forced to an odd number ≥ 1.

    Returns:
        Filtered pitch array, shape (T,).
    """
    if kernel_size <= 1:
        return f0.copy()

    # Enforce odd kernel
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    voiced = (f0 > 0) & ~np.isnan(f0)
    if not np.any(voiced):
        return f0.copy()

    # Median filter the entire array (reflect padding at edges), then keep
    # the filtered values only where the frame was originally voiced.
    filtered_full = median_filter(f0, size=k, mode="reflect")
    f0_out = f0.copy()
    f0_out[voiced] = filtered_full[voiced]
    return f0_out


def interpolate_short_gaps(
    f0: np.ndarray,
    voiced_mask: np.ndarray,
    max_gap_frames: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fill short silent gaps between voiced regions using linear interpolation.

    Singing often contains micro-pauses (e.g. between syllables) that VAD
    incorrectly classifies as unvoiced. Bridging these gaps prevents spurious
    pitch-zero events that penalize note-level scoring.

    Only gaps that have valid voiced F0 on BOTH sides are interpolated.
    Long gaps (background silence, breath pauses) are left untouched.

    Args:
        f0: Pitch in Hz, shape (T,). Unvoiced frames are 0.0.
        voiced_mask: Boolean voiced mask, shape (T,).
        max_gap_frames: Maximum unvoiced run length to interpolate over.

    Returns:
        f0_filled: Updated F0 array with gaps bridged.
        mask_filled: Updated voiced mask (gap frames now marked voiced).
    """
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

        # Found start of an unvoiced run — find its end
        j = i
        while j < T and not mask_out[j]:
            j += 1

        gap_len = j - i

        # Only fill short interior gaps with voiced context on both sides
        if gap_len <= max_gap_frames and i > 0 and j < T:
            left_f0 = f0_out[i - 1]
            right_f0 = f0_out[j]

            if left_f0 > 0 and right_f0 > 0:
                # Linear interpolation between left and right voiced endpoints
                fill_vals = np.linspace(left_f0, right_f0, gap_len + 2)[1:-1]
                f0_out[i:j] = fill_vals
                mask_out[i:j] = True

                logger.debug(
                    f"[Fusion] Filled {gap_len}-frame gap at frames [{i}:{j}] "
                    f"({left_f0:.1f}→{right_f0:.1f} Hz)"
                )

        i = j

    return f0_out, mask_out


def smooth_pitch_contour(
    f0: np.ndarray,
    voiced_mask: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Apply Gaussian smoothing to voiced regions of the pitch contour.

    Smoothing is only applied to voiced frames to preserve sharp
    voiced/unvoiced transitions.

    Args:
        f0: Pitch in Hz, shape (T,).
        voiced_mask: Boolean voiced mask, shape (T,).
        sigma: Gaussian std in frames. Set to 0.0 to disable.

    Returns:
        Smoothed pitch array, shape (T,).
    """
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

    # Run the filter over the whole array (zero padding is fine because we
    # only copy back the voiced portions anyway)
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

    This is the single entry point called by inference.py after alignment.
    It produces the final F0 array and voiced mask that are written to
    pitch_data.json for consumption by pitch_score.py.

    Args:
        f0_raw: Raw F0 predictions from the pitch model, shape (T,) Hz.
                Unvoiced frames from the model are already 0.0.
        pitch_voiced: Boolean voiced flags from the pitch model, shape (T,).
                      Typically (f0_raw > 0).
        vad_mask_aligned: VAD voiced mask resampled to pitch frame rate, shape (T,).
        config: FusionConfig instance. Uses defaults if None.

    Returns:
        f0_clean: Cleaned F0 array, shape (T,). Unvoiced = 0.0 (or NaN).
        voiced_final: Final voiced boolean mask, shape (T,).
    """
    cfg = config or FusionConfig()

    # Step 1 — Intersection: only frames voiced by BOTH pitch model and VAD
    voiced_final = pitch_voiced & vad_mask_aligned
    logger.info(
        f"[Fusion] Voiced frames → "
        f"pitch model: {int(np.sum(pitch_voiced))} | "
        f"VAD: {int(np.sum(vad_mask_aligned))} | "
        f"intersection: {int(np.sum(voiced_final))}"
    )

    # Step 2 — Apply mask (zero/NaN unvoiced)
    f0_masked = apply_vad_mask(f0_raw, voiced_final, fill=cfg.unvoiced_fill)

    # Step 3 — Fill short gaps
    f0_filled, voiced_final = interpolate_short_gaps(
        f0_masked, voiced_final, max_gap_frames=cfg.max_gap_fill_frames
    )

    # Step 4 — Median filtering
    f0_filtered = median_filter_pitch(f0_filled, kernel_size=cfg.median_filter_size)

    # Step 5 — Gaussian smoothing
    f0_clean = smooth_pitch_contour(
        f0_filtered, voiced_final, sigma=cfg.smoothing_sigma
    )

    logger.info(
        f"[Fusion] Done — {int(np.sum(voiced_final))} voiced frames after cleaning"
    )
    return f0_clean, voiced_final
