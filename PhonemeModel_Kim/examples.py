"""
Example usage scripts for Phoneme Boundary Detection Module

Demonstrates:
1. Basic usage with default configuration
2. Advanced configuration with custom parameters
3. Batch processing multiple files
4. Evaluation against ground truth
5. Visualization and analysis
6. Integration patterns for singing voice evaluation
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch

from phoneme_model import (
    extract_phoneme_boundaries_from_audio,
    load_audio,
    load_model,
    plot_phoneme_boundaries,
    compute_boundary_metrics,
    PhonemeBoundaryConfig,
    PhonemeSegment,
)

logger = logging.getLogger(__name__)


# ============================================================================
# EXAMPLE 1: Basic Usage
# ============================================================================

def example_basic_usage():
    """
    Simplest possible usage - extract phoneme boundaries from a WAV file.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    
    audio_file = "sample_audio.wav"  # Replace with actual file
    
    # One-liner to extract boundaries
    result = extract_phoneme_boundaries_from_audio(audio_file)
    
    # Print results
    print(f"\nExtracted {len(result['segments'])} phonemes:")
    print(f"Duration: {result['metadata']['duration_s']:.2f} seconds")
    
    # Display first 10 phonemes
    print("\nFirst 10 phonemes:")
    for seg in result["segments"][:10]:
        duration_ms = (seg.end_time - seg.start_time) * 1000
        print(f"  {seg.phoneme:4s} [{seg.start_time:.3f}s - {seg.end_time:.3f}s] "
              f"({duration_ms:.1f}ms)")
    
    # Display phoneme sequence
    phoneme_seq = " ".join(result["phonemes"])
    print(f"\nPhoneme sequence: {phoneme_seq[:80]}...")
    
    return result


# ============================================================================
# EXAMPLE 2: Advanced Configuration
# ============================================================================

def example_advanced_config():
    """
    Demonstration of fine-tuned configuration for optimal results.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Advanced Configuration")
    print("="*70)
    
    audio_file = "sample_audio.wav"  # Replace with actual file
    
    # Custom configuration
    config = PhonemeBoundaryConfig(
        model_name="facebook/wav2vec2-lv-60-espeak-cv-ft",
        sample_rate=16000,
        device="cuda" if torch.cuda.is_available() else "cpu",
        collapse_repeated_tokens=True,  # Collapse consecutive identical phonemes
        remove_blank_tokens=True,        # Remove silence/padding tokens
        blank_token_id=0
    )
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Device: {config.device}")
    print(f"  Sample rate: {config.sample_rate} Hz")
    print(f"  Collapse repeats: {config.collapse_repeated_tokens}")
    print(f"  Remove blanks: {config.remove_blank_tokens}")
    
    # Extract with custom config
    result = extract_phoneme_boundaries_from_audio(
        audio_file,
        config=config,
        return_segments=True,
        word_grouping=False
    )
    
    print(f"\nResults:")
    print(f"  Total phonemes: {len(result['segments'])}")
    print(f"  Coverage: {result['segments'][0].start_time:.3f}s - "
          f"{result['segments'][-1].end_time:.3f}s")
    
    # Show statistics
    durations = [
        (seg.end_time - seg.start_time) * 1000 
        for seg in result["segments"]
    ]
    print(f"  Phoneme duration (ms):")
    print(f"    Mean: {np.mean(durations):.2f}")
    print(f"    Std:  {np.std(durations):.2f}")
    print(f"    Min:  {np.min(durations):.2f}")
    print(f"    Max:  {np.max(durations):.2f}")
    
    return result


# ============================================================================
# EXAMPLE 3: Batch Processing
# ============================================================================

def example_batch_processing(audio_directory: str):
    """
    Process multiple audio files and save results.
    
    Args:
        audio_directory: Directory containing WAV files
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Batch Processing")
    print("="*70)
    
    audio_dir = Path(audio_directory)
    audio_files = list(audio_dir.glob("*.wav"))
    
    if not audio_files:
        print(f"No WAV files found in {audio_directory}")
        return
    
    print(f"\nFound {len(audio_files)} WAV files")
    
    # Configure for batch processing
    config = PhonemeBoundaryConfig(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Load model once (more efficient)
    model, processor = load_model(config)
    
    batch_results = {}
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        
        try:
            result = extract_phoneme_boundaries_from_audio(
                str(audio_file),
                config=config,
                return_segments=True
            )
            
            batch_results[audio_file.stem] = {
                "phonemes": result["phonemes"],
                "boundaries": [[b[0], b[1]] for b in result["boundaries"]],
                "metadata": result["metadata"]
            }
            
            print(f"  ✓ Extracted {len(result['segments'])} phonemes")
        
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            batch_results[audio_file.stem] = {"error": str(e)}
    
    # Save batch results
    output_file = Path("batch_results.json")
    with open(output_file, 'w') as f:
        json.dump(batch_results, f, indent=2)
    
    print(f"\n✓ Batch results saved to {output_file}")
    print(f"  Successfully processed: {len([r for r in batch_results.values() if 'error' not in r])}")
    print(f"  Failed: {len([r for r in batch_results.values() if 'error' in r])}")
    
    return batch_results


# ============================================================================
# EXAMPLE 4: Visualization
# ============================================================================

def example_visualization_and_analysis():
    """
    Extract boundaries and create visualization with analysis.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Visualization and Analysis")
    print("="*70)
    
    audio_file = "sample_audio.wav"  # Replace with actual file
    
    # Extract boundaries
    result = extract_phoneme_boundaries_from_audio(
        audio_file,
        word_grouping=True
    )
    
    segments = result["segments"]
    
    # Analysis 1: Phoneme frequency
    phoneme_counts = {}
    for seg in segments:
        phoneme_counts[seg.phoneme] = phoneme_counts.get(seg.phoneme, 0) + 1
    
    print("\nPhoneme Frequency (top 10):")
    for phoneme, count in sorted(phoneme_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        percentage = (count / len(segments)) * 100
        print(f"  {phoneme:4s}: {count:3d} times ({percentage:5.1f}%)")
    
    # Analysis 2: Duration statistics by phoneme
    print("\nDuration Statistics (ms):")
    phoneme_durations = {}
    for seg in segments:
        duration = (seg.end_time - seg.start_time) * 1000
        if seg.phoneme not in phoneme_durations:
            phoneme_durations[seg.phoneme] = []
        phoneme_durations[seg.phoneme].append(duration)
    
    print(f"{'Phoneme':<6} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-" * 38)
    for phoneme in sorted(phoneme_durations.keys())[:10]:
        durations = phoneme_durations[phoneme]
        print(f"{phoneme:<6} {np.mean(durations):<8.2f} {np.std(durations):<8.2f} "
              f"{np.min(durations):<8.2f} {np.max(durations):<8.2f}")
    
    # Analysis 3: Word-level statistics
    if "words" in result:
        print(f"\nWord-Level Analysis ({len(result['words'])} words):")
        word_durations = [
            (w["end_time"] - w["start_time"]) * 1000
            for w in result["words"]
        ]
        print(f"  Word duration (ms):")
        print(f"    Mean: {np.mean(word_durations):.2f}")
        print(f"    Std:  {np.std(word_durations):.2f}")
        print(f"    Min:  {np.min(word_durations):.2f}")
        print(f"    Max:  {np.max(word_durations):.2f}")
        
        # Phonemes per word
        phonemes_per_word = [len(w["phonemes"]) for w in result["words"]]
        print(f"  Phonemes per word:")
        print(f"    Mean: {np.mean(phonemes_per_word):.2f}")
        print(f"    Range: {np.min(phonemes_per_word)} - {np.max(phonemes_per_word)}")
    
    # Visualization
    try:
        audio, sr = load_audio(audio_file)
        plot_phoneme_boundaries(
            audio,
            segments,
            sample_rate=sr,
            save_path="phoneme_boundaries_plot.png",
            figsize=(20, 8)
        )
        print("\n✓ Visualization saved to 'phoneme_boundaries_plot.png'")
    except ImportError:
        print("\nNote: Install matplotlib to enable visualization: pip install matplotlib")
    
    return result


# ============================================================================
# EXAMPLE 5: Evaluation Against Ground Truth
# ============================================================================

def example_evaluation_against_ground_truth(
    audio_file: str,
    reference_json: str,
    tolerance_ms: float = 50.0
):
    """
    Evaluate predicted boundaries against ground truth reference.
    
    Args:
        audio_file: Path to audio file
        reference_json: Path to JSON file with reference segments
        tolerance_ms: Tolerance window for boundary matching (milliseconds)
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Evaluation Against Ground Truth")
    print("="*70)
    
    # Extract predictions
    print("\nExtracting phoneme boundaries...")
    result = extract_phoneme_boundaries_from_audio(audio_file)
    pred_segments = result["segments"]
    
    # Load ground truth
    print("Loading ground truth...")
    with open(reference_json, 'r') as f:
        ref_data = json.load(f)
    
    ref_segments = []
    if "segments" in ref_data:
        ref_segments = [PhonemeSegment(**seg) for seg in ref_data["segments"]]
    
    print(f"  Predictions: {len(pred_segments)} phonemes")
    print(f"  References:  {len(ref_segments)} phonemes")
    
    # Compute metrics
    print(f"\nComputing metrics (tolerance: {tolerance_ms}ms)...")
    metrics = compute_boundary_metrics(
        pred_segments,
        ref_segments,
        tolerance_ms=tolerance_ms
    )
    
    # Display results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nBoundary Alignment Metrics:")
    print(f"  Precision:    {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:       {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:     {metrics['f1']:.4f}")
    print(f"  MAE (ms):     {metrics['mae_ms']:.2f}")
    print(f"  Matches:      {metrics['matches']}/{metrics['total_boundaries']}")
    
    # Interpret results
    print("\nInterpretation:")
    if metrics['f1'] > 0.9:
        print("  ✓ Excellent alignment with ground truth")
    elif metrics['f1'] > 0.75:
        print("  ◐ Good alignment, some boundaries differ")
    elif metrics['f1'] > 0.5:
        print("  ◑ Moderate alignment, significant differences")
    else:
        print("  ✗ Poor alignment, substantial differences")
    
    return metrics


# ============================================================================
# EXAMPLE 6: Production Integration
# ============================================================================

class SingingVoiceEvaluationPipeline:
    """
    Production-ready integration of phoneme boundary detection
    with singing voice evaluation system.
    """
    
    def __init__(self, config: PhonemeBoundaryConfig = None):
        """Initialize pipeline with configuration."""
        self.config = config or PhonemeBoundaryConfig()
        self.model, self.processor = load_model(self.config)
        
    def evaluate_singing(self, audio_file: str) -> Dict:
        """
        Complete singing voice evaluation including phoneme timing.
        
        Args:
            audio_file: Path to singing voice recording
            
        Returns:
            Dictionary with complete analysis
        """
        logger.info(f"Evaluating: {audio_file}")
        
        # Extract phoneme boundaries
        result = extract_phoneme_boundaries_from_audio(
            audio_file,
            config=self.config,
            word_grouping=True
        )
        
        segments = result["segments"]
        
        # Compute singing-specific metrics
        evaluation = {
            "phoneme_boundaries": result,
            "metrics": self._compute_singing_metrics(segments),
            "quality_flags": self._quality_assessment(segments)
        }
        
        return evaluation
    
    @staticmethod
    def _compute_singing_metrics(segments: List[PhonemeSegment]) -> Dict:
        """Compute singing voice-specific metrics."""
        if not segments:
            return {}
        
        durations = np.array([
            (seg.end_time - seg.start_time) * 1000
            for seg in segments
        ])
        
        return {
            "total_phonemes": len(segments),
            "mean_phoneme_duration_ms": float(np.mean(durations)),
            "std_phoneme_duration_ms": float(np.std(durations)),
            "min_phoneme_duration_ms": float(np.min(durations)),
            "max_phoneme_duration_ms": float(np.max(durations)),
            "total_duration_s": float(segments[-1].end_time - segments[0].start_time)
        }
    
    @staticmethod
    def _quality_assessment(segments: List[PhonemeSegment]) -> Dict:
        """Assess phoneme boundary detection quality."""
        if not segments:
            return {}
        
        durations = np.array([
            (seg.end_time - seg.start_time) * 1000
            for seg in segments
        ])
        
        flags = []
        
        # Flag very short phonemes (< 20ms)
        short_count = np.sum(durations < 20)
        if short_count > len(segments) * 0.1:
            flags.append("HIGH_SHORT_PHONEME_RATIO")
        
        # Flag very long phonemes (> 200ms)
        long_count = np.sum(durations > 200)
        if long_count > len(segments) * 0.1:
            flags.append("HIGH_LONG_PHONEME_RATIO")
        
        # Flag high variance
        if np.std(durations) > np.mean(durations):
            flags.append("HIGH_DURATION_VARIANCE")
        
        return {
            "quality_flags": flags if flags else ["NORMAL"],
            "short_phoneme_ratio": float(short_count / len(segments))
        }


def example_production_integration():
    """Demonstrate production integration with singing evaluation."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Production Integration - Singing Voice Evaluation")
    print("="*70)
    
    # Initialize pipeline
    config = PhonemeBoundaryConfig(device="cuda")
    pipeline = SingingVoiceEvaluationPipeline(config)
    
    # Evaluate singing
    audio_file = "singing_sample.wav"  # Replace with actual file
    
    try:
        evaluation = pipeline.evaluate_singing(audio_file)
        
        print(f"\nSinging Voice Evaluation Results:")
        print(f"  Total phonemes: {evaluation['metrics']['total_phonemes']}")
        print(f"  Mean duration: {evaluation['metrics']['mean_phoneme_duration_ms']:.2f}ms")
        print(f"  Total duration: {evaluation['metrics']['total_duration_s']:.2f}s")
        
        print(f"\nQuality Assessment:")
        print(f"  Status: {evaluation['quality_flags']['quality_flags']}")
        print(f"  Short phoneme ratio: {evaluation['quality_flags']['short_phoneme_ratio']:.2%}")
        
        return evaluation
    
    except FileNotFoundError as e:
        print(f"Audio file not found: {audio_file}")
        print("Please provide a valid audio file path")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# PHONEME BOUNDARY DETECTION - EXAMPLE SCRIPTS")
    print("#"*70)
    
    # Note: Uncomment examples to run them with actual audio files
    # Make sure to provide valid audio file paths
    
    # Example 1: Basic usage
    # result = example_basic_usage()
    
    # Example 2: Advanced configuration
    # result = example_advanced_config()
    
    # Example 3: Batch processing
    # batch_results = example_batch_processing("./audio_files")
    
    # Example 4: Visualization
    # result = example_visualization_and_analysis()
    
    # Example 5: Evaluation
    # metrics = example_evaluation_against_ground_truth(
    #     "audio.wav",
    #     "ground_truth.json",
    #     tolerance_ms=50.0
    # )
    
    # Example 6: Production integration
    # evaluation = example_production_integration()
    
    print("\n" + "#"*70)
    print("# See example function docstrings for usage details")
    print("#"*70)
    print("\nTo run examples:")
    print("  1. Uncomment desired example in __main__ block")
    print("  2. Provide valid audio file path")
    print("  3. Run: python examples.py")
