"""
GTSinger Evaluation Script

Complete workflow for evaluating phoneme boundaries against GTSinger ground truth.

Usage:
    # Evaluate first 10 files
    py evaluate_gtsinger_complete.py --gtsinger-dir ./gtsinger --max-files 10
    
    # Full evaluation with GPU
    py evaluate_gtsinger_complete.py --gtsinger-dir ./gtsinger --device cuda
    
    # CPU evaluation with detailed output
    py evaluate_gtsinger_complete.py --gtsinger-dir ./gtsinger --device cpu --verbose
"""

import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import argparse

import numpy as np
from phoneme_model import (
    extract_phoneme_boundaries_from_audio,
    compute_boundary_metrics,
    PhonemeBoundaryConfig,
    PhonemeSegment,
)

try:
    import textgrid
except ImportError:
    print("Error: textgrid not installed")
    print("Install with: pip install textgrid")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# TEXTGRID PARSING
# ============================================================================

def load_gtsinger_annotations(textgrid_path: str) -> List[PhonemeSegment]:
    """
    Load phoneme boundaries from GTSinger TextGrid file.
    
    Args:
        textgrid_path: Path to .TextGrid file
    
    Returns:
        List of PhonemeSegment objects
    """
    try:
        tg = textgrid.TextGrid.fromFile(textgrid_path)
        # GTSinger has a "phone" tier for phoneme annotations
        tier = tg.getFirst('phone')
        
        segments = []
        for interval in tier:
            # Skip silence and empty intervals
            if not interval.mark or interval.mark in ['sil', 'pau', '', '<SP>']:
                continue
            
            segment = PhonemeSegment(
                phoneme=interval.mark,
                start_time=interval.minTime,
                end_time=interval.maxTime,
                confidence=1.0
            )
            segments.append(segment)
        
        return segments
    
    except Exception as e:
        logger.error(f"Failed to load TextGrid {textgrid_path}: {str(e)}")
        return []


# ============================================================================
# DATA DISCOVERY
# ============================================================================

def find_gtsinger_pairs(gtsinger_dir: str) -> List[Tuple[str, str]]:
    """
    Find matching audio-TextGrid pairs in GTSinger.
    
    Returns:
        List of (audio_path, textgrid_path) tuples
    """
    gtsinger_path = Path(gtsinger_dir)
    pairs = []
    
    for tg_file in sorted(gtsinger_path.glob("**/*.TextGrid")):
        audio_file = tg_file.with_suffix(".wav")
        
        if audio_file.exists():
            pairs.append((str(audio_file), str(tg_file)))
    
    return pairs


# ============================================================================
# SINGLE FILE EVALUATION
# ============================================================================

def evaluate_single_file(
    audio_path: str,
    textgrid_path: str,
    tolerance_ms: float = 50.0,
    config: Optional[PhonemeBoundaryConfig] = None,
    verbose: bool = False
) -> Optional[Dict]:
    """
    Evaluate predictions against GTSinger ground truth for a single file.
    
    Args:
        audio_path: Path to WAV file
        textgrid_path: Path to TextGrid annotation
        tolerance_ms: Boundary matching tolerance
        config: Optional custom configuration
        verbose: Print detailed output
    
    Returns:
        Dictionary with metrics and results, or None on error
    """
    if config is None:
        config = PhonemeBoundaryConfig(device="cuda")
    
    try:
        # Load ground truth
        ref_segments = load_gtsinger_annotations(textgrid_path)
        if not ref_segments:
            logger.warning(f"No reference annotations in {textgrid_path}")
            return None
        
        # Extract predictions
        result = extract_phoneme_boundaries_from_audio(
            audio_path,
            config=config,
            return_segments=True
        )
        pred_segments = result["segments"]
        
        # Compute metrics
        metrics = compute_boundary_metrics(
            pred_segments,
            ref_segments,
            tolerance_ms=tolerance_ms
        )
        
        # Build result dictionary
        file_result = {
            "audio_file": Path(audio_path).name,
            "duration_s": round(result["metadata"]["duration_s"], 2),
            "num_predictions": len(pred_segments),
            "num_references": len(ref_segments),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        if verbose:
            print(f"\n  Predictions: {len(pred_segments)}")
            print(f"  References:  {len(ref_segments)}")
            print(f"  Precision:   {metrics['precision']:.4f}")
            print(f"  Recall:      {metrics['recall']:.4f}")
            print(f"  F1-Score:    {metrics['f1']:.4f}")
            print(f"  MAE:         {metrics['mae_ms']:.2f}ms")
        
        return file_result
    
    except Exception as e:
        logger.error(f"Error evaluating {Path(audio_path).name}: {str(e)}")
        return None


# ============================================================================
# BATCH EVALUATION
# ============================================================================

def evaluate_batch(
    gtsinger_dir: str,
    tolerance_ms: float = 50.0,
    device: str = "cuda",
    max_files: Optional[int] = None,
    output_json: str = "gtsinger_evaluation_results.json",
    verbose: bool = False
) -> Dict:
    """
    Evaluate all files in GTSinger dataset.
    
    Args:
        gtsinger_dir: Path to GTSinger root directory
        tolerance_ms: Boundary matching tolerance
        device: "cuda" or "cpu"
        max_files: Limit number of files (for testing)
        output_json: Output results file
        verbose: Print detailed output
    
    Returns:
        Dictionary with all results
    """
    config = PhonemeBoundaryConfig(device=device)
    pairs = find_gtsinger_pairs(gtsinger_dir)
    
    if max_files:
        pairs = pairs[:max_files]
    
    print(f"\n{'='*70}")
    print("GTSinger BATCH EVALUATION")
    print(f"{'='*70}")
    print(f"Total files:  {len(pairs)}")
    print(f"Device:       {device}")
    print(f"Tolerance:    {tolerance_ms}ms")
    print(f"Output:       {output_json}")
    print(f"{'='*70}\n")
    
    # Initialize results
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "gtsinger_dir": gtsinger_dir,
            "tolerance_ms": tolerance_ms,
            "device": device,
            "total_pairs": len(pairs),
            "max_files": max_files
        },
        "files": [],
        "aggregate": {
            "successful": 0,
            "failed": 0,
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "avg_f1": 0.0,
            "avg_mae_ms": 0.0,
            "median_f1": 0.0,
            "std_f1": 0.0
        }
    }
    
    # Process each pair
    precisions = []
    recalls = []
    f1_scores = []
    maes = []
    
    for i, (audio_path, tg_path) in enumerate(pairs, 1):
        file_name = Path(audio_path).name
        print(f"[{i:3d}/{len(pairs)}] {file_name:<25}", end=" ", flush=True)
        
        file_result = evaluate_single_file(
            audio_path,
            tg_path,
            tolerance_ms=tolerance_ms,
            config=config,
            verbose=verbose
        )
        
        if file_result is not None:
            results["files"].append(file_result)
            metrics = file_result["metrics"]
            
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])
            f1_scores.append(metrics["f1"])
            maes.append(metrics["mae_ms"])
            
            results["aggregate"]["successful"] += 1
            
            # Status indicator (ASCII-safe for Windows CP1252 console)
            f1 = metrics["f1"]
            if f1 > 0.85:
                status = "[+++]"
            elif f1 > 0.75:
                status = "[++ ]"
            elif f1 > 0.65:
                status = "[+  ]"
            elif f1 > 0.5:
                status = "[~  ]"
            else:
                status = "[---]"

            print(f"{status} F1={f1:.4f}")
        else:
            results["aggregate"]["failed"] += 1
            print("ERROR")
    
    # Compute aggregate metrics
    if f1_scores:
        results["aggregate"]["avg_precision"] = round(np.mean(precisions), 4)
        results["aggregate"]["avg_recall"] = round(np.mean(recalls), 4)
        results["aggregate"]["avg_f1"] = round(np.mean(f1_scores), 4)
        results["aggregate"]["avg_mae_ms"] = round(np.mean(maes), 2)
        results["aggregate"]["median_f1"] = round(np.median(f1_scores), 4)
        results["aggregate"]["std_f1"] = round(np.std(f1_scores), 4)
        results["aggregate"]["min_f1"] = round(np.min(f1_scores), 4)
        results["aggregate"]["max_f1"] = round(np.max(f1_scores), 4)
    else:
        # No successful evaluations - set defaults
        results["aggregate"]["avg_precision"] = 0.0
        results["aggregate"]["avg_recall"] = 0.0
        results["aggregate"]["avg_f1"] = 0.0
        results["aggregate"]["avg_mae_ms"] = 0.0
        results["aggregate"]["median_f1"] = 0.0
        results["aggregate"]["std_f1"] = 0.0
        results["aggregate"]["min_f1"] = 0.0
        results["aggregate"]["max_f1"] = 0.0
    
    # Print summary
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Successful:   {results['aggregate']['successful']}/{len(pairs)}")
    print(f"Failed:       {results['aggregate']['failed']}/{len(pairs)}")
    print(f"\nAggregate Metrics:")
    print(f"  Avg Precision: {results['aggregate']['avg_precision']:.4f}")
    print(f"  Avg Recall:    {results['aggregate']['avg_recall']:.4f}")
    print(f"  Avg F1-Score:  {results['aggregate']['avg_f1']:.4f}")
    print(f"  Median F1:     {results['aggregate']['median_f1']:.4f}")
    print(f"  Std F1:        {results['aggregate']['std_f1']:.4f}")
    print(f"  Min F1:        {results['aggregate']['min_f1']:.4f}")
    print(f"  Max F1:        {results['aggregate']['max_f1']:.4f}")
    print(f"  Avg MAE:       {results['aggregate']['avg_mae_ms']:.2f}ms")
    print(f"{'='*70}\n")
    
    # Save results
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Results saved to {output_json}\n")
    
    return results


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_results(results_json: str, output_dir: str = "."):
    """
    Analyze and visualize evaluation results.
    """
    with open(results_json) as f:
        results = json.load(f)
    
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        print("Pandas and matplotlib required for analysis")
        print("Install with: pip install pandas matplotlib")
        return
    
    # Extract metrics
    files = results["files"]
    metrics_list = [f["metrics"] for f in files if "metrics" in f]
    
    if not metrics_list:
        print("No metrics to analyze")
        return
    
    # Create DataFrame
    df = pd.DataFrame({
        "file": [f["audio_file"] for f in files if "metrics" in f],
        "precision": [m["precision"] for m in metrics_list],
        "recall": [m["recall"] for m in metrics_list],
        "f1": [m["f1"] for m in metrics_list],
        "mae_ms": [m["mae_ms"] for m in metrics_list]
    })
    
    print("\n" + "="*70)
    print("DETAILED RESULTS")
    print("="*70)
    print(df.to_string(index=False))
    
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(df[["precision", "recall", "f1", "mae_ms"]].describe())
    
    # Save to CSV
    csv_file = Path(output_dir) / "gtsinger_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n[OK] Results saved to {csv_file}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GTSinger Evaluation Results", fontsize=16, fontweight='bold')
    
    axes[0, 0].hist(df["precision"], bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].set_title("Precision Distribution")
    axes[0, 0].set_xlabel("Precision")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].axvline(df["precision"].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {df["precision"].mean():.3f}')
    axes[0, 0].legend()
    
    axes[0, 1].hist(df["recall"], bins=15, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_title("Recall Distribution")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].axvline(df["recall"].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {df["recall"].mean():.3f}')
    axes[0, 1].legend()
    
    axes[1, 0].hist(df["f1"], bins=15, edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].set_title("F1-Score Distribution")
    axes[1, 0].set_xlabel("F1-Score")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].axvline(df["f1"].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {df["f1"].mean():.3f}')
    axes[1, 0].legend()
    
    axes[1, 1].hist(df["mae_ms"], bins=15, edgecolor='black', alpha=0.7, color='coral')
    axes[1, 1].set_title("MAE Distribution")
    axes[1, 1].set_xlabel("MAE (milliseconds)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].axvline(df["mae_ms"].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {df["mae_ms"].mean():.1f}ms')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    plot_file = Path(output_dir) / "gtsinger_results_analysis.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"[OK] Visualization saved to {plot_file}")
    
    plt.close()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate phoneme boundaries against GTSinger ground truth"
    )
    parser.add_argument(
        "--gtsinger-dir",
        required=True,
        help="Path to GTSinger dataset directory"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to evaluate (for testing)"
    )
    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=50.0,
        help="Boundary matching tolerance in milliseconds"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use for inference"
    )
    parser.add_argument(
        "--output",
        default="gtsinger_evaluation_results.json",
        help="Output results file (JSON)"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze results after evaluation"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_batch(
        gtsinger_dir=args.gtsinger_dir,
        tolerance_ms=args.tolerance_ms,
        device=args.device,
        max_files=args.max_files,
        output_json=args.output,
        verbose=args.verbose
    )
    
    # Analyze if requested
    if args.analyze:
        print("\nAnalyzing results...")
        analyze_results(args.output)


if __name__ == "__main__":
    main()
