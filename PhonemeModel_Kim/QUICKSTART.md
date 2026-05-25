# Quick Start Guide: Phoneme Boundary Detection

Get up and running in 5 minutes.

## 1. Installation (2 minutes)

```bash
# Clone repository
git clone <your-repo>
cd phoneme-boundary-detector

# Install dependencies
pip install -r requirements.txt

# (Optional) For GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 2. Basic Usage (1 minute)

### Python Script

```python
from phoneme_boundary_detector import extract_phoneme_boundaries_from_audio

# That's it! One line to extract phoneme boundaries
result = extract_phoneme_boundaries_from_audio("your_audio.wav")

# Print results
for segment in result["segments"][:5]:
    print(f"{segment.phoneme}: {segment.start_time:.3f}s - {segment.end_time:.3f}s")
```

### Command Line

```bash
# Extract and save to JSON
python phoneme_boundary_detector.py singing.wav --output results.json

# With visualization
python phoneme_boundary_detector.py singing.wav --plot
```

## 3. Example Output

```
AA: 0.000s - 0.200s
S:  0.200s - 0.350s
UW: 0.350s - 0.500s
M:  0.500s - 0.650s
```

## 4. Next Steps

### Visualize Results
```python
from phoneme_boundary_detector import load_audio, plot_phoneme_boundaries

audio, sr = load_audio("your_audio.wav")
plot_phoneme_boundaries(audio, result["segments"], save_path="plot.png")
```

### Evaluate Against Ground Truth
```python
from phoneme_boundary_detector import compute_boundary_metrics
import json

# Load reference
with open("ground_truth.json") as f:
    ref_data = json.load(f)

# Compare
metrics = compute_boundary_metrics(result["segments"], ref_data["segments"])
print(f"F1-Score: {metrics['f1']:.4f}")
print(f"MAE: {metrics['mae_ms']:.2f}ms")
```

### Process Multiple Files
```python
from pathlib import Path

for audio_file in Path("audio_dir").glob("*.wav"):
    result = extract_phoneme_boundaries_from_audio(str(audio_file))
    print(f"{audio_file.name}: {len(result['segments'])} phonemes")
```

## 5. Common Tasks

### Task 1: Get Just Phoneme Sequence
```python
result = extract_phoneme_boundaries_from_audio("audio.wav")
phoneme_sequence = " ".join(result["phonemes"])
print(phoneme_sequence)
# Output: AA S UW M Z
```

### Task 2: Export to JSON
```python
import json

result = extract_phoneme_boundaries_from_audio("audio.wav")

# Save to JSON
with open("phonemes.json", 'w') as f:
    json.dump({
        "phonemes": result["phonemes"],
        "boundaries": [[b[0], b[1]] for b in result["boundaries"]],
        "duration": result["metadata"]["duration_s"]
    }, f, indent=2)
```

### Task 3: Get Word-Level Info
```python
result = extract_phoneme_boundaries_from_audio(
    "audio.wav",
    word_grouping=True
)

for word in result["words"]:
    print(f"{'-'.join(word['phonemes'])}: {word['start_time']:.2f}s")
```

### Task 4: Use GPU Instead of CPU
```python
from phoneme_boundary_detector import PhonemeBoundaryConfig

config = PhonemeBoundaryConfig(device="cuda")
result = extract_phoneme_boundaries_from_audio("audio.wav", config=config)
```

### Task 5: Measure Timing Accuracy
```python
from phoneme_boundary_detector import compute_boundary_metrics
import json

# Get predictions
result = extract_phoneme_boundaries_from_audio("audio.wav")
predictions = result["segments"]

# Load ground truth
with open("reference.json") as f:
    reference_data = json.load(f)
references = reference_data["segments"]

# Compare
metrics = compute_boundary_metrics(predictions, references, tolerance_ms=50)
print(f"Precision: {metrics['precision']:.1%}")
print(f"Recall:    {metrics['recall']:.1%}")
print(f"F1-Score:  {metrics['f1']:.4f}")
print(f"Error:     {metrics['mae_ms']:.2f}ms")
```

## 6. Troubleshooting

### Problem: GPU out of memory
```python
# Use CPU
config = PhonemeBoundaryConfig(device="cpu")
```

### Problem: Very slow on CPU
```bash
# Increase CPU threads
export OMP_NUM_THREADS=8
python script.py
```

### Problem: Audio file not found
```python
from pathlib import Path

# Check file exists
audio_path = Path("your_audio.wav")
if not audio_path.exists():
    print(f"File not found: {audio_path}")
else:
    result = extract_phoneme_boundaries_from_audio(str(audio_path))
```

### Problem: Matplotlib not installed
```bash
pip install matplotlib
```

## 7. Performance Tips

### For Real-Time Processing
- Use GPU (5-10x faster than CPU)
- Process fixed-size chunks
- Pre-load model once, reuse for multiple files

### For Batch Processing
- Use GPU
- Process files sequentially or in parallel
- Cache model in memory

### For Low-Latency Inference
- Use smaller model variants if available
- Process shorter audio chunks
- Use FP16 precision if GPU supports it

## 8. Output Files

### JSON Output Structure
```json
{
  "phonemes": ["AA", "S", "UW"],
  "boundaries": [[0.0, 0.2], [0.2, 0.35], [0.35, 0.5]],
  "segments": [
    {
      "phoneme": "AA",
      "start_time": 0.0,
      "end_time": 0.2,
      "confidence": 1.0,
      "frame_start": 0,
      "frame_end": 10
    }
  ],
  "metadata": {
    "audio_file": "singing.wav",
    "duration_s": 0.5,
    "num_phonemes": 3,
    "model": "facebook/wav2vec2-lv-60-espeak-cv-ft",
    "device": "cuda"
  }
}
```

### Plot Output
The visualization shows:
- **Top panel**: Audio waveform
- **Bottom panel**: Phoneme segments with labels and time boundaries

## 9. Example Workflow

Complete workflow from audio to evaluation:

```python
from phoneme_boundary_detector import (
    extract_phoneme_boundaries_from_audio,
    load_audio,
    plot_phoneme_boundaries,
    compute_boundary_metrics,
    PhonemeBoundaryConfig
)
import json

# Step 1: Configure
config = PhonemeBoundaryConfig(device="cuda")

# Step 2: Extract
audio_file = "singing_sample.wav"
result = extract_phoneme_boundaries_from_audio(audio_file, config=config)

# Step 3: Visualize
audio, sr = load_audio(audio_file)
plot_phoneme_boundaries(audio, result["segments"], save_path="plot.png")

# Step 4: Save
with open("results.json", 'w') as f:
    json.dump({
        "phonemes": result["phonemes"],
        "segments": [s.to_dict() for s in result["segments"]],
        "metadata": result["metadata"]
    }, f, indent=2)

# Step 5: Evaluate (if reference available)
with open("ground_truth.json") as f:
    ref = json.load(f)

from phoneme_boundary_detector import PhonemeSegment
ref_segments = [PhonemeSegment(**s) for s in ref["segments"]]
metrics = compute_boundary_metrics(result["segments"], ref_segments)

print(f"✓ Extracted {len(result['segments'])} phonemes")
print(f"✓ F1-Score: {metrics['f1']:.4f}")
print(f"✓ Saved: results.json, plot.png")
```

## 10. Key Features Summary

| Feature | Command | Time |
|---------|---------|------|
| Extract boundaries | 1 line | < 5s |
| Visualize | `.plot()` | < 2s |
| Evaluate metrics | `.compute_boundary_metrics()` | < 1s |
| Batch process | Loop + GPU | 1-2s per 10s audio |
| Export JSON | `.to_dict()` | < 1s |

## 11. API Cheat Sheet

```python
# Import main function
from phoneme_boundary_detector import extract_phoneme_boundaries_from_audio

# Basic extraction
result = extract_phoneme_boundaries_from_audio("audio.wav")

# Access results
phonemes = result["phonemes"]           # List of phoneme strings
boundaries = result["boundaries"]       # List of (start, end) tuples
segments = result["segments"]           # List of PhonemeSegment objects
metadata = result["metadata"]           # Processing info

# From segments, get info
seg = segments[0]
seg.phoneme                             # "AA"
seg.start_time                          # 0.0
seg.end_time                            # 0.2
seg.to_dict()                          # Serialize to dict

# Configuration
from phoneme_boundary_detector import PhonemeBoundaryConfig
config = PhonemeBoundaryConfig(device="cuda")

# Custom parameters
result = extract_phoneme_boundaries_from_audio(
    "audio.wav",
    config=config,
    return_segments=True,
    word_grouping=True
)

# Visualization
from phoneme_boundary_detector import plot_phoneme_boundaries
plot_phoneme_boundaries(audio, segments, save_path="plot.png")

# Evaluation
from phoneme_boundary_detector import compute_boundary_metrics
metrics = compute_boundary_metrics(predictions, references, tolerance_ms=50)
```

## 12. What's Next?

1. **Read the full README.md** for comprehensive documentation
2. **Check examples.py** for advanced usage patterns
3. **Run tests** with `pytest test_phoneme_detector.py -v`
4. **Integrate** with your singing voice evaluation system

---

**Need help?** See the [README.md](README.md) for detailed documentation.

**Questions?** Check the Troubleshooting section above or review [examples.py](examples.py).

**Ready to integrate?** See the production integration example in [examples.py](examples.py).
