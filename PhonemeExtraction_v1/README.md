# PhonemeExtraction_v1

Hugging Face phoneme extraction baseline using:

```text
facebook/wav2vec2-lv-60-espeak-cv-ft
```

The model predicts a CTC phoneme sequence. This package collapses frame-level
CTC predictions into rough phoneme intervals:

Install dependencies:

```bash
conda activate torch_it
pip install phonemizer
conda install -c conda-forge espeak-ng
```

```python
from PhonemeExtraction_v1 import HFPhonemeExtractor

extractor = HFPhonemeExtractor(device="cuda")
prediction = extractor.predict("example.wav")
```

Output:

```python
{
  "transcript": "...",
  "phonemes": [
    {
      "phoneme": "...",
      "start_sec": 0.12,
      "end_sec": 0.18,
      "confidence": 0.82
    }
  ]
}
```

Important: these are rough CTC-derived boundaries, not forced-alignment-quality
phoneme timestamps.
