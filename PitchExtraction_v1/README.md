# PitchExtraction_v1

Runtime pitch extraction package for the voice-coaching system.

This folder contains:

```text
PitchExtraction_v1/
  weights.pth                 # Priontu_Chowdhury NanoPitch weights
  nanopitch_runtime.py         # GPU-oriented NanoPitch batch extractor
  batch_converter.py           # integrates pitch extraction with runtime preprocessing
  datamodule.py                # LightningDataModule wrapper
  run_pitch_extraction.py      # smoke-test CLI
```

The converter reuses the existing runtime preprocessing code in `src/voice_coach`:

```text
raw WAV + MusicXML + TextGrid
-> VoiceCoachDeviceBatchConverter
   -> input_features
   -> labels: f0, voicing, onset, offset, phoneme_boundary
-> NanoPitchBatchExtractor
   -> nanopitch: f0_hz, voiced, vad_prob, pitch_confidence, posteriorgram
-> aligned labels:
   -> labels.nanopitch_f0
   -> labels.nanopitch_voicing
```

Run a smoke test:

```bash
/home/DREXEL/pc833/miniconda3/envs/torch_it/bin/python PitchExtraction_v1/run_pitch_extraction.py \
  --manifest path/to/manifest.json \
  --device cuda
```

Swap weights without changing code:

```python
converter = PitchExtractionBatchConverter(device="cuda")
converter.load_weights("path/to/another/weights.pth")
```

Use with Lightning:

```python
from pathlib import Path
from PitchExtraction_v1 import PitchExtractionDataConfig, PitchExtractionDataModule

dm = PitchExtractionDataModule(
    PitchExtractionDataConfig(train_manifest=Path("manifest.json"))
)
trainer.fit(model, datamodule=dm)
```

NanoPitch uses its own feature settings:

```text
sample_rate = 16000
n_mels = 40
win_length = 400
hop_length = 160
```

The main training features can stay at the voice-coach defaults, currently
48 kHz / 80-mel. NanoPitch outputs are attached separately and also aligned
to the main feature frame count as `labels.nanopitch_f0` and
`labels.nanopitch_voicing`.
