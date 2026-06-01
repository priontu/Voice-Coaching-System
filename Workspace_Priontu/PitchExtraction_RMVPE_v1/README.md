# PitchExtraction_RMVPE_v1

RMVPE pitch extraction baseline for the voice coaching system.

This module uses the `rmvpe-onnx` package, which downloads/loads the Hugging Face
RMVPE ONNX checkpoint and runs inference with ONNX Runtime.

Install in `torch_it`:

```bash
pip install rmvpe-onnx
```

For GPU acceleration through ONNX Runtime, install a CUDA-compatible ONNX Runtime
build in the same environment:

```bash
pip install onnxruntime-gpu
```

The runtime outputs padded tensors:

- `rmvpe.f0_hz`
- `rmvpe.voiced`
- `rmvpe.confidence`
- `rmvpe.times`
- `rmvpe.attention_mask`
- `labels.rmvpe_ref_f0`
- `labels.rmvpe_ref_voicing`

The reference labels are sampled at RMVPE's returned frame times, so evaluation
does not resample RMVPE onto the training feature grid.

The default runtime confidence threshold is `0.35`, matching the best setting
from the current GTSinger Control_Group comparison. `RMVPEBatchConverter` also
adds `rmvpe_recommended`, a per-recording list of stable-note median pitch
events intended for note-level coaching feedback.
