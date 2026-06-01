# PitchExtraction_RMVPE_v1 Test 1

Evaluates RMVPE on GTSinger `Control_Group` recordings.

Install dependency in `torch_it`:

```bash
pip install rmvpe-onnx
```

Run:

```bash
python test/PitchExtraction_RMVPE_v1/test_1/run_test.py \
  --dataset-root /mnt/archive/GTSinger/English/EN-Alto-1 \
  --limit 5 \
  --device cuda
```

Optional ONNX Runtime CUDA provider:

```bash
python test/PitchExtraction_RMVPE_v1/test_1/run_test.py \
  --dataset-root /mnt/archive/GTSinger/English/EN-Alto-1 \
  --limit 5 \
  --device cuda \
  --providers CUDAExecutionProvider CPUExecutionProvider
```

The first run may download the RMVPE ONNX model through `rmvpe-onnx`.
