# PhonemeExtraction_v1 Test 1

Runs the Hugging Face Wav2Vec2 phoneme model on GTSinger Control_Group files
and compares rough CTC phoneme boundaries against TextGrid phoneme intervals.

Run:

```bash
conda activate torch_it
pip install phonemizer
conda install -c conda-forge espeak-ng

python test/PhonemeExtraction_v1/test_1/run_test.py \
  --dataset-root /mnt/archive/GTSinger/English/EN-Alto-1 \
  --limit 2 \
  --device cuda
```

Outputs:

```text
test/PhonemeExtraction_v1/test_1/outputs/
  manifest.json
  summary.json
  predictions/*.json
  matches/*.csv
```
