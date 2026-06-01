# VocalParse Test 1

Evaluates VocalParse on GTSinger `Control_Group` recordings.

This test compares VocalParse's symbolic outputs against MusicXML:

```text
VocalParse pitch tokens <P_*>  -> MusicXML MIDI notes
VocalParse BPM token <BPM_*>   -> MusicXML tempo
```

It does not evaluate onset/offset because VocalParse does not output physical
note start/end timestamps.

Install first:

```bash
conda activate torch_it
pip install git+https://github.com/pymaster17/VocalParse.git
```

Run:

```bash
python test/VocalParse/test_1/run_test.py \
  --dataset-root /mnt/archive/GTSinger/English/EN-Alto-1 \
  --limit 2
```

Outputs:

```text
test/VocalParse/test_1/outputs/
  manifest.json
  summary.json
  raw_outputs/*.txt
```

