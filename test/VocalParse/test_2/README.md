# VocalParse Test 2

Experimental pseudo-onset/offset test.

VocalParse does not output physical onset/offset timestamps. This test derives
approximate note boundaries by:

```text
<BPM_*> + <NOTE_*> symbolic note values
-> cumulative note durations in seconds
-> pseudo onset/offset timestamps
```

Then it compares those pseudo boundaries against MusicXML note start/end times.

This is useful only as a rough baseline. It is not a replacement for a real
audio onset/offset model.

Run:

```bash
python test/VocalParse/test_2/run_test.py \
  --dataset-root /mnt/archive/GTSinger/English/EN-Alto-1 \
  --limit 2
```

Outputs:

```text
test/VocalParse/test_2/outputs/
  manifest.json
  summary.json
  raw_outputs/*.txt
  note_events/*.csv
```

