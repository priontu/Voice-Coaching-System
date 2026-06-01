# VocalParse Integration

This folder wraps the Hugging Face model `pymaster/VocalParse`.

According to the model card, VocalParse is a singing voice transcription model
fine-tuned from Qwen3-ASR-1.7B. It transcribes mono 16 kHz singing audio into
an autoregressive token sequence containing:

```text
lyrics
pitch tokens: <P_0> ... <P_127>  # MIDI notes
note value tokens: <NOTE_4>, <NOTE_8>, ...
BPM token: <BPM_89>
```

It does **not** directly produce physical onset and offset timestamps. The
model card explicitly states that physical note durations are not predicted by
this checkpoint. The note tokens are symbolic note values, not second-level
start/end times.

Install:

```bash
conda activate torch_it
pip install git+https://github.com/pymaster17/VocalParse.git
```

Usage:

```python
from VocalParse import VocalParseTranscriber

transcriber = VocalParseTranscriber()
result = transcriber.transcribe("song.wav")
print(result.pitch_tokens)
print(result.note_tokens)
print(result.bpm)
```

