# Voice-Coaching-System
This project is an early prototype for a musical AI coaching system.  
Right now, the code focuses on pitch extraction and pitch scoring.

The current pipeline is:

```text
test.wav
→ test_pitch.py
→ pitch_data.json
→ pitch_score.py
→ pitch metrics + visualizations
```
# Structure
```text
Musical-AI-Coach/
│
├── test_pitch.py               # Extracts pitch/F0 from test.wav
├── pitch_score.py              # Compares extracted pitch against MusicXML reference
│
├── test.wav                    # Input singing audio file
├── test.musicxml               # Reference MusicXML score
│
├── pitch_data.json             # Output from test_pitch.py
├── note_pitch_results.json     # Output from pitch_score.py
│
├── pitch_contour.png           # Pitch contour visualization
├── pitch_vs_reference.png      # Detected pitch vs MusicXML reference
├── cent_error_over_time.png    # Pitch error over time
├── pitch_accuracy_summary.png  # Metric summary chart
└── note_level_pitch_errors.png # Note-level pitch error chart
```
