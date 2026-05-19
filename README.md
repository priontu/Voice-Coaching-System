# Voice-Coaching-System
Voice Coaching System for Musci-AI.

# Project Structure 
Musical-AI-Coach/
│
├── test_pitch.py              # Extracts pitch/F0 from test.wav
├── pitch_score.py             # Compares extracted pitch against MusicXML reference
│
├── test.wav                   # Input singing audio file
├── test.musicxml              # Reference MusicXML score
│
├── pitch_data.json            # Output from test_pitch.py
├── note_pitch_results.json    # Output from pitch_score.py
│
├── pitch_contour.png          # Pitch contour visualization
├── pitch_vs_reference.png     # Detected pitch vs MusicXML reference
├── cent_error_over_time.png   # Pitch error over time
├── pitch_accuracy_summary.png # Metric summary chart
├── note_level_pitch_errors.png # Note-level pitch error chart
