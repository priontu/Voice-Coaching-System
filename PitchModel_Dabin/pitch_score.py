import json
import numpy as np
import music21

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Config
PITCH_JSON_PATH = "pitch_data.json"
MUSICXML_PATH = "test.musicxml"

NOTE_TOLERANCE_CENTS = 50
MIN_FRAMES_PER_NOTE = 3


# Conversion
def midi_to_hz(midi_note):
    return 440.0 * (2 ** ((midi_note - 69) / 12))


def hz_to_midi(f0_hz):
    return 69 + 12 * np.log2(f0_hz / 440.0)


# Load extracted pitch data from JSON
def load_pitch_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    times = np.array([frame["time"] for frame in data["frames"]])
    f0 = np.array([frame["f0"] for frame in data["frames"]])

    return times, f0


# Get tempo (BPM) and seconds per beat from MusicXML
def get_tempo_seconds_per_beat(xml_path):
    score = music21.converter.parse(xml_path)

    tempo_marks = list(
        score.flatten().getElementsByClass(music21.tempo.MetronomeMark)
    )

    if len(tempo_marks) == 0 or tempo_marks[0].number is None:
        bpm = 84.0
    else:
        bpm = float(tempo_marks[0].number)

    seconds_per_beat = 60.0 / bpm

    return bpm, seconds_per_beat


# Load MusicXML notes and convert to frame-level reference F0 timeline
def load_musicxml_notes(xml_path):
    score = music21.converter.parse(xml_path)

    # Merge tied notes so one sustained note is not treated as many tiny notes.
    try:
        score = score.stripTies(inPlace=False)
    except Exception:
        print("Warning: Could not strip ties. Continuing with original notes.")

    notes = []

    for element in score.flatten().notesAndRests:
        if isinstance(element, music21.note.Rest):
            continue

        if isinstance(element, music21.note.Note):
            start_beat = float(element.offset)
            duration_beats = float(element.quarterLength)
            end_beat = start_beat + duration_beats

            midi = element.pitch.midi
            hz = midi_to_hz(midi)

            lyric = element.lyric if element.lyric else None

            notes.append({
                "midi": midi,
                "hz": hz,
                "start_beat": start_beat,
                "end_beat": end_beat,
                "duration_beats": duration_beats,
                "lyric": lyric
            })

    return notes


# Build frame-level reference F0 timeline from MusicXML notes
def build_reference_f0(times, notes, seconds_per_beat):
    reference_f0 = np.zeros_like(times)

    for note in notes:
        start_sec = note["start_beat"] * seconds_per_beat
        end_sec = note["end_beat"] * seconds_per_beat

        mask = (times >= start_sec) & (times < end_sec)
        reference_f0[mask] = note["hz"]

    return reference_f0


# Frame-Level Pitch Metrics
def compute_frame_pitch_metrics(times, predicted_f0, reference_f0):
    voiced_mask = (predicted_f0 > 0) & (reference_f0 > 0)

    if np.sum(voiced_mask) == 0:
        return None

    compared_times = times[voiced_mask]
    predicted = predicted_f0[voiced_mask]
    reference = reference_f0[voiced_mask]

    cent_error = 1200 * np.log2(predicted / reference)
    abs_cent_error = np.abs(cent_error)

    pitch_acc_50 = np.mean(abs_cent_error <= 50) * 100      # Percentage of frames within ±50 cents
    mace = np.mean(abs_cent_error)                          # Mean Absolute Cent Error
    pitch_rmse = np.sqrt(np.mean(cent_error ** 2))          # Root Mean Square Error in cents

    # Worst frame-level mismatches
    sorted_indices = np.argsort(abs_cent_error)[::-1]

    worst_mismatches = []
    min_time_gap = 0.30

    for idx in sorted_indices:
        t = compared_times[idx]

        too_close = any(
            abs(t - item["time"]) < min_time_gap
            for item in worst_mismatches
        )

        if too_close:
            continue

        predicted_hz = predicted[idx]
        reference_hz = reference[idx]
        error = cent_error[idx]

        worst_mismatches.append({
            "time": float(t),
            "predicted_hz": float(predicted_hz),
            "reference_hz": float(reference_hz),
            "predicted_midi": float(hz_to_midi(predicted_hz)),
            "reference_midi": float(hz_to_midi(reference_hz)),
            "cent_error": float(error),
            "abs_cent_error": float(abs(error)),
            "direction": "sharp" if error > 0 else "flat"
        })

        if len(worst_mismatches) == 2:
            break

    return {
        "PitchAcc50": pitch_acc_50,
        "MACE": mace,
        "PitchRMSE": pitch_rmse,
        "VoicedComparedFrames": int(np.sum(voiced_mask)),
        "WorstMismatches": worst_mismatches
    }


# Level Pitch Correctness (Note-Level)
def compute_note_level_pitch_correctness(times, predicted_f0, notes, seconds_per_beat):
    note_results = []

    for note_index, note in enumerate(notes, start=1):
        start_sec = note["start_beat"] * seconds_per_beat
        end_sec = note["end_beat"] * seconds_per_beat

        note_mask = (
            (times >= start_sec) &
            (times < end_sec) &
            (predicted_f0 > 0)
        )

        note_f0 = predicted_f0[note_mask]

        if len(note_f0) < MIN_FRAMES_PER_NOTE:
            note_results.append({
                "note_index": note_index,
                "scored": False,
                "reason": "not enough voiced frames",
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "reference_hz": float(note["hz"]),
                "reference_midi": float(note["midi"]),
                "lyric": note["lyric"]
            })
            continue

        reference_hz = note["hz"]

        cent_errors = 1200 * np.log2(note_f0 / reference_hz)

        median_cent_error = np.median(cent_errors)
        abs_median_cent_error = abs(median_cent_error)

        is_correct = abs_median_cent_error <= NOTE_TOLERANCE_CENTS

        note_results.append({
            "note_index": note_index,
            "scored": True,
            "correct": bool(is_correct),
            "start_sec": float(start_sec),
            "end_sec": float(end_sec),
            "duration_sec": float(end_sec - start_sec),
            "reference_hz": float(reference_hz),
            "reference_midi": float(note["midi"]),
            "median_detected_hz": float(np.median(note_f0)),
            "median_detected_midi": float(hz_to_midi(np.median(note_f0))),
            "median_cent_error": float(median_cent_error),
            "abs_median_cent_error": float(abs_median_cent_error),
            "direction": "sharp" if median_cent_error > 0 else "flat",
            "num_voiced_frames": int(len(note_f0)),
            "lyric": note["lyric"]
        })

    scored_notes = [r for r in note_results if r["scored"]]

    if len(scored_notes) == 0:
        return None

    correct_notes = [r for r in scored_notes if r["correct"]]

    note_pitch_acc_50 = (len(correct_notes) / len(scored_notes)) * 100

    worst_notes = sorted(
        scored_notes,
        key=lambda r: r["abs_median_cent_error"],
        reverse=True
    )[:2]

    return {
        "NotePitchAcc50": note_pitch_acc_50,
        "ScoredNotes": len(scored_notes),
        "CorrectNotes": len(correct_notes),
        "TotalReferenceNotes": len(notes),
        "NoteResults": note_results,
        "WorstNotes": worst_notes
    }


# Save Results
def save_note_results(note_metrics, output_path="note_pitch_results.json"):
    with open(output_path, "w") as f:
        json.dump(note_metrics, f, indent=2)

    print(f"Saved note-level results to {output_path}")


# Save Visualizations
def save_visualizations(times, predicted_f0, reference_f0, frame_metrics, note_metrics):
    # Pitch contour comparison
    plt.figure(figsize=(14, 5))
    plt.plot(times, predicted_f0, label="Detected Pitch (F0)", linewidth=1)
    plt.plot(times, reference_f0, label="MusicXML Reference Pitch", linewidth=1)

    plt.title("Detected Pitch vs MusicXML Reference Pitch")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pitch_vs_reference.png", dpi=200)
    plt.close()

    print("Saved pitch comparison plot to pitch_vs_reference.png")

    # Cent error over time
    voiced_mask = (predicted_f0 > 0) & (reference_f0 > 0)

    if np.sum(voiced_mask) > 0:
        compared_times = times[voiced_mask]
        predicted = predicted_f0[voiced_mask]
        reference = reference_f0[voiced_mask]

        cent_error = 1200 * np.log2(predicted / reference)

        plt.figure(figsize=(14, 5))
        plt.plot(compared_times, cent_error, label="Cent Error", linewidth=1)
        plt.axhline(50, linestyle="--", label="+50 cents")
        plt.axhline(-50, linestyle="--", label="-50 cents")
        plt.axhline(0, linestyle="-", label="Perfect Match")

        plt.title("Pitch Error Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Cent Error")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("cent_error_over_time.png", dpi=200)
        plt.close()

        print("Saved cent error plot to cent_error_over_time.png")

    # Metric summary bar chart
    metric_names = [
        "PitchAcc50",
        "NotePitchAcc50"
    ]

    metric_values = [
        frame_metrics["PitchAcc50"],
        note_metrics["NotePitchAcc50"]
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(metric_names, metric_values)

    plt.title("Pitch Accuracy Summary")
    plt.ylabel("Score (%)")
    plt.ylim(0, 100)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("pitch_accuracy_summary.png", dpi=200)
    plt.close()

    print("Saved metric summary chart to pitch_accuracy_summary.png")

    # Note-level error bar chart
    scored_notes = [
        note for note in note_metrics["NoteResults"]
        if note.get("scored") is True
    ]

    if len(scored_notes) > 0:
        note_indices = [note["note_index"] for note in scored_notes]
        note_errors = [note["median_cent_error"] for note in scored_notes]

        plt.figure(figsize=(14, 5))
        plt.bar(note_indices, note_errors)
        plt.axhline(50, linestyle="--", label="+50 cents")
        plt.axhline(-50, linestyle="--", label="-50 cents")
        plt.axhline(0, linestyle="-", label="Perfect Match")

        plt.title("Note-Level Median Pitch Error")
        plt.xlabel("Note Index")
        plt.ylabel("Median Cent Error")
        plt.legend()
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig("note_level_pitch_errors.png", dpi=200)
        plt.close()

        print("Saved note-level pitch error chart to note_level_pitch_errors.png")


def main():
    print("Loading extracted pitch data...")
    times, predicted_f0 = load_pitch_data(PITCH_JSON_PATH)

    print("Loading MusicXML reference...")
    notes = load_musicxml_notes(MUSICXML_PATH)

    print(f"Loaded {len(notes)} reference notes")

    bpm, seconds_per_beat = get_tempo_seconds_per_beat(MUSICXML_PATH)

    print(f"Detected BPM: {bpm}")
    print(f"Seconds per beat: {seconds_per_beat:.4f}")

    print("Building reference pitch timeline...")
    reference_f0 = build_reference_f0(times, notes, seconds_per_beat)

    print("Computing frame-level pitch metrics...")
    frame_metrics = compute_frame_pitch_metrics(times, predicted_f0, reference_f0)

    if frame_metrics is None:
        print("No overlapping voiced frames between extracted pitch and MusicXML reference.")
        return

    print("\nFrame-Level Pitch Scoring Results")
    print("---------------------------------")
    print(f"Pitch Accuracy ±50 cents: {frame_metrics['PitchAcc50']:.2f}%")
    print(f"Mean Absolute Cent Error: {frame_metrics['MACE']:.2f} cents")
    print(f"Pitch RMSE: {frame_metrics['PitchRMSE']:.2f} cents")
    print(f"Compared voiced frames: {frame_metrics['VoicedComparedFrames']}")

    print("\nWorst Frame-Level Pitch Mismatches")
    print("----------------------------------")

    for i, mismatch in enumerate(frame_metrics["WorstMismatches"], start=1):
        print(f"\nMismatch #{i}")
        print(f"Time: {mismatch['time']:.2f} sec")
        print(
            f"Expected: {mismatch['reference_hz']:.2f} Hz "
            f"/ MIDI {mismatch['reference_midi']:.2f}"
        )
        print(
            f"Detected: {mismatch['predicted_hz']:.2f} Hz "
            f"/ MIDI {mismatch['predicted_midi']:.2f}"
        )
        print(
            f"Error: {mismatch['cent_error']:.2f} cents "
            f"({mismatch['direction']})"
        )

    print("\nComputing note-level pitch correctness...")
    note_metrics = compute_note_level_pitch_correctness(
        times,
        predicted_f0,
        notes,
        seconds_per_beat
    )

    if note_metrics is None:
        print("No notes had enough voiced frames to score.")
        return

    print("\nNote-Level Pitch Correctness")
    print("----------------------------")
    print(f"NotePitchAcc50: {note_metrics['NotePitchAcc50']:.2f}%")
    print(f"Correct notes: {note_metrics['CorrectNotes']} / {note_metrics['ScoredNotes']}")
    print(f"Total reference notes: {note_metrics['TotalReferenceNotes']}")

    print("\nWorst Note-Level Pitch Mismatches")
    print("---------------------------------")

    for i, note in enumerate(note_metrics["WorstNotes"], start=1):
        lyric_text = note["lyric"] if note["lyric"] else "(no lyric)"

        print(f"\nWorst Note #{i}")
        print(f"Note index: {note['note_index']}")
        print(f"Lyric: {lyric_text}")
        print(f"Time: {note['start_sec']:.2f}s → {note['end_sec']:.2f}s")
        print(
            f"Expected: {note['reference_hz']:.2f} Hz "
            f"/ MIDI {note['reference_midi']:.2f}"
        )
        print(
            f"Detected median: {note['median_detected_hz']:.2f} Hz "
            f"/ MIDI {note['median_detected_midi']:.2f}"
        )
        print(
            f"Median error: {note['median_cent_error']:.2f} cents "
            f"({note['direction']})"
        )
        print(f"Voiced frames in note: {note['num_voiced_frames']}")

    save_note_results(note_metrics)

    save_visualizations(
        times,
        predicted_f0,
        reference_f0,
        frame_metrics,
        note_metrics
    )


if __name__ == "__main__":
    main()