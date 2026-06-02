from nicegui import ui, app
from pathlib import Path
import subprocess
import sys
import os
import time
import re
import asyncio
import json
import math
import music21

# Theme setting
dark = ui.dark_mode()
dark.enable()

# ----------------------------
# Paths
# ----------------------------

BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent

BACKEND_DIR = PROJECT_DIR / "VocalCoach_Kim"

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BACKEND_DIR / "outputs"

# MusicXML song references are stored separately from backend code.
SONG_REFERENCE_DIR = PROJECT_DIR / "song_references"

INFERENCE_SCRIPT = BACKEND_DIR / "inference" / "run_pitch.py"
PITCH_SCORE_SCRIPT = BACKEND_DIR / "scoring" / "pitch_score.py"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SONG_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

SONG_REGISTRY = {
    "Test Song": {
        "musicxml": SONG_REFERENCE_DIR / "test.musicxml",
    },
}

# Serve backend outputs so images can appear in browser
if OUTPUT_DIR.exists():
    app.add_static_files("/vocalcoach_outputs", str(OUTPUT_DIR))


mock_results = {
    "final": 84,
    "pitch": 88,
    "timing": 79,
    "duration": 82,
    "lyrics": 86,
    "feedback": [
        "Your pitch was mostly accurate, but a few notes were slightly sharp.",
        "Your timing was a little late in the middle phrase.",
        "Some note endings were cut slightly short.",
    ],
}


state = {
    "selected_song": "Test Song",

    "uploaded_file_ready": False,
    "uploaded_audio_path": None,

    "is_analyzing": False,
    "analyze_button": None,

    "current_output_stem": None,
    "pitch_data_path": None,

    "last_inference_stdout": "",
    "last_inference_stderr": "",
    "last_score_stdout": "",
    "last_score_stderr": "",

    "pitch_plot_url": None,
    "pitch_vs_reference_url": None,
    "cent_error_url": None,
    "pitch_summary_url": None,
    "note_error_url": None,

    "pitch_chart_data": [],

    "pitch_accuracy": None,
    "mace": None,
    "pitch_rmse": None,
    "note_pitch_acc": None,
    "correct_notes": None,
    "total_notes": None,
}


# ----------------------------
# State reset helpers
# ----------------------------

def reset_analysis_outputs():
    state["current_output_stem"] = None
    state["pitch_data_path"] = None

    state["last_inference_stdout"] = ""
    state["last_inference_stderr"] = ""
    state["last_score_stdout"] = ""
    state["last_score_stderr"] = ""

    state["pitch_plot_url"] = None
    state["pitch_vs_reference_url"] = None
    state["cent_error_url"] = None
    state["pitch_summary_url"] = None
    state["note_error_url"] = None

    state["pitch_chart_data"] = []

    state["pitch_accuracy"] = None
    state["mace"] = None
    state["pitch_rmse"] = None
    state["note_pitch_acc"] = None
    state["correct_notes"] = None
    state["total_notes"] = None


# ----------------------------
# UI helpers
# ----------------------------

def clamp(value, low=0, high=100):
    if value is None:
        return 0
    return max(low, min(high, value))


def format_percent(value):
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def format_cents(value):
    if value is None:
        return "N/A"
    return f"{value:.1f} cents"


def rounded_or_mock(value, mock_value):
    if value is None:
        return mock_value
    return round(value)


def get_pitch_status():
    pitch = state["pitch_accuracy"]

    if pitch is None:
        return "No pitch score available yet."

    if pitch >= 85:
        return "Excellent pitch control"
    elif pitch >= 70:
        return "Good pitch control"
    elif pitch >= 55:
        return "Moderate pitch control"
    else:
        return "Needs focused pitch practice"


def is_dark_mode():
    try:
        return bool(dark.value)
    except Exception:
        return True


def toggle_theme():
    dark.toggle()

    if state["pitch_chart_data"]:
        show_results()


def professional_score_card(title, value, subtitle=""):
    with ui.card().classes(
        "w-72 h-40 p-6 rounded-2xl shadow-lg justify-center"
    ):
        ui.label(title).classes("text-lg text-gray-400")
        ui.label(value).classes("text-5xl font-bold mt-2")
        if subtitle:
            ui.label(subtitle).classes("text-sm text-gray-500 mt-2")


def custom_progress_bar(percent):
    percent = clamp(percent)
    display_text = f"{percent:.1f}%"

    with ui.element("div").classes(
        "relative w-full h-16 rounded-full overflow-hidden bg-gray-600 mt-5"
    ):
        ui.element("div").classes(
            "absolute left-0 top-0 h-full bg-blue-400 rounded-full"
        ).style(f"width: {percent:.1f}%;")

        ui.label(display_text).classes(
            "absolute inset-0 flex items-center justify-center "
            "text-2xl font-bold text-white"
        )


def progress_metric(title, value, description):
    percent = clamp(value)

    with ui.card().classes("w-full p-7 rounded-2xl shadow-md"):
        with ui.row().classes("w-full justify-between items-center"):
            ui.label(title).classes("text-3xl font-bold")
            ui.label(format_percent(value)).classes("text-3xl font-bold")

        custom_progress_bar(percent)

        ui.label(description).classes("text-xl text-gray-400 mt-5")


def error_metric(title, value, good_value, bad_value, description):
    if value is None:
        percent = 0
        display_value = "N/A"
    else:
        percent = 100 * (bad_value - value) / (bad_value - good_value)
        percent = clamp(percent)
        display_value = format_cents(value)

    with ui.card().classes("w-full p-7 rounded-2xl shadow-md"):
        with ui.row().classes("w-full justify-between items-center"):
            ui.label(title).classes("text-3xl font-bold")
            ui.label(display_value).classes("text-3xl font-bold")

        custom_progress_bar(percent)

        ui.label(description).classes("text-xl text-gray-400 mt-5")


def correct_notes_metric():
    if state["correct_notes"] is None or state["total_notes"] is None:
        percent = 0
        display_value = "N/A"
    else:
        note_ratio = state["correct_notes"] / state["total_notes"]
        percent = note_ratio * 100
        display_value = f"{state['correct_notes']} / {state['total_notes']}"

    with ui.card().classes("w-full p-7 rounded-2xl shadow-md"):
        with ui.row().classes("w-full justify-between items-center"):
            ui.label("Correct Notes").classes("text-3xl font-bold")
            ui.label(display_value).classes("text-3xl font-bold")

        custom_progress_bar(percent)

        ui.label(
            "Number of reference notes sung correctly within the pitch tolerance."
        ).classes("text-xl text-gray-400 mt-5")


# ----------------------------
# Pitch chart helpers
# ----------------------------

def hz_to_midi(frequency):
    if frequency is None or frequency <= 0:
        return None
    return 69 + 12 * math.log2(frequency / 440.0)


def midi_to_note_name(midi_value):
    if midi_value is None:
        return ""

    note_names = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
    rounded = int(round(midi_value))
    octave = rounded // 12 - 1

    return f"{note_names[rounded % 12]}{octave}"


def get_first_existing(data, keys, default=None):
    for key in keys:
        if key in data:
            return data[key]
    return default


def load_pitch_frames():
    pitch_data_path = state.get("pitch_data_path")

    if pitch_data_path is None:
        return []

    pitch_data_path = Path(pitch_data_path)

    if not pitch_data_path.exists():
        return []

    with open(pitch_data_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    frames = []

    if isinstance(data, list):
        frames = data

    elif isinstance(data, dict):
        if "frames" in data and isinstance(data["frames"], list):
            frames = data["frames"]
        elif "data" in data and isinstance(data["data"], list):
            frames = data["data"]
        else:
            time_values = (
                data.get("time")
                or data.get("times")
                or data.get("time_s")
                or data.get("timestamps")
            )

            f0_values = (
                data.get("f0")
                or data.get("f0_hz")
                or data.get("pitch")
                or data.get("frequency")
            )

            voiced_values = data.get("voiced") or data.get("is_voiced")

            if isinstance(time_values, list) and isinstance(f0_values, list):
                for index, time_value in enumerate(time_values):
                    if index >= len(f0_values):
                        break

                    voiced = True
                    if isinstance(voiced_values, list) and index < len(voiced_values):
                        voiced = voiced_values[index]

                    frames.append(
                        {
                            "time": time_value,
                            "f0": f0_values[index],
                            "voiced": voiced,
                        }
                    )

    parsed_frames = []

    for frame in frames:
        if not isinstance(frame, dict):
            continue

        time_value = get_first_existing(
            frame,
            ["time", "t", "time_s", "timestamp", "sec", "seconds"],
        )

        f0_value = get_first_existing(
            frame,
            ["f0", "f0_hz", "pitch", "frequency", "freq", "hz"],
        )

        voiced = get_first_existing(
            frame,
            ["voiced", "is_voiced", "voice"],
            True,
        )

        try:
            time_value = float(time_value)
        except (TypeError, ValueError):
            continue

        try:
            f0_value = float(f0_value)
        except (TypeError, ValueError):
            f0_value = 0.0

        if not voiced or f0_value <= 0:
            detected_midi = None
        else:
            detected_midi = hz_to_midi(f0_value)

        parsed_frames.append(
            {
                "time": time_value,
                "detected_midi": detected_midi,
                "detected_hz": f0_value,
            }
        )

    return parsed_frames


def get_seconds_per_beat(score):
    bpm = 84.0
    metronomes = list(score.recurse().getElementsByClass(music21.tempo.MetronomeMark))

    if metronomes:
        first_bpm = metronomes[0].number
        if first_bpm is not None:
            bpm = float(first_bpm)

    return 60.0 / bpm


def load_reference_notes():
    selected_song = state["selected_song"]
    musicxml_path = SONG_REGISTRY[selected_song]["musicxml"]

    if not musicxml_path.exists():
        return []

    score = music21.converter.parse(str(musicxml_path))
    seconds_per_beat = get_seconds_per_beat(score)

    reference_notes = []

    for element in score.flatten().notesAndRests:
        if element.isRest:
            continue

        if element.isNote:
            pitch = element.pitch
        elif element.isChord:
            pitch = element.pitches[0]
        else:
            continue

        start_time = float(element.offset) * seconds_per_beat
        duration = float(element.quarterLength) * seconds_per_beat
        end_time = start_time + duration

        reference_notes.append(
            {
                "start": start_time,
                "end": end_time,
                "midi": float(pitch.midi),
                "name": pitch.nameWithOctave,
            }
        )

    return reference_notes


def find_reference_pitch(time_value, reference_notes):
    for note in reference_notes:
        if note["start"] <= time_value < note["end"]:
            return note["midi"]

    return None


def build_pitch_chart_data():
    pitch_frames = load_pitch_frames()
    reference_notes = load_reference_notes()

    chart_data = []

    for frame in pitch_frames:
        time_value = frame["time"]
        detected_midi = frame["detected_midi"]
        reference_midi = find_reference_pitch(time_value, reference_notes)

        chart_data.append(
            {
                "time": round(time_value, 3),
                "detected_midi": round(detected_midi, 3)
                if detected_midi is not None
                else None,
                "reference_midi": round(reference_midi, 3)
                if reference_midi is not None
                else None,
                "detected_note": midi_to_note_name(detected_midi),
                "reference_note": midi_to_note_name(reference_midi),
            }
        )

    return chart_data


def show_clean_pitch_chart():
    chart_data = state.get("pitch_chart_data", [])

    if not chart_data:
        return

    detected_series = [
        [point["time"], point["detected_midi"]]
        for point in chart_data
    ]

    reference_series = [
        [point["time"], point["reference_midi"]]
        for point in chart_data
    ]

    dark_mode = is_dark_mode()

    if dark_mode:
        card_class = "w-full max-w-[1000px] mx-auto p-7 rounded-2xl shadow-lg mt-8 bg-[#1b1b1b]"
        text_color = "#e5e7eb"
        muted_color = "#9ca3af"
        split_line = "#374151"
        axis_line = "#6b7280"
        tooltip_bg = "#111827"
        chart_bg = "transparent"
    else:
        card_class = "w-full max-w-[1000px] mx-auto p-7 rounded-2xl shadow-lg mt-8 bg-white"
        text_color = "#111827"
        muted_color = "#4b5563"
        split_line = "#e5e7eb"
        axis_line = "#9ca3af"
        tooltip_bg = "#ffffff"
        chart_bg = "transparent"

    options = {
        "backgroundColor": chart_bg,
        "tooltip": {
            "trigger": "axis",
            "backgroundColor": tooltip_bg,
            "borderColor": axis_line,
            "textStyle": {"color": text_color, "fontSize": 14},
        },
        "legend": {
            "data": ["Your Pitch", "Reference Melody"],
            "top": 10,
            "textStyle": {"color": text_color, "fontSize": 16},
        },
        "grid": {
            "left": "8%",
            "right": "5%",
            "top": "18%",
            "bottom": "18%",
        },
        "xAxis": {
            "type": "value",
            "name": "Time (seconds)",
            "nameTextStyle": {"color": muted_color, "fontSize": 16},
            "axisLabel": {"color": muted_color},
            "axisLine": {"lineStyle": {"color": axis_line}},
            "splitLine": {"lineStyle": {"color": split_line}},
        },
        "yAxis": {
            "type": "value",
            "name": "Pitch level",
            "nameTextStyle": {"color": muted_color, "fontSize": 16},
            "axisLabel": {"color": muted_color},
            "axisLine": {"lineStyle": {"color": axis_line}},
            "splitLine": {"lineStyle": {"color": split_line}},
        },
        "dataZoom": [
            {
                "type": "inside",
                "xAxisIndex": 0,
            },
        ],
        "series": [
            {
                "name": "Your Pitch",
                "type": "line",
                "data": detected_series,
                "connectNulls": False,
                "smooth": True,
                "symbol": "none",
                "lineStyle": {
                    "width": 4,
                    "color": "#60a5fa",
                },
                "areaStyle": {
                    "color": "rgba(96, 165, 250, 0.14)",
                },
            },
            {
                "name": "Reference Melody",
                "type": "line",
                "data": reference_series,
                "connectNulls": False,
                "smooth": False,
                "symbol": "none",
                "lineStyle": {
                    "width": 4,
                    "color": "#facc15",
                    "type": "dashed",
                },
            },
        ],
    }

    with ui.card().classes(card_class):
        ui.label("Pitch Tracking").classes("text-4xl font-bold")
        ui.label(
            "A cleaner view of how your sung pitch follows the reference melody."
        ).classes("text-xl text-gray-400 mt-2")

        # 4:1 graph ratio: 1000px wide, 250px tall
        ui.echart(options).classes("w-full h-[250px] mt-6")


# ----------------------------
# Output parsing
# ----------------------------

def parse_pitch_score_output(text: str):
    pitch_accuracy = re.search(r"Pitch Accuracy ±50 cents:\s*([\d.]+)%", text)
    mace = re.search(r"Mean Absolute Cent Error:\s*([\d.]+)\s*cents", text)
    rmse = re.search(r"Pitch RMSE:\s*([\d.]+)\s*cents", text)
    note_acc = re.search(r"NotePitchAcc50:\s*([\d.]+)%", text)
    correct_notes = re.search(r"Correct notes:\s*(\d+)\s*/\s*(\d+)", text)

    state["pitch_accuracy"] = float(pitch_accuracy.group(1)) if pitch_accuracy else None
    state["mace"] = float(mace.group(1)) if mace else None
    state["pitch_rmse"] = float(rmse.group(1)) if rmse else None
    state["note_pitch_acc"] = float(note_acc.group(1)) if note_acc else None

    if correct_notes:
        state["correct_notes"] = int(correct_notes.group(1))
        state["total_notes"] = int(correct_notes.group(2))
    else:
        state["correct_notes"] = None
        state["total_notes"] = None


# ----------------------------
# Loading UI
# ----------------------------

def show_loading(message):
    loading_area.clear()

    with loading_area:
        with ui.card().classes(
            "w-full p-10 mt-6 rounded-2xl shadow-lg items-center justify-center"
        ):
            with ui.row().classes("items-center gap-8"):
                ui.spinner(size="xl")
                ui.label(message).classes("text-5xl font-bold")

            ui.label(
                "Please wait while the pitch model analyzes the uploaded audio."
            ).classes("text-2xl text-gray-400 mt-5")

            ui.label(
                "This may take a little while because pitch extraction is currently running on the CPU."
            ).classes("text-xl text-gray-500 mt-2")


def update_loading(message):
    show_loading(message)


def hide_loading():
    loading_area.clear()


def set_analyze_button_enabled(enabled):
    button = state.get("analyze_button")

    if button is None:
        return

    if enabled:
        button.enable()
    else:
        button.disable()


# ----------------------------
# Results page
# ----------------------------

def show_results():
    results_area.clear()

    pitch_score_display = rounded_or_mock(
        state["pitch_accuracy"],
        mock_results["pitch"],
    )

    final_score_display = pitch_score_display

    with results_area:
        ui.label("Performance Results").classes("text-5xl font-bold mt-10")

        ui.label(f"Song: {state['selected_song']}").classes(
            "text-2xl text-gray-400 mt-2"
        )

        if state["uploaded_audio_path"] is not None:
            ui.label(f"Analyzed file: {state['uploaded_audio_path'].name}").classes(
                "text-xl text-gray-500"
            )

        with ui.row().classes("gap-6 flex-wrap mt-8"):
            professional_score_card(
                "Overall Score",
                str(final_score_display),
                "Based on pitch for now",
            )
            professional_score_card(
                "Pitch Score",
                str(pitch_score_display),
                get_pitch_status(),
            )
            professional_score_card(
                "Timing",
                "Soon",
                "Full pipeline not connected yet",
            )
            professional_score_card(
                "Lyrics",
                "Soon",
                "Phoneme model not connected yet",
            )

        ui.label("Pitch Summary").classes("text-4xl font-bold mt-12")

        with ui.grid(columns=2).classes("w-full gap-6 mt-4"):
            progress_metric(
                "Pitch Accuracy",
                state["pitch_accuracy"],
                "Percentage of sung pitch frames that were close to the reference melody.",
            )

            progress_metric(
                "Note-Level Pitch Correctness",
                state["note_pitch_acc"],
                "Percentage of notes where the average sung pitch matched the reference note.",
            )

            error_metric(
                "Average Pitch Difference",
                state["mace"],
                good_value=15,
                bad_value=100,
                description="Lower is better. 100 cents is one musical half-step, like C to C♯ or B to C.",
            )

            correct_notes_metric()

        show_clean_pitch_chart()

        ui.label("Coach Feedback").classes("text-4xl font-bold mt-12")

        feedback = []

        if state["pitch_accuracy"] is not None:
            if state["pitch_accuracy"] >= 85:
                feedback.append("Pitch accuracy is strong. Most sung frames are close to the reference melody.")
            elif state["pitch_accuracy"] >= 70:
                feedback.append("Pitch accuracy is good, but a few moments still drift away from the target pitch.")
            elif state["pitch_accuracy"] >= 55:
                feedback.append("Pitch accuracy is moderate. Several notes are slightly sharp or flat.")
            else:
                feedback.append("Pitch accuracy needs work. Many sung frames are outside the target pitch range.")

        if state["mace"] is not None:
            feedback.append(
                f"Average pitch difference is {state['mace']:.1f} cents. "
                "For reference, 100 cents is one musical half-step, like C to C♯ or B to C."
            )

        if state["note_pitch_acc"] is not None:
            feedback.append(f"Note-level pitch correctness is {state['note_pitch_acc']:.1f}%.")

        if state["correct_notes"] is not None and state["total_notes"] is not None:
            feedback.append(
                f"{state['correct_notes']} of {state['total_notes']} reference notes were matched correctly."
            )

        if not feedback:
            feedback = mock_results["feedback"]

        with ui.column().classes("w-full gap-4 mt-4"):
            for message in feedback:
                with ui.card().classes("w-full p-5 rounded-xl shadow-md"):
                    ui.label(message).classes("text-xl")

        with ui.expansion("Advanced Graphs", icon="analytics").classes("w-full mt-12"):
            ui.label(
                "Detailed analysis plots generated by the pitch pipeline."
            ).classes("text-lg text-gray-400 mb-4")

            if state["pitch_vs_reference_url"] is not None:
                ui.label("Original Pitch vs Reference Plot").classes("text-2xl font-bold mt-4")
                ui.image(state["pitch_vs_reference_url"]).classes(
                    "w-full max-w-6xl mx-auto rounded-xl mt-2"
                )

            if state["pitch_plot_url"] is not None:
                ui.label("Pitch Extraction Visualization").classes("text-2xl font-bold mt-6")
                ui.image(state["pitch_plot_url"]).classes(
                    "w-full max-w-6xl mx-auto rounded-xl mt-2"
                )

            if state["cent_error_url"] is not None:
                ui.label("Cent Error Over Time").classes("text-2xl font-bold mt-6")
                ui.image(state["cent_error_url"]).classes(
                    "w-full max-w-6xl mx-auto rounded-xl mt-2"
                )

            if state["note_error_url"] is not None:
                ui.label("Note-Level Pitch Errors").classes("text-2xl font-bold mt-6")
                ui.image(state["note_error_url"]).classes(
                    "w-full max-w-6xl mx-auto rounded-xl mt-2"
                )

        with ui.expansion("Debug Output", icon="terminal").classes("w-full mt-4"):
            if state["last_inference_stdout"]:
                ui.label("run_pitch.py output").classes("text-xl font-bold")
                ui.label(state["last_inference_stdout"]).classes(
                    "text-sm whitespace-pre-wrap"
                )

            if state["last_score_stdout"]:
                ui.label("pitch_score.py output").classes("text-xl font-bold mt-4")
                ui.label(state["last_score_stdout"]).classes(
                    "text-sm whitespace-pre-wrap"
                )

            combined_logs = "\n".join(
                [
                    state["last_inference_stderr"],
                    state["last_score_stderr"],
                ]
            ).strip()

            if combined_logs:
                ui.label("Log output").classes("text-xl font-bold mt-4")
                ui.label(combined_logs).classes("text-sm whitespace-pre-wrap")

    ui.notify("Analysis complete")


# ----------------------------
# Command runners
# ----------------------------

async def run_command(command, cwd):
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PYTHONPATH": str(BACKEND_DIR) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }

    return await asyncio.to_thread(
        subprocess.run,
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def prepare_selected_musicxml() -> bool:
    selected_song = state["selected_song"]

    if selected_song not in SONG_REGISTRY:
        analysis_status_label.set_text("Selected song is not in the song registry.")
        ui.notify("Selected song is not registered.", color="negative")
        return False

    source_musicxml = SONG_REGISTRY[selected_song]["musicxml"]

    if not source_musicxml.exists():
        analysis_status_label.set_text("Selected song MusicXML file not found.")
        ui.notify(f"Missing MusicXML: {source_musicxml}", color="negative")
        return False

    print("Selected song:", selected_song)
    print("Using MusicXML:", source_musicxml)

    return True


async def run_pitch_inference(uploaded_path: Path) -> bool:
    analysis_status_label.set_text("Running pitch extraction...")
    update_loading("Running pitch extraction...")

    state["last_inference_stdout"] = ""
    state["last_inference_stderr"] = ""
    state["pitch_plot_url"] = None
    state["pitch_chart_data"] = []

    if not BACKEND_DIR.exists():
        analysis_status_label.set_text("VocalCoach_Kim folder not found.")
        ui.notify(f"Missing folder: {BACKEND_DIR}", color="negative")
        return False

    if not INFERENCE_SCRIPT.exists():
        analysis_status_label.set_text("run_pitch.py not found.")
        ui.notify(f"Missing file: {INFERENCE_SCRIPT}", color="negative")
        return False

    uploaded_path = uploaded_path.resolve()

    output_stem = uploaded_path.stem
    state["current_output_stem"] = output_stem

    pitch_data_path = (OUTPUT_DIR / f"{output_stem}_pitch_data.json").resolve()
    pitch_plot_path = (OUTPUT_DIR / f"{output_stem}_pitch_data_vad_pitch.png").resolve()

    state["pitch_data_path"] = pitch_data_path

    if pitch_data_path.exists():
        pitch_data_path.unlink()

    if pitch_plot_path.exists():
        pitch_plot_path.unlink()

    command = [
        sys.executable,
        str(INFERENCE_SCRIPT.resolve()),
        "--audio",
        str(uploaded_path),
        "--output",
        str(pitch_data_path),
        "--visualize",
    ]

    print("Running inference command:")
    print(" ".join(command))
    print("Working directory:", OUTPUT_DIR.resolve())

    result = await run_command(command, OUTPUT_DIR.resolve())

    state["last_inference_stdout"] = result.stdout
    state["last_inference_stderr"] = result.stderr

    print("Inference return code:", result.returncode)
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)

    if result.returncode != 0:
        analysis_status_label.set_text("Pitch extraction failed.")
        ui.notify("Pitch extraction failed. Check terminal output.", color="negative")
        return False

    if not pitch_data_path.exists():
        analysis_status_label.set_text("Pitch extraction did not create pitch data.")
        ui.notify(f"Missing output JSON: {pitch_data_path}", color="negative")
        return False

    if pitch_plot_path.exists():
        state["pitch_plot_url"] = (
            f"/vocalcoach_outputs/{pitch_plot_path.name}?t={int(time.time())}"
        )
    else:
        state["pitch_plot_url"] = None

    analysis_status_label.set_text("Pitch extraction complete.")
    return True


async def run_pitch_score() -> bool:
    analysis_status_label.set_text("Running pitch scoring...")
    update_loading("Running pitch scoring...")

    state["last_score_stdout"] = ""
    state["last_score_stderr"] = ""

    state["pitch_accuracy"] = None
    state["mace"] = None
    state["pitch_rmse"] = None
    state["note_pitch_acc"] = None
    state["correct_notes"] = None
    state["total_notes"] = None

    state["pitch_vs_reference_url"] = None
    state["cent_error_url"] = None
    state["note_error_url"] = None

    if not PITCH_SCORE_SCRIPT.exists():
        analysis_status_label.set_text("pitch_score.py not found.")
        ui.notify(f"Missing file: {PITCH_SCORE_SCRIPT}", color="negative")
        return False

    if not prepare_selected_musicxml():
        return False

    pitch_data_path = state.get("pitch_data_path")

    if pitch_data_path is None:
        analysis_status_label.set_text("pitch data path is missing.")
        ui.notify("Missing pitch data path. Run inference first.", color="negative")
        return False

    pitch_data_path = Path(pitch_data_path).resolve()

    if not pitch_data_path.exists():
        analysis_status_label.set_text("pitch data JSON not found.")
        ui.notify(f"Missing pitch JSON: {pitch_data_path}", color="negative")
        return False

    musicxml_path = SONG_REGISTRY[state["selected_song"]]["musicxml"].resolve()

    output_stem = state.get("current_output_stem") or pitch_data_path.stem
    note_results_path = (OUTPUT_DIR / f"{output_stem}_note_pitch_results.json").resolve()

    pitch_vs_reference_path = (OUTPUT_DIR / "pitch_vs_reference.png").resolve()
    cent_error_path = (OUTPUT_DIR / "cent_error_over_time.png").resolve()
    note_error_path = (OUTPUT_DIR / "note_level_pitch_errors.png").resolve()

    for old_file in [pitch_vs_reference_path, cent_error_path, note_error_path, note_results_path]:
        if old_file.exists():
            old_file.unlink()

    command = [
        sys.executable,
        str(PITCH_SCORE_SCRIPT.resolve()),
        "--pitch-json",
        str(pitch_data_path),
        "--musicxml",
        str(musicxml_path),
        "--output",
        str(note_results_path),
        "--visualize",
    ]

    print("Running pitch score command:")
    print(" ".join(command))
    print("Working directory:", OUTPUT_DIR.resolve())

    result = await run_command(command, OUTPUT_DIR.resolve())

    state["last_score_stdout"] = result.stdout
    state["last_score_stderr"] = result.stderr

    print("Pitch score return code:", result.returncode)
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)

    if result.returncode != 0:
        analysis_status_label.set_text("Pitch scoring failed.")
        ui.notify("Pitch scoring failed. Check terminal output.", color="negative")
        return False

    parse_pitch_score_output(result.stdout)
    state["pitch_chart_data"] = build_pitch_chart_data()

    timestamp = int(time.time())

    if pitch_vs_reference_path.exists():
        state["pitch_vs_reference_url"] = (
            f"/vocalcoach_outputs/{pitch_vs_reference_path.name}?t={timestamp}"
        )

    if cent_error_path.exists():
        state["cent_error_url"] = (
            f"/vocalcoach_outputs/{cent_error_path.name}?t={timestamp}"
        )

    if note_error_path.exists():
        state["note_error_url"] = (
            f"/vocalcoach_outputs/{note_error_path.name}?t={timestamp}"
        )

    analysis_status_label.set_text("Pitch scoring complete.")
    return True


# ----------------------------
# App actions
# ----------------------------

async def analyze_singing():
    if state["is_analyzing"]:
        ui.notify("Analysis is already running.", color="warning")
        return

    print("Analyze clicked")
    print("Selected song:", state["selected_song"])
    print("Uploaded ready:", state["uploaded_file_ready"])
    print("Uploaded path:", state["uploaded_audio_path"])

    uploaded_path = state["uploaded_audio_path"]

    if not state["uploaded_file_ready"] or uploaded_path is None:
        ui.notify("Please upload one audio file first.", color="negative")
        return

    if not uploaded_path.exists():
        ui.notify("The uploaded file was not found on disk.", color="negative")
        upload_status_label.set_text("Uploaded file is missing.")
        upload_path_label.set_text(f"Missing path: {uploaded_path}")
        return

    state["is_analyzing"] = True
    set_analyze_button_enabled(False)
    results_area.clear()

    upload_status_label.set_text(f"Ready to analyze: {uploaded_path.name}")
    show_loading("Starting analysis...")

    await asyncio.sleep(0.1)

    try:
        success = await run_pitch_inference(uploaded_path)
        if not success:
            return

        success = await run_pitch_score()
        if not success:
            return

        hide_loading()
        show_results()

    finally:
        state["is_analyzing"] = False
        set_analyze_button_enabled(True)
        hide_loading()


def handle_rejected(e):
    ui.notify("File rejected. Please upload only one audio file.", color="negative")
    upload_status_label.set_text("File rejected. Upload exactly one audio file.")


async def handle_upload(e):
    uploaded_file = e.file
    safe_name = Path(uploaded_file.name).name
    save_path = UPLOAD_DIR / safe_name

    print("Upload handler called")
    print("File name:", safe_name)
    print("Saving to:", save_path)

    try:
        for old_file in UPLOAD_DIR.iterdir():
            if old_file.is_file():
                old_file.unlink()

        await uploaded_file.save(save_path)

        if not save_path.exists():
            raise FileNotFoundError(f"Could not save file to {save_path}")

        if save_path.stat().st_size == 0:
            raise ValueError("Uploaded file was saved, but it is empty.")

        state["uploaded_audio_path"] = save_path
        state["uploaded_file_ready"] = True

        reset_analysis_outputs()

        upload_status_label.set_text(f"Uploaded: {safe_name}")
        upload_path_label.set_text(f"Saved path: {save_path}")
        analysis_status_label.set_text("Ready. Click Analyze Singing.")

        remove_file_area.clear()
        with remove_file_area:
            ui.button("✕ Remove File", on_click=remove_uploaded_file).classes(
                "text-lg p-4 mt-2"
            ).props("color=negative")

        ui.notify(f"Uploaded: {safe_name}")

        print("Uploaded ready:", state["uploaded_file_ready"])
        print("Uploaded path:", state["uploaded_audio_path"])
        print("File exists:", save_path.exists())
        print("File size:", save_path.stat().st_size)

    except Exception as error:
        state["uploaded_audio_path"] = None
        state["uploaded_file_ready"] = False

        upload_status_label.set_text("Upload failed.")
        upload_path_label.set_text(str(error))
        analysis_status_label.set_text("Upload failed.")

        ui.notify("Upload failed. Check terminal output.", color="negative")
        print("Upload failed:", error)


def remove_uploaded_file():
    uploaded_path = state["uploaded_audio_path"]

    if uploaded_path is not None and uploaded_path.exists():
        uploaded_path.unlink()

    state["uploaded_audio_path"] = None
    state["uploaded_file_ready"] = False

    reset_analysis_outputs()

    upload_status_label.set_text("No file uploaded yet.")
    upload_path_label.set_text("")
    analysis_status_label.set_text("")
    remove_file_area.clear()
    loading_area.clear()
    results_area.clear()

    ui.notify("Uploaded file removed. You can upload another file.")


def update_selected_song(value):
    state["selected_song"] = value
    analysis_status_label.set_text(f"Selected song: {value}")


# ----------------------------
# Page layout
# ----------------------------

ui.page_title("Vocal Coach AI")

with ui.row().classes("w-full justify-between items-center p-4"):
    ui.label("Vocal Coach AI").classes("text-6xl font-bold")
    ui.button("Toggle Theme", on_click=toggle_theme).classes("text-lg p-4")

ui.label(
    "Choose a song, upload your singing, and receive feedback on pitch performance."
).classes("text-2xl text-gray-400 px-4")


with ui.card().classes("w-full mt-8 p-8 rounded-2xl shadow-lg"):
    ui.label("Song Setup").classes("text-4xl font-bold")

    song = ui.select(
        list(SONG_REGISTRY.keys()),
        value=state["selected_song"],
        label="Choose a song",
        on_change=lambda e: update_selected_song(e.value),
    ).classes(
        "w-96 text-xl mt-4"
    ).props(
        "dense=false outlined"
    ).style(
        "min-height: 72px; padding-top: 8px;"
    )

    ui.label("Upload Singing Audio").classes("text-4xl font-bold mt-8")
    ui.label(
        "Upload exactly one WAV, MP3, M4A, or FLAC file."
    ).classes("text-xl text-gray-400 mt-2")

    with ui.row().classes("w-full justify-center mt-8"):
        with ui.card().classes(
            "w-[1000px] h-[750px] p-16 items-center justify-center "
            "border-4 border-dashed border-gray-500 rounded-2xl"
        ):
            ui.label("Upload one singing file").classes(
                "text-5xl font-bold text-center"
            )

            ui.label(
                "Drop one file directly onto the upload box below"
            ).classes("text-2xl text-gray-400 text-center mt-4")

            ui.upload(
                label="DROP OR CHOOSE ONE AUDIO FILE HERE",
                auto_upload=True,
                multiple=False,
                on_upload=handle_upload,
                on_rejected=handle_rejected,
            ).props(
                'accept=".wav,.mp3,.m4a,.flac" max-files="1"'
            ).classes(
                "w-[900px] h-[360px] mt-12 text-4xl font-bold"
            ).style("font-size: 40px;")

            ui.label(
                "After the file uploads, its name should appear below."
            ).classes("text-lg text-gray-500 mt-4")

    state["analyze_button"] = ui.button(
        "Analyze Singing",
        on_click=analyze_singing,
    ).classes("mt-8 text-xl p-5")


upload_status_label = ui.label("No file uploaded yet.").classes(
    "text-xl text-gray-400 mt-4 px-4"
)
upload_path_label = ui.label("").classes("text-lg text-blue-400 mt-2 px-4")
analysis_status_label = ui.label("").classes("text-2xl text-green-400 mt-2 px-4")
remove_file_area = ui.row().classes("px-4")

loading_area = ui.column().classes("w-full px-4")
results_area = ui.column().classes("w-full px-4 pb-12")

ui.run()
