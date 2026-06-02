from nicegui import ui, app
from pathlib import Path
import subprocess
import sys
import os
import time
import asyncio
import json
import re

# ----------------------------
# Theme
# ----------------------------

dark = ui.dark_mode()
dark.enable()


# ----------------------------
# Paths
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

BACKEND_DIR = PROJECT_DIR / "VocalCoach_Kim"

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BACKEND_DIR / "outputs"

# Option B:
# Song references are stored inside VocalCoach_Kim.
SONG_REFERENCE_DIR = BACKEND_DIR / "song_references"

RUN_PIPELINE_SCRIPT = BACKEND_DIR / "inference" / "run_pipeline.py"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SONG_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

SONG_REGISTRY = {
    "Test Song": {
        "musicxml": SONG_REFERENCE_DIR / "test.musicxml",
        "textgrid": SONG_REFERENCE_DIR / "test.TextGrid",
    },
}

if OUTPUT_DIR.exists():
    app.add_static_files("/vocalcoach_outputs", str(OUTPUT_DIR))


# ----------------------------
# State
# ----------------------------

state = {
    "selected_song": "Test Song",

    "uploaded_file_ready": False,
    "uploaded_audio_path": None,

    "is_analyzing": False,
    "analyze_button": None,

    "current_output_stem": None,
    "unified_json_path": None,
    "dashboard_url": None,

    "last_pipeline_stdout": "",
    "last_pipeline_stderr": "",

    "overall_score": None,
    "pitch_score": None,
    "timing_score": None,
    "duration_score": None,
    "lyrics_score": None,

    "pitch_accuracy": None,
    "mace": None,
    "pitch_rmse": None,
    "note_pitch_acc": None,

    "timing_accuracy": None,
    "onset_mae_ms": None,
    "offset_mae_ms": None,
    "ioi_mae_ms": None,

    "duration_error_s": None,
    "relative_duration_error": None,
    "duration_ratio": None,

    "word_accuracy": None,
    "phoneme_overlap_accuracy": None,
    "label_match_rate": None,
    "phoneme_boundary_error_ms": None,

    "feedback": [],
}


# ----------------------------
# Reset helpers
# ----------------------------

def reset_analysis_outputs():
    state["current_output_stem"] = None
    state["unified_json_path"] = None
    state["dashboard_url"] = None

    state["last_pipeline_stdout"] = ""
    state["last_pipeline_stderr"] = ""

    state["overall_score"] = None
    state["pitch_score"] = None
    state["timing_score"] = None
    state["duration_score"] = None
    state["lyrics_score"] = None

    state["pitch_accuracy"] = None
    state["mace"] = None
    state["pitch_rmse"] = None
    state["note_pitch_acc"] = None

    state["timing_accuracy"] = None
    state["onset_mae_ms"] = None
    state["offset_mae_ms"] = None
    state["ioi_mae_ms"] = None

    state["duration_error_s"] = None
    state["relative_duration_error"] = None
    state["duration_ratio"] = None

    state["word_accuracy"] = None
    state["phoneme_overlap_accuracy"] = None
    state["label_match_rate"] = None
    state["phoneme_boundary_error_ms"] = None

    state["feedback"] = []


# ----------------------------
# Formatting helpers
# ----------------------------

def has_value(value):
    return value is not None


def clamp(value, low=0, high=100):
    if value is None:
        return 0

    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0

    return max(low, min(high, value))


def normalize_score(value):
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if 0 <= value <= 1:
        return value * 100

    return value


def format_score(value):
    value = normalize_score(value)

    if value is None:
        return "N/A"

    return f"{value:.1f}"


def format_percent(value):
    value = normalize_score(value)

    if value is None:
        return "N/A"

    return f"{value:.1f}%"


def format_ms(value):
    if value is None:
        return "N/A"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    return f"{value:.1f} ms"


def format_seconds(value):
    if value is None:
        return "N/A"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    return f"{value:.3f} s"


def format_cents(value):
    if value is None:
        return "N/A"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    return f"{value:.1f} cents"


def format_ratio(value):
    if value is None:
        return "N/A"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    return f"{value:.2f}x"


def get_level(value):
    value = normalize_score(value)

    if value is None:
        return "No score available yet"

    if value >= 90:
        return "Excellent"
    if value >= 75:
        return "Good"
    if value >= 55:
        return "Fair"
    return "Needs work"


def toggle_theme():
    dark.toggle()
    if state["overall_score"] is not None:
        show_results()


# ----------------------------
# Recursive JSON parsing helpers
# ----------------------------

def find_key_recursive(data, possible_keys):
    possible_keys = set(possible_keys)

    if isinstance(data, dict):
        for key, value in data.items():
            if key in possible_keys:
                return value

        for value in data.values():
            found = find_key_recursive(value, possible_keys)
            if found is not None:
                return found

    elif isinstance(data, list):
        for item in data:
            found = find_key_recursive(item, possible_keys)
            if found is not None:
                return found

    return None


def find_score_by_category(data, category_name):
    if isinstance(data, dict):
        for key, value in data.items():
            key_lower = str(key).lower()

            if category_name in key_lower and "score" in key_lower:
                return value

            if key_lower == category_name and isinstance(value, dict):
                score = find_key_recursive(
                    value,
                    ["score", "value", "category_score", "normalized_score"],
                )
                if score is not None:
                    return score

        for value in data.values():
            found = find_score_by_category(value, category_name)
            if found is not None:
                return found

    elif isinstance(data, list):
        for item in data:
            found = find_score_by_category(item, category_name)
            if found is not None:
                return found

    return None


def parse_unified_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    state["overall_score"] = find_key_recursive(
        data,
        [
            "overall_score",
            "final_score",
            "performance_score",
            "total_score",
            "score_final",
        ],
    )

    state["pitch_score"] = (
        find_score_by_category(data, "pitch")
        or find_key_recursive(data, ["pitch_score", "score_pitch"])
    )

    state["timing_score"] = (
        find_score_by_category(data, "timing")
        or find_key_recursive(data, ["timing_score", "score_timing"])
    )

    state["duration_score"] = (
        find_score_by_category(data, "duration")
        or find_key_recursive(
            data,
            ["duration_score", "duration_control_score", "score_duration"],
        )
    )

    state["lyrics_score"] = (
        find_score_by_category(data, "lyric")
        or find_score_by_category(data, "lyrics")
        or find_key_recursive(
            data,
            ["lyric_score", "lyrics_score", "lyric_timing_score", "score_lyric"],
        )
    )

    state["pitch_accuracy"] = find_key_recursive(
        data,
        [
            "pitch_accuracy",
            "pitch_acc",
            "pitch_acc_50",
            "pitchacc50",
            "PitchAcc50",
        ],
    )

    state["mace"] = find_key_recursive(
        data,
        [
            "mace",
            "mace_cents",
            "mean_absolute_cent_error",
            "mean_abs_cent_error",
        ],
    )

    state["pitch_rmse"] = find_key_recursive(
        data,
        [
            "pitch_rmse",
            "pitch_rmse_cents",
            "rmse_cents",
            "PitchRMSE",
        ],
    )

    state["note_pitch_acc"] = find_key_recursive(
        data,
        [
            "note_pitch_accuracy",
            "note_pitch_acc",
            "note_pitch_acc_50",
            "NotePitchAcc50",
        ],
    )

    state["timing_accuracy"] = find_key_recursive(
        data,
        [
            "timing_accuracy",
            "timing_acc",
            "timing_acc_50",
            "TimingAcc50",
        ],
    )

    state["onset_mae_ms"] = find_key_recursive(
        data,
        [
            "onset_mae_ms",
            "mean_abs_onset_error_ms",
            "mean_absolute_onset_error_ms",
            "onset_error_mae_ms",
        ],
    )

    state["offset_mae_ms"] = find_key_recursive(
        data,
        [
            "offset_mae_ms",
            "mean_abs_offset_error_ms",
            "mean_absolute_offset_error_ms",
            "offset_error_mae_ms",
        ],
    )

    state["ioi_mae_ms"] = find_key_recursive(
        data,
        [
            "ioi_mae_ms",
            "ioi_mean_absolute_error_ms",
            "ioi_error_ms",
        ],
    )

    state["duration_error_s"] = find_key_recursive(
        data,
        [
            "duration_error",
            "duration_error_s",
            "duration_mae_s",
            "mean_duration_error",
            "duration_mae",
        ],
    )

    state["relative_duration_error"] = find_key_recursive(
        data,
        [
            "relative_duration_error",
            "rel_duration_error",
            "rel_dur_err",
            "RelDurErr",
        ],
    )

    state["duration_ratio"] = find_key_recursive(
        data,
        [
            "duration_ratio",
            "dur_ratio",
            "DurRatio",
        ],
    )

    state["word_accuracy"] = find_key_recursive(
        data,
        [
            "word_accuracy",
            "word_acc",
            "WordAcc",
        ],
    )

    state["phoneme_overlap_accuracy"] = find_key_recursive(
        data,
        [
            "phoneme_overlap_accuracy",
            "overlap_accuracy",
            "overlap_acc",
            "OverlapAcc",
        ],
    )

    state["label_match_rate"] = find_key_recursive(
        data,
        [
            "label_match_rate",
            "label_match",
            "LabelMatch",
        ],
    )

    state["phoneme_boundary_error_ms"] = find_key_recursive(
        data,
        [
            "phoneme_boundary_error_ms",
            "phoneme_boundary_mae_ms",
            "phon_bnd_err_ms",
            "PhonBndErr",
        ],
    )

    feedback = find_key_recursive(
        data,
        [
            "feedback",
            "interpretation",
            "interpretation_summary",
            "comments",
            "messages",
        ],
    )

    if isinstance(feedback, list):
        state["feedback"] = [str(item) for item in feedback]
    elif isinstance(feedback, dict):
        collected = []

        for key in ["strengths", "weaknesses", "suggestions", "messages"]:
            value = feedback.get(key)
            if isinstance(value, list):
                collected.extend([str(item) for item in value])
            elif isinstance(value, str):
                collected.append(value)

        state["feedback"] = collected
    elif isinstance(feedback, str):
        state["feedback"] = [feedback]
    else:
        state["feedback"] = []


def parse_stdout_fallback(text: str):
    overall_match = re.search(r"Score:\s*([\d.]+)\s*/\s*100", text, re.IGNORECASE)
    if overall_match:
        state["overall_score"] = float(overall_match.group(1))

    pitch_match = re.search(r"pitch\s*=\s*(-?[\d.]+)", text, re.IGNORECASE)
    if pitch_match:
        state["pitch_score"] = float(pitch_match.group(1))

    timing_match = re.search(r"timing\s*=\s*(-?[\d.]+)", text, re.IGNORECASE)
    if timing_match:
        state["timing_score"] = float(timing_match.group(1))

    duration_match = re.search(r"duration\s*=\s*(-?[\d.]+)", text, re.IGNORECASE)
    if duration_match:
        state["duration_score"] = float(duration_match.group(1))

    lyric_match = re.search(r"lyric\s*=\s*(-?[\d.]+)", text, re.IGNORECASE)
    if lyric_match:
        state["lyrics_score"] = float(lyric_match.group(1))

    pitch_acc_match = re.search(r"Pitch\s+acc:\s*([\d.]+)%", text, re.IGNORECASE)
    if pitch_acc_match:
        state["pitch_accuracy"] = float(pitch_acc_match.group(1))

    mace_match = re.search(r"MACE\s*([\d.]+)", text, re.IGNORECASE)
    if mace_match:
        state["mace"] = float(mace_match.group(1))

    timing_line_match = re.search(
        r"Timing:\s*([\d.]+)%\s*MAE\s*([\d.]+)\s*ms",
        text,
        re.IGNORECASE,
    )
    if timing_line_match:
        state["timing_accuracy"] = float(timing_line_match.group(1))
        state["onset_mae_ms"] = float(timing_line_match.group(2))

    strength_match = re.search(r"Strengths:\s*(.+)", text, re.IGNORECASE)
    improve_match = re.search(r"Improve:\s*(.+)", text, re.IGNORECASE)

    fallback_feedback = []

    if strength_match:
        fallback_feedback.append(f"Strength: {strength_match.group(1).strip()}")

    if improve_match:
        fallback_feedback.append(f"Focus area: {improve_match.group(1).strip()}")

    if fallback_feedback:
        state["feedback"] = fallback_feedback


# ----------------------------
# Output file discovery
# ----------------------------

def newest_file_after(directory: Path, suffixes, start_time, stem_hint=None):
    candidates = []

    for path in directory.glob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in suffixes:
            continue

        if path.stat().st_mtime < start_time:
            continue

        if stem_hint is not None and stem_hint.lower() not in path.name.lower():
            continue

        candidates.append(path)

    if not candidates:
        for path in directory.glob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in suffixes:
                continue

            if path.stat().st_mtime < start_time:
                continue

            candidates.append(path)

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


# ----------------------------
# UI components
# ----------------------------

def score_card(title, value, subtitle="", show_out_of_100=False, wide=False):
    card_width = "w-96" if wide else "w-72"

    with ui.card().classes(f"{card_width} h-40 p-6 rounded-2xl shadow-lg justify-center"):
        ui.label(title).classes("text-lg text-gray-400")

        if show_out_of_100 and value is not None:
            score_text = f"{format_score(value)} / 100"
        else:
            score_text = format_score(value)

        ui.label(score_text).classes("text-5xl font-bold mt-2")

        if subtitle:
            ui.label(subtitle).classes("text-sm text-gray-500 mt-2")


def custom_progress_bar(percent):
    percent = clamp(normalize_score(percent))
    display_text = f"{percent:.1f}%"

    with ui.element("div").classes(
        "relative w-full h-16 rounded-full overflow-hidden bg-gray-600 mt-5"
    ):
        ui.element("div").classes(
            "absolute left-0 top-0 h-full bg-blue-400 rounded-full"
        ).style(f"width: {percent:.1f}%;")

        ui.label(display_text).classes(
            "absolute inset-0 flex items-center justify-center text-2xl font-bold text-white"
        )


def score_metric(title, value, description):
    with ui.card().classes("w-full p-7 rounded-2xl shadow-md"):
        with ui.row().classes("w-full justify-between items-center"):
            ui.label(title).classes("text-3xl font-bold")
            ui.label(format_percent(value)).classes("text-3xl font-bold")

        custom_progress_bar(value)

        ui.label(description).classes("text-xl text-gray-400 mt-5")


def error_metric(title, value, good_value, bad_value, unit, description):
    if value is None:
        percent = 0
        display_value = "N/A"
    else:
        try:
            numeric_value = float(value)
            percent = 100 * (bad_value - numeric_value) / (bad_value - good_value)
            percent = clamp(percent)
        except (TypeError, ValueError):
            percent = 0

        if unit == "ms":
            display_value = format_ms(value)
        elif unit == "s":
            display_value = format_seconds(value)
        elif unit == "cents":
            display_value = format_cents(value)
        elif unit == "ratio":
            display_value = format_ratio(value)
        else:
            display_value = str(value)

    with ui.card().classes("w-full p-7 rounded-2xl shadow-md"):
        with ui.row().classes("w-full justify-between items-center"):
            ui.label(title).classes("text-3xl font-bold")
            ui.label(display_value).classes("text-3xl font-bold")

        custom_progress_bar(percent)

        ui.label(description).classes("text-xl text-gray-400 mt-5")


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
                "Please wait while the full vocal coaching pipeline analyzes the uploaded audio."
            ).classes("text-2xl text-gray-400 mt-5")

            ui.label(
                "This may take a while on CPU because pitch, onset/offset, and phoneme models may all run."
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
# Command runner
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


def prepare_selected_references() -> bool:
    selected_song = state["selected_song"]

    if selected_song not in SONG_REGISTRY:
        analysis_status_label.set_text("Selected song is not in the song registry.")
        ui.notify("Selected song is not registered.", color="negative")
        return False

    musicxml_path = SONG_REGISTRY[selected_song]["musicxml"]
    textgrid_path = SONG_REGISTRY[selected_song]["textgrid"]

    print("PROJECT_DIR:", PROJECT_DIR)
    print("BACKEND_DIR:", BACKEND_DIR)
    print("SONG_REFERENCE_DIR:", SONG_REFERENCE_DIR)
    print("MusicXML path:", musicxml_path)
    print("MusicXML exists:", musicxml_path.exists())
    print("TextGrid path:", textgrid_path)
    print("TextGrid exists:", textgrid_path.exists())

    if not musicxml_path.exists():
        analysis_status_label.set_text("Selected song MusicXML file not found.")
        ui.notify(f"Missing MusicXML: {musicxml_path}", color="negative")
        return False

    if not textgrid_path.exists():
        analysis_status_label.set_text("Selected song TextGrid file not found.")
        ui.notify(f"Missing TextGrid: {textgrid_path}", color="negative")
        return False

    print("Selected song:", selected_song)
    print("Using MusicXML:", musicxml_path)
    print("Using TextGrid:", textgrid_path)

    return True


async def run_full_pipeline(uploaded_path: Path) -> bool:
    analysis_status_label.set_text("Running full vocal coaching pipeline...")
    update_loading("Running pitch, timing, duration, and lyric analysis...")

    state["last_pipeline_stdout"] = ""
    state["last_pipeline_stderr"] = ""

    if not BACKEND_DIR.exists():
        analysis_status_label.set_text("VocalCoach_Kim folder not found.")
        ui.notify(f"Missing folder: {BACKEND_DIR}", color="negative")
        return False

    if not RUN_PIPELINE_SCRIPT.exists():
        analysis_status_label.set_text("run_pipeline.py not found.")
        ui.notify(f"Missing file: {RUN_PIPELINE_SCRIPT}", color="negative")
        return False

    if not prepare_selected_references():
        return False

    uploaded_path = uploaded_path.resolve()
    output_stem = uploaded_path.stem
    state["current_output_stem"] = output_stem

    musicxml_path = SONG_REGISTRY[state["selected_song"]]["musicxml"].resolve()
    textgrid_path = SONG_REGISTRY[state["selected_song"]]["textgrid"].resolve()

    start_time = time.time()

    command = [
        sys.executable,
        str(RUN_PIPELINE_SCRIPT.resolve()),
        "--audio",
        str(uploaded_path),
        "--musicxml",
        str(musicxml_path),
        "--textgrid",
        str(textgrid_path),
        "--compute-metrics",
        "--compute-scores",
        "--export-json",
        "--plot",
        "--output_dir",
        str(OUTPUT_DIR.resolve()),
    ]

    print("Running full pipeline command:")
    print(" ".join(command))
    print("Working directory:", BACKEND_DIR.resolve())

    result = await run_command(command, BACKEND_DIR.resolve())

    state["last_pipeline_stdout"] = result.stdout
    state["last_pipeline_stderr"] = result.stderr

    print("Full pipeline return code:", result.returncode)
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)

    if result.returncode != 0:
        analysis_status_label.set_text("Full pipeline failed.")
        ui.notify("Full pipeline failed. Check terminal output.", color="negative")
        return False

    json_path = newest_file_after(
        OUTPUT_DIR,
        suffixes={".json"},
        start_time=start_time,
        stem_hint=output_stem,
    )

    if json_path is None:
        analysis_status_label.set_text("Full pipeline did not create JSON output.")
        ui.notify("Could not find pipeline JSON output.", color="negative")
    else:
        state["unified_json_path"] = json_path
        parse_unified_json(json_path)

    parse_stdout_fallback(result.stdout + "\n" + result.stderr)

    dashboard_path = newest_file_after(
        OUTPUT_DIR,
        suffixes={".png", ".jpg", ".jpeg"},
        start_time=start_time,
        stem_hint=output_stem,
    )

    if dashboard_path is None:
        dashboard_path = newest_file_after(
            OUTPUT_DIR,
            suffixes={".png", ".jpg", ".jpeg"},
            start_time=start_time,
            stem_hint=None,
        )

    if dashboard_path is not None:
        timestamp = int(time.time())
        state["dashboard_url"] = f"/vocalcoach_outputs/{dashboard_path.name}?t={timestamp}"
    else:
        state["dashboard_url"] = None

    analysis_status_label.set_text("Full analysis complete.")
    return True


# ----------------------------
# Results UI
# ----------------------------

def show_results():
    results_area.clear()

    with results_area:
        ui.label("Performance Results").classes("text-5xl font-bold mt-10")

        ui.label(f"Song: {state['selected_song']}").classes(
            "text-2xl text-gray-400 mt-2"
        )

        if state["uploaded_audio_path"] is not None:
            ui.label(f"Analyzed file: {state['uploaded_audio_path'].name}").classes(
                "text-xl text-gray-500"
            )

        with ui.row().classes("gap-8 flex-wrap mt-8"):
            score_card(
                "Overall Score",
                state["overall_score"],
                get_level(state["overall_score"]),
                show_out_of_100=True,
                wide=True,
            )
            score_card("Pitch Score", state["pitch_score"], get_level(state["pitch_score"]))
            score_card("Timing Score", state["timing_score"], get_level(state["timing_score"]))
            score_card("Duration Score", state["duration_score"], get_level(state["duration_score"]))
            score_card("Lyrics Score", state["lyrics_score"], get_level(state["lyrics_score"]))

        if state["dashboard_url"] is not None:
            ui.label("Analysis Visualization").classes("text-4xl font-bold mt-12")
            ui.label(
                "Timeline view generated by the full vocal coaching pipeline."
            ).classes("text-xl text-gray-400 mt-2")

            ui.image(state["dashboard_url"]).classes(
                "w-full max-w-6xl mx-auto rounded-xl mt-4 shadow-lg"
            )

        ui.label("Coach Feedback").classes("text-4xl font-bold mt-12")

        feedback = state["feedback"]

        if not feedback:
            feedback = build_default_feedback()

        with ui.column().classes("w-full gap-4 mt-4"):
            for message in feedback:
                with ui.card().classes("w-full p-5 rounded-xl shadow-md"):
                    ui.label(message).classes("text-xl")

        with ui.expansion("Advanced Details", icon="analytics").classes("w-full mt-12"):
            ui.label(
                "Detailed backend metrics. Empty metrics are hidden from the main result view."
            ).classes("text-lg text-gray-400 mb-4")

            ui.label("Pitch Details").classes("text-3xl font-bold mt-4")

            with ui.grid(columns=2).classes("w-full gap-6 mt-4"):
                if has_value(state["pitch_accuracy"]):
                    score_metric(
                        "Pitch Accuracy",
                        state["pitch_accuracy"],
                        "Percentage of sung notes or frames close to the reference melody.",
                    )

                if has_value(state["note_pitch_acc"]):
                    score_metric(
                        "Note-Level Pitch Correctness",
                        state["note_pitch_acc"],
                        "Percentage of notes where the sung pitch matched the reference note.",
                    )

                if has_value(state["mace"]):
                    error_metric(
                        "Average Pitch Difference",
                        state["mace"],
                        good_value=25,
                        bad_value=200,
                        unit="cents",
                        description="Lower is better. 100 cents is one musical half-step, like C to C♯ or B to C.",
                    )

                if has_value(state["pitch_rmse"]):
                    error_metric(
                        "Pitch RMSE",
                        state["pitch_rmse"],
                        good_value=25,
                        bad_value=200,
                        unit="cents",
                        description="Lower is better. This penalizes larger pitch mistakes more strongly.",
                    )

            ui.label("Timing Details").classes("text-3xl font-bold mt-8")

            with ui.grid(columns=2).classes("w-full gap-6 mt-4"):
                if has_value(state["timing_accuracy"]):
                    score_metric(
                        "Timing Accuracy",
                        state["timing_accuracy"],
                        "Percentage of notes that started close to the reference timing.",
                    )

                if has_value(state["onset_mae_ms"]):
                    error_metric(
                        "Average Start Difference",
                        state["onset_mae_ms"],
                        good_value=25,
                        bad_value=200,
                        unit="ms",
                        description="Lower is better. This shows how early or late note starts were on average.",
                    )

                if has_value(state["offset_mae_ms"]):
                    error_metric(
                        "Average Ending Difference",
                        state["offset_mae_ms"],
                        good_value=35,
                        bad_value=250,
                        unit="ms",
                        description="Lower is better. This shows how early or late note endings were on average.",
                    )

                if has_value(state["ioi_mae_ms"]):
                    error_metric(
                        "Rhythm Consistency",
                        state["ioi_mae_ms"],
                        good_value=30,
                        bad_value=240,
                        unit="ms",
                        description="Lower is better. This measures consistency between note-to-note timing.",
                    )

            ui.label("Duration Details").classes("text-3xl font-bold mt-8")

            with ui.grid(columns=2).classes("w-full gap-6 mt-4"):
                score_metric(
                    "Duration Score",
                    state["duration_score"],
                    "Overall score for how well notes were held for the correct length.",
                )

                if has_value(state["duration_error_s"]):
                    error_metric(
                        "Duration Error",
                        state["duration_error_s"],
                        good_value=0.06,
                        bad_value=0.40,
                        unit="s",
                        description="Lower is better. This shows whether notes were held for the correct length.",
                    )

                if has_value(state["relative_duration_error"]):
                    error_metric(
                        "Relative Duration Error",
                        state["relative_duration_error"],
                        good_value=0.10,
                        bad_value=1.00,
                        unit="ratio",
                        description="Lower is better. This compares duration mistakes relative to note length.",
                    )

                if has_value(state["duration_ratio"]):
                    error_metric(
                        "Duration Ratio",
                        state["duration_ratio"],
                        good_value=1.00,
                        bad_value=2.00,
                        unit="ratio",
                        description="1.00x is ideal. Higher means notes were held too long; lower means they were cut short.",
                    )

            ui.label("Lyrics Details").classes("text-3xl font-bold mt-8")

            with ui.grid(columns=2).classes("w-full gap-6 mt-4"):
                score_metric(
                    "Lyrics Score",
                    state["lyrics_score"],
                    "Overall score for how well the lyric/phoneme timing matched the reference.",
                )

                if has_value(state["word_accuracy"]):
                    score_metric(
                        "Word Accuracy",
                        state["word_accuracy"],
                        "Percentage of reference words matched by the lyric/phoneme model.",
                    )

                if has_value(state["phoneme_overlap_accuracy"]):
                    score_metric(
                        "Phoneme Overlap Accuracy",
                        state["phoneme_overlap_accuracy"],
                        "Percentage of phoneme matches with enough temporal overlap.",
                    )

                if has_value(state["phoneme_boundary_error_ms"]):
                    error_metric(
                        "Phoneme Boundary Error",
                        state["phoneme_boundary_error_ms"],
                        good_value=15,
                        bad_value=120,
                        unit="ms",
                        description="Lower is better. This measures how close phoneme timing is to the TextGrid reference.",
                    )

        with ui.expansion("Debug Output", icon="terminal").classes("w-full mt-8"):
            ui.label("Resolved Paths").classes("text-xl font-bold")
            ui.label(
                f"PROJECT_DIR: {PROJECT_DIR}\n"
                f"BACKEND_DIR: {BACKEND_DIR}\n"
                f"SONG_REFERENCE_DIR: {SONG_REFERENCE_DIR}\n"
                f"MusicXML: {SONG_REGISTRY[state['selected_song']]['musicxml']}\n"
                f"TextGrid: {SONG_REGISTRY[state['selected_song']]['textgrid']}"
            ).classes("text-sm whitespace-pre-wrap")

            if state["unified_json_path"] is not None:
                ui.label("Unified JSON output").classes("text-xl font-bold mt-4")
                ui.label(str(state["unified_json_path"])).classes("text-sm whitespace-pre-wrap")

            if state["last_pipeline_stdout"]:
                ui.label("run_pipeline.py output").classes("text-xl font-bold mt-4")
                ui.label(state["last_pipeline_stdout"]).classes("text-sm whitespace-pre-wrap")

            if state["last_pipeline_stderr"]:
                ui.label("Log output").classes("text-xl font-bold mt-4")
                ui.label(state["last_pipeline_stderr"]).classes("text-sm whitespace-pre-wrap")

    ui.notify("Analysis complete")


def build_default_feedback():
    feedback = []

    if state["pitch_score"] is not None:
        feedback.append(f"Pitch performance is rated as {get_level(state['pitch_score']).lower()}.")

    if state["timing_score"] is not None:
        feedback.append(f"Timing performance is rated as {get_level(state['timing_score']).lower()}.")

    if state["duration_score"] is not None:
        feedback.append(f"Duration control is rated as {get_level(state['duration_score']).lower()}.")

    if state["lyrics_score"] is not None:
        feedback.append(f"Lyric timing is rated as {get_level(state['lyrics_score']).lower()}.")

    if state["onset_mae_ms"] is not None:
        feedback.append(
            f"Your note starts were about {format_ms(state['onset_mae_ms'])} away from the reference on average."
        )

    if state["offset_mae_ms"] is not None:
        feedback.append(
            f"Your note endings were about {format_ms(state['offset_mae_ms'])} away from the reference on average."
        )

    if state["phoneme_boundary_error_ms"] is not None:
        feedback.append(
            f"Your lyric/phoneme timing differed by about {format_ms(state['phoneme_boundary_error_ms'])} on average."
        )

    if not feedback:
        feedback.append("The full pipeline ran, but detailed feedback could not be parsed from the output yet.")

    return feedback


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
    show_loading("Starting full analysis...")

    await asyncio.sleep(0.1)

    try:
        success = await run_full_pipeline(uploaded_path)
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
    reset_analysis_outputs()
    results_area.clear()


# ----------------------------
# Page layout
# ----------------------------

ui.page_title("Vocal Coach AI")

with ui.row().classes("w-full justify-between items-center p-4"):
    ui.label("Vocal Coach AI").classes("text-6xl font-bold")
    ui.button("Toggle Theme", on_click=toggle_theme).classes("text-lg p-4")

ui.label(
    "Choose a song, upload your singing, and receive feedback on pitch, timing, duration, and lyrics."
).classes("text-2xl text-gray-400 px-4")


with ui.card().classes("w-full mt-8 p-8 rounded-2xl shadow-lg"):
    ui.label("Song Setup").classes("text-4xl font-bold")

    ui.select(
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
