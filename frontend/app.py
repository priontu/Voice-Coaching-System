from nicegui import ui, app
from pathlib import Path
import asyncio
import json
import os
import re
import subprocess
import sys
import time


# ============================================================
# Theme
# ============================================================

dark = ui.dark_mode()
dark.enable()


# ============================================================
# Paths / Config
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
BACKEND_DIR = PROJECT_DIR / "pipeline"

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BACKEND_DIR / "outputs"
SONG_REFERENCE_DIR = BACKEND_DIR / "song_references"

RUN_PIPELINE_SCRIPT = BACKEND_DIR / "inference" / "run_pipeline.py"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SONG_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

app.add_static_files("/vocalcoach_outputs", str(OUTPUT_DIR))
app.add_static_files("/song_references", str(SONG_REFERENCE_DIR))


SONG_REGISTRY = {
    "Test Song": {
        "musicxml": SONG_REFERENCE_DIR / "test.musicxml",
        "textgrid": SONG_REFERENCE_DIR / "test.TextGrid",
        "audio": SONG_REFERENCE_DIR / "test.wav",
        "description": "Default test reference song.",
    },
    "Rolling in the Deep": {
        "musicxml": SONG_REFERENCE_DIR / "song_1.musicxml",
        "textgrid": SONG_REFERENCE_DIR / "song_1.TextGrid",
        "audio": SONG_REFERENCE_DIR / "song_1.wav",
        "description": "Reference song: Rolling in the Deep.",
    },
    "I Knew You Were Trouble": {
        "musicxml": SONG_REFERENCE_DIR / "song_2.musicxml",
        "textgrid": SONG_REFERENCE_DIR / "song_2.TextGrid",
        "audio": SONG_REFERENCE_DIR / "song_2.wav",
        "description": "Reference song: I Knew You Were Trouble.",
    },
    "Enchanted": {
        "musicxml": SONG_REFERENCE_DIR / "song_3.musicxml",
        "textgrid": SONG_REFERENCE_DIR / "song_3.TextGrid",
        "audio": SONG_REFERENCE_DIR / "song_3.wav",
        "description": "Reference song: Enchanted.",
    },
    "All I Ask": {
        "musicxml": SONG_REFERENCE_DIR / "song_4.musicxml",
        "textgrid": SONG_REFERENCE_DIR / "song_4.TextGrid",
        "audio": SONG_REFERENCE_DIR / "song_4.wav",
        "description": "Reference song: All I Ask.",
    },
    "Someone Like You": {
        "musicxml": SONG_REFERENCE_DIR / "song_5.musicxml",
        "textgrid": SONG_REFERENCE_DIR / "song_5.TextGrid",
        "audio": SONG_REFERENCE_DIR / "song_5.wav",
        "description": "Reference song: Someone Like You.",
    },
}


def get_song_options():
    return sorted(SONG_REGISTRY.keys())


# ============================================================
# State
# ============================================================

SCORE_KEYS = [
    "overall_score",
    "pitch_score",
    "timing_score",
    "duration_score",
    "lyrics_score",
]

DETAIL_KEYS = [
    "pitch_accuracy",
    "mace",
    "pitch_rmse",
    "note_pitch_acc",
    "timing_accuracy",
    "onset_mae_ms",
    "offset_mae_ms",
    "ioi_mae_ms",
    "duration_error_s",
    "relative_duration_error",
    "duration_ratio",
    "word_accuracy",
    "phoneme_overlap_accuracy",
    "label_match_rate",
    "phoneme_boundary_error_ms",
]

state = {
    "selected_song": "Test Song",
    "uploaded_file_ready": False,
    "uploaded_audio_path": None,
    "uploaded_original_name": None,

    "is_analyzing": False,
    "analyze_button": None,
    "cancel_button": None,
    "current_process": None,
    "cancel_requested": False,

    "upload_box_area": None,
    "reference_preview_area": None,

    "current_output_stem": None,
    "unified_json_path": None,
    "dashboard_url": None,
    "last_pipeline_stdout": "",
    "last_pipeline_stderr": "",
    "feedback": [],
}

for key in SCORE_KEYS + DETAIL_KEYS:
    state[key] = None


def reset_analysis_outputs():
    state["current_output_stem"] = None
    state["unified_json_path"] = None
    state["dashboard_url"] = None
    state["last_pipeline_stdout"] = ""
    state["last_pipeline_stderr"] = ""
    state["feedback"] = []

    for key in SCORE_KEYS + DETAIL_KEYS:
        state[key] = None


# ============================================================
# Formatting Helpers
# ============================================================

def has_value(value):
    return value is not None


def clamp(value, low=0, high=100):
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return 0


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


def safe_score(value):
    value = normalize_score(value)
    return 0 if value is None else clamp(value)


def piecewise_score(value, x_points, s_points):
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if value <= x_points[0]:
        return float(s_points[0])

    if value >= x_points[-1]:
        return float(s_points[-1])

    for i in range(len(x_points) - 1):
        x0 = x_points[i]
        x1 = x_points[i + 1]
        s0 = s_points[i]
        s1 = s_points[i + 1]

        if x0 <= value <= x1:
            ratio = (value - x0) / (x1 - x0)
            return float(s0 + ratio * (s1 - s0))

    return None


def score_color(score):
    score = safe_score(score)

    if score >= 90:
        return "#4CAF50"
    if score >= 75:
        return "#8BC34A"
    if score >= 55:
        return "#FFC107"
    return "#F44336"


def component_color(score):
    score = safe_score(score)

    if score >= 75:
        return "#2196F3"

    return "#FF5722"


def score_card_bg_class(score):
    score = safe_score(score)

    if score >= 75:
        return "bg-green-600"
    if score >= 55:
        return "bg-yellow-500"
    return "bg-red-600"


def format_score(value):
    value = normalize_score(value)
    return "N/A" if value is None else f"{value:.1f}"


def format_percent(value):
    value = normalize_score(value)
    return "N/A" if value is None else f"{value:.1f}%"


def format_ms(value):
    try:
        return f"{float(value):.1f} ms"
    except (TypeError, ValueError):
        return "N/A"


def format_seconds(value):
    try:
        return f"{float(value):.3f} s"
    except (TypeError, ValueError):
        return "N/A"


def format_cents(value):
    try:
        return f"{float(value):.1f} cents"
    except (TypeError, ValueError):
        return "N/A"


def format_ratio(value):
    try:
        return f"{float(value):.2f}x"
    except (TypeError, ValueError):
        return "N/A"


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


def get_selected_song_description():
    song = SONG_REGISTRY.get(state["selected_song"], {})
    return song.get("description", "")


def get_selected_song_audio_path():
    song = SONG_REGISTRY.get(state["selected_song"], {})
    return song.get("audio")


def get_selected_song_audio_url():
    audio_path = get_selected_song_audio_path()

    if audio_path is None or not audio_path.exists():
        return None

    return f"/song_references/{audio_path.name}?t={int(time.time())}"


def toggle_theme():
    dark.toggle()

    if state["overall_score"] is not None:
        show_results()

    render_reference_preview()


# ============================================================
# JSON / STDOUT Parsing
# ============================================================

def find_key_recursive(data, possible_keys):
    possible_keys = {key.lower() for key in possible_keys}

    if isinstance(data, dict):
        for key, value in data.items():
            if str(key).lower() in possible_keys:
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
    category_name = category_name.lower()

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


def parse_output_path_from_stdout(text, label):
    pattern = rf"{label}:\s+(.+)"
    match = re.search(pattern, text, re.IGNORECASE)

    if not match:
        return None

    path_text = match.group(1).strip()

    if not path_text:
        return None

    path = Path(path_text)

    if path.exists():
        return path

    return None


def parse_unified_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    state["overall_score"] = find_key_recursive(
        data,
        ["overall_score", "final_score", "performance_score", "total_score", "score_final"],
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
        or find_key_recursive(data, ["duration_score", "duration_control_score", "score_duration"])
    )

    state["lyrics_score"] = (
        find_score_by_category(data, "lyric")
        or find_score_by_category(data, "lyrics")
        or find_key_recursive(data, ["lyric_score", "lyrics_score", "lyric_timing_score", "score_lyric"])
    )

    metric_key_map = {
        "pitch_accuracy": ["pitch_accuracy", "pitch_acc", "pitch_acc_50", "pitchacc50", "PitchAcc50"],
        "mace": ["mace", "mace_cents", "mean_absolute_cent_error", "mean_abs_cent_error"],
        "pitch_rmse": ["pitch_rmse", "pitch_rmse_cents", "rmse_cents", "PitchRMSE"],
        "note_pitch_acc": ["note_pitch_accuracy", "note_pitch_acc", "note_pitch_acc_50", "NotePitchAcc50"],
        "timing_accuracy": ["timing_accuracy", "timing_acc", "timing_acc_50", "TimingAcc50"],
        "onset_mae_ms": ["onset_mae_ms", "mean_abs_onset_error_ms", "mean_absolute_onset_error_ms", "onset_error_mae_ms"],
        "offset_mae_ms": ["offset_mae_ms", "mean_abs_offset_error_ms", "mean_absolute_offset_error_ms", "offset_error_mae_ms"],
        "ioi_mae_ms": ["ioi_mae_ms", "ioi_mean_absolute_error_ms", "ioi_error_ms"],
        "duration_error_s": ["duration_error", "duration_error_s", "duration_mae_s", "mean_duration_error", "duration_mae"],
        "relative_duration_error": ["relative_duration_error", "rel_duration_error", "rel_dur_err", "RelDurErr"],
        "duration_ratio": ["duration_ratio", "dur_ratio", "DurRatio"],
        "word_accuracy": ["word_accuracy", "word_acc", "WordAcc"],
        "phoneme_overlap_accuracy": ["phoneme_overlap_accuracy", "overlap_accuracy", "overlap_acc", "OverlapAcc"],
        "label_match_rate": ["label_match_rate", "label_match", "LabelMatch"],
        "phoneme_boundary_error_ms": ["phoneme_boundary_error_ms", "phoneme_boundary_mae_ms", "phon_bnd_err_ms", "PhonBndErr"],
    }

    for state_key, possible_keys in metric_key_map.items():
        state[state_key] = find_key_recursive(data, possible_keys)

    feedback = find_key_recursive(
        data,
        ["feedback", "interpretation", "interpretation_summary", "comments", "messages"],
    )

    if isinstance(feedback, list):
        state["feedback"] = [str(item) for item in feedback]
    elif isinstance(feedback, dict):
        collected = []

        for key in ["strengths", "weaknesses", "suggestions", "messages"]:
            value = feedback.get(key)

            if isinstance(value, list):
                collected.extend(str(item) for item in value)
            elif isinstance(value, str):
                collected.append(value)

        state["feedback"] = collected
    elif isinstance(feedback, str):
        state["feedback"] = [feedback]


def parse_stdout_fallback(text: str):
    score_patterns = {
        "overall_score": r"Score:\s*([\d.]+)\s*/\s*100",
        "pitch_score": r"pitch\s*=\s*(-?[\d.]+)",
        "timing_score": r"timing\s*=\s*(-?[\d.]+)",
        "duration_score": r"duration\s*=\s*(-?[\d.]+)",
        "lyrics_score": r"lyric\s*=\s*(-?[\d.]+)",
    }

    for state_key, pattern in score_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            state[state_key] = float(match.group(1))

    pitch_acc_match = re.search(r"Pitch\s+acc:\s*([\d.]+)%", text, re.IGNORECASE)
    if pitch_acc_match:
        state["pitch_accuracy"] = float(pitch_acc_match.group(1))

    mace_match = re.search(r"MACE\s*([\d.]+)", text, re.IGNORECASE)
    if mace_match:
        state["mace"] = float(mace_match.group(1))

    rmse_match = re.search(r"RMSE\s*([\d.]+)", text, re.IGNORECASE)
    if rmse_match:
        state["pitch_rmse"] = float(rmse_match.group(1))

    timing_match = re.search(
        r"Timing:\s*([\d.]+)%\s*MAE\s*([\d.]+)\s*ms",
        text,
        re.IGNORECASE,
    )
    if timing_match:
        state["timing_accuracy"] = float(timing_match.group(1))
        state["onset_mae_ms"] = float(timing_match.group(2))

    feedback = []

    strength_match = re.search(r"Strengths:\s*(.+)", text, re.IGNORECASE)
    if strength_match:
        feedback.append(f"Strength: {strength_match.group(1).strip()}")

    improve_match = re.search(r"Improve:\s*(.+)", text, re.IGNORECASE)
    if improve_match:
        feedback.append(f"Focus area: {improve_match.group(1).strip()}")

    if feedback:
        state["feedback"] = feedback


# ============================================================
# Component Calculations
# ============================================================

def pitch_component_values():
    components = []

    if state["pitch_accuracy"] is not None:
        components.append(("accuracy", round(safe_score(state["pitch_accuracy"]), 1)))

    if state["mace"] is not None:
        intonation_score = piecewise_score(
            state["mace"],
            x_points=[0, 25, 50, 100, 200],
            s_points=[100, 88, 75, 50, 0],
        )

        if intonation_score is not None:
            components.append(("intonation", round(intonation_score, 1)))

    if state["pitch_rmse"] is not None:
        stability_score = piecewise_score(
            state["pitch_rmse"],
            x_points=[0, 25, 50, 100, 200],
            s_points=[100, 88, 72, 45, 0],
        )

        if stability_score is not None:
            components.append(("stability", round(stability_score, 1)))

    return components


def timing_component_values():
    components = []

    if state["timing_accuracy"] is not None:
        components.append(("accuracy", round(safe_score(state["timing_accuracy"]), 1)))

    if state["onset_mae_ms"] is not None:
        onset_score = piecewise_score(
            state["onset_mae_ms"],
            x_points=[0, 25, 50, 100, 200],
            s_points=[100, 88, 75, 50, 0],
        )

        if onset_score is not None:
            components.append(("onset_mae", round(onset_score, 1)))

    if state["timing_score"] is not None:
        components.append(("rhythm_stability", round(safe_score(state["timing_score"]), 1)))

    return components


# ============================================================
# File Discovery
# ============================================================

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

    if not candidates and stem_hint is not None:
        return newest_file_after(directory, suffixes, start_time, stem_hint=None)

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


# ============================================================
# UI Components
# ============================================================

def score_card(title, value, subtitle="", show_out_of_100=False, highlight=False):
    if highlight:
        bg_class = score_card_bg_class(value)

        with ui.card().classes(
            f"w-[560px] h-56 p-8 rounded-3xl shadow-2xl justify-center {bg_class}"
        ):
            ui.label(title).classes("text-3xl font-bold text-white opacity-95")

            if show_out_of_100 and value is not None:
                score_text = f"{format_score(value)} / 100"
            else:
                score_text = format_score(value)

            ui.label(score_text).classes("text-8xl font-black text-white mt-2")

            if subtitle:
                ui.label(subtitle).classes("text-xl font-semibold text-white opacity-95 mt-3")

        return

    with ui.card().classes("w-72 h-40 p-6 rounded-2xl shadow-lg justify-center"):
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


def category_score_chart(height="h-[560px]"):
    categories = [
        ("Pitch", safe_score(state["pitch_score"])),
        ("Timing", safe_score(state["timing_score"])),
        ("Duration", safe_score(state["duration_score"])),
        ("Lyrics", safe_score(state["lyrics_score"])),
    ]

    is_dark = bool(dark.value)

    text_color = "#F8FAFC" if is_dark else "#111827"
    grid_color = "#64748B" if is_dark else "#D1D5DB"

    options = {
        "backgroundColor": "transparent",
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "shadow"},
            "backgroundColor": "#111827" if is_dark else "#FFFFFF",
            "borderColor": "#64748B" if is_dark else "#D1D5DB",
            "textStyle": {"color": text_color, "fontSize": 16},
            "formatter": "{b}: {c} / 100",
        },
        "grid": {
            "left": "8%",
            "right": "8%",
            "top": "12%",
            "bottom": "16%",
            "containLabel": True,
        },
        "xAxis": {
            "type": "category",
            "data": [name for name, _ in categories],
            "axisLabel": {"color": text_color, "fontSize": 20, "fontWeight": "bold"},
            "axisLine": {"lineStyle": {"color": grid_color, "width": 2}},
            "axisTick": {"show": False},
        },
        "yAxis": {
            "type": "value",
            "min": 0,
            "max": 100,
            "axisLabel": {"color": text_color, "fontSize": 16},
            "splitLine": {
                "lineStyle": {
                    "color": grid_color,
                    "type": "dashed",
                    "opacity": 0.45,
                }
            },
        },
        "series": [
            {
                "name": "Category Score",
                "type": "bar",
                "data": [
                    {
                        "value": score,
                        "itemStyle": {
                            "color": score_color(score),
                            "borderRadius": [14, 14, 0, 0],
                        },
                    }
                    for _, score in categories
                ],
                "barWidth": "46%",
                "label": {
                    "show": True,
                    "position": "top",
                    "color": text_color,
                    "fontSize": 20,
                    "fontWeight": "bold",
                    "formatter": "{c}",
                },
                "markLine": {
                    "symbol": "none",
                    "label": {
                        "show": True,
                        "color": text_color,
                        "fontSize": 14,
                        "fontWeight": "bold",
                        "formatter": "{b}",
                    },
                    "lineStyle": {"type": "dotted", "width": 3},
                    "data": [
                        {
                            "name": "Good",
                            "yAxis": 75,
                            "lineStyle": {"color": "#FFC107"},
                        },
                        {
                            "name": "Excellent",
                            "yAxis": 90,
                            "lineStyle": {"color": "#4CAF50"},
                        },
                    ],
                },
            }
        ],
    }

    ui.echart(options).classes(f"w-full {height}")


def radar_chart(height="h-[560px]"):
    pitch = safe_score(state["pitch_score"])
    timing = safe_score(state["timing_score"])
    duration = safe_score(state["duration_score"])
    lyrics = safe_score(state["lyrics_score"])

    is_dark = bool(dark.value)

    text_color = "#FFFFFF" if is_dark else "#111827"
    split_line_color = "#E5E7EB" if is_dark else "#9CA3AF"

    split_area_colors = (
        ["rgba(229, 231, 235, 0.22)", "rgba(229, 231, 235, 0.10)"]
        if is_dark
        else ["rgba(59, 130, 246, 0.10)", "rgba(59, 130, 246, 0.04)"]
    )

    radar_line_color = "#38BDF8" if is_dark else "#2563EB"
    radar_fill_color = (
        "rgba(56, 189, 248, 0.45)"
        if is_dark
        else "rgba(37, 99, 235, 0.25)"
    )

    options = {
        "backgroundColor": "transparent",
        "tooltip": {
            "trigger": "item",
            "backgroundColor": "#111827" if is_dark else "#FFFFFF",
            "borderColor": "#475569" if is_dark else "#D1D5DB",
            "textStyle": {"color": text_color, "fontSize": 16},
        },
        "radar": {
            "center": ["50%", "52%"],
            "radius": "74%",
            "indicator": [
                {"name": "Pitch", "max": 100},
                {"name": "Timing", "max": 100},
                {"name": "Duration", "max": 100},
                {"name": "Lyrics", "max": 100},
            ],
            "axisName": {
                "color": text_color,
                "fontSize": 22,
                "fontWeight": "bold",
                "padding": [10, 10, 10, 10],
            },
            "axisLine": {"lineStyle": {"color": split_line_color, "width": 2}},
            "splitLine": {"lineStyle": {"color": split_line_color, "width": 2}},
            "splitArea": {"areaStyle": {"color": split_area_colors}},
            "splitNumber": 5,
        },
        "series": [
            {
                "name": "Performance Profile",
                "type": "radar",
                "data": [
                    {
                        "value": [pitch, timing, duration, lyrics],
                        "name": (
                            f"Pitch: {pitch:.1f}<br/>"
                            f"Timing: {timing:.1f}<br/>"
                            f"Duration: {duration:.1f}<br/>"
                            f"Lyrics: {lyrics:.1f}"
                        ),
                        "areaStyle": {"color": radar_fill_color, "opacity": 0.65},
                        "lineStyle": {"color": radar_line_color, "width": 5},
                        "itemStyle": {
                            "color": radar_line_color,
                            "borderColor": text_color,
                            "borderWidth": 2,
                        },
                        "symbolSize": 12,
                    }
                ],
            }
        ],
    }

    ui.echart(options).classes(f"w-full {height}")


def component_score_chart(title, components, height=None):
    is_dark = bool(dark.value)

    text_color = "#F8FAFC" if is_dark else "#111827"
    grid_color = "#94A3B8" if is_dark else "#444444"
    tooltip_bg = "#111827" if is_dark else "#FFFFFF"
    tooltip_border = "#64748B" if is_dark else "#D1D5DB"

    names = [name for name, _ in components]
    scores = [safe_score(score) for _, score in components]

    chart_height = height or max(360, 130 + len(components) * 95)

    options = {
        "backgroundColor": "transparent",
        "title": {
            "text": title,
            "left": "center",
            "top": 8,
            "textStyle": {
                "color": text_color,
                "fontSize": 20,
                "fontWeight": "normal",
            },
        },
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "shadow"},
            "backgroundColor": tooltip_bg,
            "borderColor": tooltip_border,
            "textStyle": {
                "color": text_color,
                "fontSize": 14,
            },
            "formatter": "{b}: {c}",
        },
        "grid": {
            "left": "18%",
            "right": "10%",
            "top": "18%",
            "bottom": "14%",
            "containLabel": True,
        },
        "xAxis": {
            "type": "value",
            "name": "Score",
            "nameLocation": "middle",
            "nameGap": 32,
            "min": 0,
            "max": 100,
            "axisLabel": {
                "color": text_color,
                "fontSize": 15,
            },
            "nameTextStyle": {
                "color": text_color,
                "fontSize": 18,
            },
            "axisLine": {
                "show": True,
                "lineStyle": {
                    "color": grid_color,
                    "width": 2,
                },
            },
            "splitLine": {"show": False},
        },
        "yAxis": {
            "type": "category",
            "data": names,
            "axisLabel": {
                "color": text_color,
                "fontSize": 15,
            },
            "axisLine": {
                "show": True,
                "lineStyle": {
                    "color": grid_color,
                    "width": 2,
                },
            },
            "axisTick": {"show": False},
        },
        "series": [
            {
                "name": title,
                "type": "bar",
                "data": [
                    {
                        "value": score,
                        "itemStyle": {
                            "color": component_color(score),
                            "borderRadius": [0, 8, 8, 0],
                        },
                    }
                    for score in scores
                ],
                "barWidth": "54%",
                "label": {
                    "show": True,
                    "position": "right",
                    "color": text_color,
                    "fontSize": 14,
                    "formatter": "{c}",
                },
            }
        ],
    }

    ui.echart(options).classes(f"w-full h-[{chart_height}px]")


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


def set_cancel_button_visible(visible):
    button = state.get("cancel_button")

    if button is None:
        return

    button.set_visibility(visible)


def render_reference_preview():
    reference_preview_area = state.get("reference_preview_area")

    if reference_preview_area is None:
        return

    reference_preview_area.clear()
    audio_url = get_selected_song_audio_url()

    with reference_preview_area:
        ui.label("Reference Preview").classes("text-2xl font-bold mt-4")

        if audio_url is None:
            ui.label(
                "No reference audio found yet. Add the matching .wav file to song_references."
            ).classes("text-lg text-yellow-500 mt-1")
            return

        ui.audio(audio_url).classes("w-[480px] mt-2")
        ui.label(
            "Use this only to confirm the selected song/version before uploading your singing."
        ).classes("text-sm text-gray-500 mt-1")


def render_upload_box():
    upload_box_area = state.get("upload_box_area")

    if upload_box_area is None:
        return

    upload_box_area.clear()

    uploaded_path = state.get("uploaded_audio_path")
    original_name = state.get("uploaded_original_name")
    show_uploaded_file = state["uploaded_file_ready"] and uploaded_path is not None

    with upload_box_area:
        if show_uploaded_file:
            display_name = original_name or uploaded_path.name

            with ui.card().classes(
                "w-[900px] h-[360px] mt-12 p-8 items-center justify-center "
                "border-4 border-dashed border-gray-500 rounded-2xl opacity-90"
            ):
                ui.icon("audio_file").classes("text-7xl text-blue-400")
                ui.label(display_name).classes(
                    "text-4xl font-bold text-center mt-4"
                )
                ui.label("File uploaded").classes(
                    "text-2xl text-green-400 font-semibold mt-2"
                )

                if state["is_analyzing"]:
                    ui.label("Analysis is running. Upload is locked.").classes(
                        "text-lg text-yellow-500 mt-4 font-semibold"
                    )
                else:
                    ui.label("Remove the current file before uploading another one.").classes(
                        "text-lg text-yellow-500 mt-4 font-semibold"
                    )

            return

        upload = ui.upload(
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

        if state["is_analyzing"]:
            upload.disable()
            ui.label("Analysis is running. Upload is locked.").classes(
                "text-lg text-yellow-500 mt-4 font-semibold"
            )
        else:
            ui.label(
                "After the file uploads, its name should appear below."
            ).classes("text-lg text-gray-500 mt-4")


# ============================================================
# Backend Runner
# ============================================================

async def run_command(command, cwd):
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PYTHONPATH": str(BACKEND_DIR) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }

    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    state["current_process"] = process

    try:
        stdout, stderr = await asyncio.to_thread(process.communicate)

        return subprocess.CompletedProcess(
            args=command,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    finally:
        state["current_process"] = None


def get_selected_references():
    selected_song = state["selected_song"]

    if selected_song not in SONG_REGISTRY:
        raise FileNotFoundError("Selected song is not registered.")

    musicxml_path = SONG_REGISTRY[selected_song]["musicxml"]
    textgrid_path = SONG_REGISTRY[selected_song]["textgrid"]

    print("PROJECT_DIR:", PROJECT_DIR)
    print("BACKEND_DIR:", BACKEND_DIR)
    print("SONG_REFERENCE_DIR:", SONG_REFERENCE_DIR)
    print("Selected song:", selected_song)
    print("MusicXML path:", musicxml_path)
    print("MusicXML exists:", musicxml_path.exists())
    print("TextGrid path:", textgrid_path)
    print("TextGrid exists:", textgrid_path.exists())

    if not musicxml_path.exists():
        raise FileNotFoundError(f"Missing MusicXML: {musicxml_path}")

    if not textgrid_path.exists():
        raise FileNotFoundError(f"Missing TextGrid: {textgrid_path}")

    return musicxml_path.resolve(), textgrid_path.resolve()


async def run_full_pipeline(uploaded_path: Path) -> bool:
    analysis_status_label.set_text("Running full vocal coaching pipeline...")
    show_loading("Running pitch, timing, duration, and lyric analysis...")

    state["last_pipeline_stdout"] = ""
    state["last_pipeline_stderr"] = ""

    if not BACKEND_DIR.exists():
        ui.notify(f"Missing folder: {BACKEND_DIR}", color="negative")
        return False

    if not RUN_PIPELINE_SCRIPT.exists():
        ui.notify(f"Missing file: {RUN_PIPELINE_SCRIPT}", color="negative")
        return False

    try:
        musicxml_path, textgrid_path = get_selected_references()
    except FileNotFoundError as error:
        analysis_status_label.set_text(str(error))
        ui.notify(str(error), color="negative")
        return False

    uploaded_path = uploaded_path.resolve()
    output_stem = uploaded_path.stem
    state["current_output_stem"] = output_stem

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

    combined_output = result.stdout + "\n" + result.stderr

    state["last_pipeline_stdout"] = result.stdout
    state["last_pipeline_stderr"] = result.stderr

    print("Full pipeline return code:", result.returncode)
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)

    if state["cancel_requested"]:
        analysis_status_label.set_text("Analysis cancelled.")
        return False

    if result.returncode != 0:
        analysis_status_label.set_text("Full pipeline failed.")
        ui.notify("Full pipeline failed. Check terminal output.", color="negative")
        return False

    json_path = parse_output_path_from_stdout(combined_output, "JSON")

    if json_path is None:
        json_path = newest_file_after(
            OUTPUT_DIR,
            suffixes={".json"},
            start_time=start_time,
            stem_hint=output_stem,
        )

    if json_path is not None:
        state["unified_json_path"] = json_path
        parse_unified_json(json_path)
    else:
        ui.notify("Could not find pipeline JSON output.", color="warning")

    parse_stdout_fallback(combined_output)

    dashboard_path = parse_output_path_from_stdout(combined_output, "Plot")

    if dashboard_path is None:
        dashboard_path = newest_file_after(
            OUTPUT_DIR,
            suffixes={".png", ".jpg", ".jpeg"},
            start_time=start_time,
            stem_hint=output_stem,
        )

    if dashboard_path is not None:
        timestamp = int(time.time())
        state["dashboard_url"] = f"/vocalcoach_outputs/{dashboard_path.name}?t={timestamp}"
    else:
        state["dashboard_url"] = None

    analysis_status_label.set_text("Full analysis complete.")
    return True


# ============================================================
# Results Rendering
# ============================================================

def render_result_header():
    ui.label("Performance Results").classes("text-5xl font-bold mt-10")

    ui.label(f"Song: {state['selected_song']}").classes(
        "text-2xl text-gray-400 mt-2"
    )

    description = get_selected_song_description()
    if description:
        ui.label(description).classes("text-lg text-gray-500 mt-1")

    if state["uploaded_audio_path"] is not None:
        display_name = state.get("uploaded_original_name") or state["uploaded_audio_path"].name
        ui.label(f"Analyzed file: {display_name}").classes(
            "text-xl text-gray-500"
        )


def render_score_cards():
    with ui.column().classes("w-full items-center gap-6 mt-8"):
        with ui.row().classes("w-full justify-center"):
            score_card(
                "Overall Score",
                state["overall_score"],
                get_level(state["overall_score"]),
                show_out_of_100=True,
                highlight=True,
            )

        with ui.row().classes("w-full justify-center items-center gap-8 flex-wrap"):
            score_card("Pitch Score", state["pitch_score"], get_level(state["pitch_score"]))
            score_card("Timing Score", state["timing_score"], get_level(state["timing_score"]))
            score_card("Duration Score", state["duration_score"], get_level(state["duration_score"]))
            score_card("Lyrics Score", state["lyrics_score"], get_level(state["lyrics_score"]))


def render_visual_dashboard_grid():
    pitch_components = pitch_component_values()
    timing_components = timing_component_values()

    ui.label("Performance Score Dashboard").classes(
        "text-4xl font-bold mt-12 text-center w-full"
    )

    ui.label(
        "Visual summary of overall balance, category scores, pitch components, and timing components."
    ).classes("text-xl text-gray-400 mt-2 text-center w-full")

    with ui.grid(columns=2).classes("w-full gap-8 mt-6"):
        with ui.card().classes("w-full p-6 rounded-2xl shadow-lg"):
            ui.label("Performance Profile").classes("text-3xl font-bold text-center")
            ui.label(
                "Balance between pitch, timing, duration, and lyrics."
            ).classes("text-lg text-gray-400 text-center mt-1")
            radar_chart(height="h-[500px]")

        with ui.card().classes("w-full p-6 rounded-2xl shadow-lg"):
            ui.label("Category Scores").classes("text-3xl font-bold text-center")
            ui.label(
                "Score comparison for each main evaluation category."
            ).classes("text-lg text-gray-400 text-center mt-1")
            category_score_chart(height="h-[500px]")

        with ui.card().classes("w-full p-6 rounded-2xl shadow-lg"):
            ui.label("Pitch Components").classes("text-3xl font-bold text-center")
            ui.label(
                "Recreated from pitch accuracy, MACE, and RMSE when available."
            ).classes("text-lg text-gray-400 text-center mt-1")

            if pitch_components:
                component_score_chart("Pitch components", pitch_components, height=420)
            else:
                ui.label("Pitch component metrics were not available.").classes(
                    "text-xl font-bold text-center mt-12"
                )

        with ui.card().classes("w-full p-6 rounded-2xl shadow-lg"):
            ui.label("Timing Components").classes("text-3xl font-bold text-center")
            ui.label(
                "Recreated from timing accuracy, onset MAE, and timing score."
            ).classes("text-lg text-gray-400 text-center mt-1")

            if timing_components:
                component_score_chart("Timing components", timing_components, height=420)
            else:
                ui.label("Timing component metrics were not available.").classes(
                    "text-xl font-bold text-center mt-12"
                )


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

    if not feedback:
        feedback.append("The full pipeline ran, but detailed feedback could not be parsed from the output yet.")

    return feedback


def render_feedback():
    ui.label("Coach Feedback").classes("text-4xl font-bold mt-12")

    feedback = state["feedback"] or build_default_feedback()

    with ui.column().classes("w-full gap-4 mt-4"):
        for message in feedback:
            with ui.card().classes("w-full p-5 rounded-xl shadow-md"):
                ui.label(message).classes("text-xl")


def render_backend_visualization():
    if state["dashboard_url"] is None:
        return

    ui.label("Backend Pipeline Visualization").classes("text-3xl font-bold mt-8")
    ui.label(
        "Static dashboard image generated directly by the backend pipeline."
    ).classes("text-lg text-gray-400 mt-1")

    ui.image(state["dashboard_url"]).classes(
        "w-full max-w-6xl mx-auto rounded-xl mt-4 shadow-lg"
    )


def render_advanced_details():
    with ui.expansion("Advanced Details", icon="analytics").classes("w-full mt-12"):
        ui.label(
            "Detailed backend metrics. Empty metrics are hidden from the main result view."
        ).classes("text-lg text-gray-400 mb-4")

        render_pitch_details()
        render_timing_details()
        render_duration_details()
        render_lyrics_details()
        render_backend_visualization()


def render_pitch_details():
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


def render_timing_details():
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


def render_duration_details():
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


def render_lyrics_details():
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


def render_debug_output():
    with ui.expansion("Debug Output", icon="terminal").classes("w-full mt-8"):
        ui.label("Resolved Paths").classes("text-xl font-bold")

        ui.label(
            f"PROJECT_DIR: {PROJECT_DIR}\n"
            f"BACKEND_DIR: {BACKEND_DIR}\n"
            f"SONG_REFERENCE_DIR: {SONG_REFERENCE_DIR}\n"
            f"Selected song: {state['selected_song']}\n"
            f"MusicXML: {SONG_REGISTRY[state['selected_song']]['musicxml']}\n"
            f"TextGrid: {SONG_REGISTRY[state['selected_song']]['textgrid']}\n"
            f"Audio: {SONG_REGISTRY[state['selected_song']]['audio']}"
        ).classes("text-sm whitespace-pre-wrap")

        if state["unified_json_path"] is not None:
            ui.label("Unified JSON output").classes("text-xl font-bold mt-4")
            ui.label(str(state["unified_json_path"])).classes("text-sm whitespace-pre-wrap")

        pitch_components = pitch_component_values()
        if pitch_components:
            ui.label("Calculated pitch components").classes("text-xl font-bold mt-4")
            ui.label(str(pitch_components)).classes("text-sm whitespace-pre-wrap")

        timing_components = timing_component_values()
        if timing_components:
            ui.label("Calculated timing components").classes("text-xl font-bold mt-4")
            ui.label(str(timing_components)).classes("text-sm whitespace-pre-wrap")

        if state["last_pipeline_stdout"]:
            ui.label("run_pipeline.py output").classes("text-xl font-bold mt-4")
            ui.label(state["last_pipeline_stdout"]).classes("text-sm whitespace-pre-wrap")

        if state["last_pipeline_stderr"]:
            ui.label("Log output").classes("text-xl font-bold mt-4")
            ui.label(state["last_pipeline_stderr"]).classes("text-sm whitespace-pre-wrap")


def show_results():
    results_area.clear()

    with results_area:
        render_result_header()
        render_score_cards()
        render_visual_dashboard_grid()
        render_feedback()
        render_advanced_details()
        render_debug_output()

    ui.notify("Analysis complete")


# ============================================================
# App Actions
# ============================================================

def cancel_analysis():
    process = state.get("current_process")

    state["cancel_requested"] = True
    analysis_status_label.set_text("Cancelling analysis...")

    if process is not None and process.poll() is None:
        try:
            process.terminate()
            ui.notify("Cancelling analysis...", color="warning")
        except Exception as error:
            ui.notify(f"Could not cancel process: {error}", color="negative")
            print("Cancel failed:", error)
    else:
        ui.notify("No active analysis process to cancel.", color="warning")


async def analyze_singing():
    if state["is_analyzing"]:
        ui.notify("Analysis is already running.", color="warning")
        return

    uploaded_path = state["uploaded_audio_path"]

    print("Analyze clicked")
    print("Selected song:", state["selected_song"])
    print("Uploaded ready:", state["uploaded_file_ready"])
    print("Uploaded path:", uploaded_path)

    if not state["uploaded_file_ready"] or uploaded_path is None:
        ui.notify("Please upload one audio file first.", color="negative")
        return

    if not uploaded_path.exists():
        ui.notify("The uploaded file was not found on disk.", color="negative")
        upload_status_label.set_text("Uploaded file is missing.")
        upload_path_label.set_text(f"Missing path: {uploaded_path}")
        return

    state["is_analyzing"] = True
    state["cancel_requested"] = False

    set_analyze_button_enabled(False)
    set_cancel_button_visible(True)
    render_upload_box()

    results_area.clear()
    upload_status_label.set_text(f"Ready to analyze: {state.get('uploaded_original_name') or uploaded_path.name}")
    show_loading("Starting full analysis...")

    await asyncio.sleep(0.1)

    try:
        success = await run_full_pipeline(uploaded_path)

        if state["cancel_requested"]:
            hide_loading()
            results_area.clear()
            analysis_status_label.set_text("Analysis cancelled. You can try again.")
            ui.notify("Analysis cancelled.", color="warning")
            return

        if success:
            hide_loading()
            show_results()

    finally:
        state["is_analyzing"] = False
        state["cancel_requested"] = False
        state["current_process"] = None

        set_analyze_button_enabled(True)
        set_cancel_button_visible(False)
        hide_loading()
        render_upload_box()


async def handle_upload(e):
    uploaded_file = e.file
    original_name = Path(uploaded_file.name).name

    timestamp = int(time.time())
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix

    safe_name = f"{stem}_{timestamp}{suffix}"
    save_path = UPLOAD_DIR / safe_name

    print("Upload handler called")
    print("Original file name:", original_name)
    print("Saved file name:", safe_name)
    print("Saving to:", save_path)

    try:
        await uploaded_file.save(save_path)

        if not save_path.exists():
            raise FileNotFoundError(f"Could not save file to {save_path}")

        if save_path.stat().st_size == 0:
            raise ValueError("Uploaded file was saved, but it is empty.")

        # Clean old uploaded files after saving the new file.
        # If Windows says a file is still locked, skip it instead of crashing.
        for old_file in UPLOAD_DIR.iterdir():
            if old_file.is_file() and old_file != save_path:
                try:
                    old_file.unlink()
                except PermissionError:
                    print(f"Skipped locked file: {old_file}")

        state["uploaded_audio_path"] = save_path
        state["uploaded_original_name"] = original_name
        state["uploaded_file_ready"] = True

        reset_analysis_outputs()

        upload_status_label.set_text(f"Uploaded: {original_name}")
        upload_path_label.set_text(f"Saved path: {save_path}")
        analysis_status_label.set_text("Ready. Click Analyze Singing.")

        remove_file_area.clear()

        with remove_file_area:
            ui.button("✕ Remove File", on_click=remove_uploaded_file).classes(
                "text-lg p-4 mt-2"
            ).props("color=negative")

        render_upload_box()
        ui.notify(f"Uploaded: {original_name}")

        print("Uploaded ready:", state["uploaded_file_ready"])
        print("Uploaded path:", state["uploaded_audio_path"])
        print("Original name:", state["uploaded_original_name"])
        print("File exists:", save_path.exists())
        print("File size:", save_path.stat().st_size)

    except Exception as error:
        state["uploaded_audio_path"] = None
        state["uploaded_original_name"] = None
        state["uploaded_file_ready"] = False

        upload_status_label.set_text("Upload failed.")
        upload_path_label.set_text(str(error))
        analysis_status_label.set_text("Upload failed.")
        render_upload_box()

        ui.notify("Upload failed. Check terminal output.", color="negative")
        print("Upload failed:", error)


def handle_rejected(e):
    ui.notify("File rejected. Please upload only one audio file.", color="negative")
    upload_status_label.set_text("File rejected. Upload exactly one audio file.")


def remove_uploaded_file():
    if state["is_analyzing"]:
        cancel_analysis()

    uploaded_path = state["uploaded_audio_path"]

    if uploaded_path is not None and uploaded_path.exists():
        try:
            uploaded_path.unlink()
        except PermissionError:
            print(f"Could not delete locked file: {uploaded_path}")
            ui.notify(
                "File is still being used. It was removed from the app, but Windows may delete it later.",
                color="warning",
            )

    state["uploaded_audio_path"] = None
    state["uploaded_original_name"] = None
    state["uploaded_file_ready"] = False

    reset_analysis_outputs()

    upload_status_label.set_text("No file uploaded yet.")
    upload_path_label.set_text("")
    analysis_status_label.set_text("")
    remove_file_area.clear()
    loading_area.clear()
    results_area.clear()

    render_upload_box()

    ui.notify("Uploaded file removed. You can upload another file.")


def update_selected_song(value):
    if state["is_analyzing"]:
        ui.notify("Cancel the current analysis before changing songs.", color="warning")
        return

    state["selected_song"] = value
    song_description_label.set_text(get_selected_song_description())
    render_reference_preview()
    analysis_status_label.set_text(f"Selected song: {value}")
    reset_analysis_outputs()
    results_area.clear()


# ============================================================
# Page Layout
# ============================================================

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
        get_song_options(),
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

    song_description_label = ui.label(get_selected_song_description()).classes(
        "text-lg text-gray-400 mt-2"
    )

    state["reference_preview_area"] = ui.column().classes("w-full")
    render_reference_preview()

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

            state["upload_box_area"] = ui.column().classes("items-center")
            render_upload_box()

    with ui.row().classes("gap-4 mt-8"):
        state["analyze_button"] = ui.button(
            "Analyze Singing",
            on_click=analyze_singing,
        ).classes("text-xl p-5")

        state["cancel_button"] = ui.button(
            "Cancel Analysis",
            on_click=cancel_analysis,
        ).classes("text-xl p-5").props("color=negative")

        state["cancel_button"].set_visibility(False)


upload_status_label = ui.label("No file uploaded yet.").classes(
    "text-xl text-gray-400 mt-4 px-4"
)

upload_path_label = ui.label("").classes(
    "text-lg text-blue-400 mt-2 px-4"
)

analysis_status_label = ui.label("").classes(
    "text-2xl text-green-400 mt-2 px-4"
)

remove_file_area = ui.row().classes("px-4")
loading_area = ui.column().classes("w-full px-4")
results_area = ui.column().classes("w-full px-4 pb-12")

ui.run()
