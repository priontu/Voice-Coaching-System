from nicegui import ui, app
from pathlib import Path
import subprocess
import sys
import os
import time
import re
import shutil

# Theme setting
dark = ui.dark_mode()
dark.enable()

# Paths
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
UPLOAD_DIR = BASE_DIR / "uploads"

PITCH_MODEL_DIR = PROJECT_DIR / "PitchModel_Kim"
INFERENCE_SCRIPT = PITCH_MODEL_DIR / "inference.py"
PITCH_SCORE_SCRIPT = PITCH_MODEL_DIR / "pitch_score.py"

PITCH_SCORE_REFERENCE_PATH = PITCH_MODEL_DIR / "test.musicxml"

PITCH_PLOT_PATH = PITCH_MODEL_DIR / "pitch_data_vad_pitch.png"
PITCH_VS_REFERENCE_PATH = PITCH_MODEL_DIR / "pitch_vs_reference.png"
CENT_ERROR_PATH = PITCH_MODEL_DIR / "cent_error_over_time.png"
PITCH_SUMMARY_PATH = PITCH_MODEL_DIR / "pitch_accuracy_summary.png"
NOTE_ERROR_PATH = PITCH_MODEL_DIR / "note_level_pitch_errors.png"

UPLOAD_DIR.mkdir(exist_ok=True)

# Song registry
SONG_REGISTRY = {
    "Test Song": {
        "musicxml": PITCH_MODEL_DIR / "test.musicxml",
    },
}

# Serve PitchModel_Kim as a static folder so PNG outputs can appear in the browser
if PITCH_MODEL_DIR.exists():
    app.add_static_files("/pitch_outputs", str(PITCH_MODEL_DIR))

# Default mock results for non-pitch categories
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

# App state
state = {
    "selected_mode": None,
    "selected_song": "Test Song",
    "recording_ready": False,
    "uploaded_file_ready": False,
    "uploaded_audio_path": None,

    "last_inference_stdout": "",
    "last_inference_stderr": "",
    "last_score_stdout": "",
    "last_score_stderr": "",

    "pitch_plot_url": None,
    "pitch_vs_reference_url": None,
    "cent_error_url": None,
    "pitch_summary_url": None,
    "note_error_url": None,

    "pitch_accuracy": None,
    "mace": None,
    "pitch_rmse": None,
    "note_pitch_acc": None,
    "correct_notes": None,
    "total_notes": None,
}


# ----------------------------
# UI helper functions
# ----------------------------

def clamp(value, low=0, high=100):
    if value is None:
        return 0
    return max(low, min(high, value))


def format_percent(value):
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


def format_cents(value):
    if value is None:
        return "N/A"
    return f"{value:.2f} cents"


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


def professional_score_card(title, value, subtitle=""):
    with ui.card().classes(
        "w-72 h-40 p-6 rounded-2xl shadow-lg justify-center"
    ):
        ui.label(title).classes("text-lg text-gray-400")
        ui.label(value).classes("text-5xl font-bold mt-2")
        if subtitle:
            ui.label(subtitle).classes("text-sm text-gray-500 mt-2")


def progress_metric(title, value, description):
    percent = clamp(value)

    with ui.card().classes("w-full p-6 rounded-2xl shadow-md"):
        with ui.row().classes("w-full justify-between items-center"):
            ui.label(title).classes("text-2xl font-bold")
            ui.label(format_percent(value)).classes("text-2xl font-bold")

        ui.linear_progress(value=percent / 100).classes("w-full mt-4 h-4")
        ui.label(description).classes("text-lg text-gray-400 mt-3")


def error_metric(title, value, good_value, bad_value, description):
    if value is None:
        percent = 0
        display_value = "N/A"
    else:
        # Lower error is better, so convert error into a score-like progress value
        percent = 100 * (bad_value - value) / (bad_value - good_value)
        percent = clamp(percent)
        display_value = format_cents(value)

    with ui.card().classes("w-full p-6 rounded-2xl shadow-md"):
        with ui.row().classes("w-full justify-between items-center"):
            ui.label(title).classes("text-2xl font-bold")
            ui.label(display_value).classes("text-2xl font-bold")

        ui.linear_progress(value=percent / 100).classes("w-full mt-4 h-4")
        ui.label(description).classes("text-lg text-gray-400 mt-3")


# ----------------------------
# Parsing pitch_score.py output
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
# Results page
# ----------------------------

def show_results():
    results_area.clear()

    pitch_score_display = rounded_or_mock(
        state["pitch_accuracy"],
        mock_results["pitch"],
    )

    # For now, overall score is based on pitch only.
    # Later, this can combine pitch + timing + duration + lyrics.
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

        # Main score cards
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
                "Timing model not connected yet",
            )
            professional_score_card(
                "Lyrics",
                "Soon",
                "Phoneme model not connected yet",
            )

        # Pitch summary
        ui.label("Pitch Summary").classes("text-4xl font-bold mt-12")

        with ui.column().classes("w-full gap-4 mt-4"):
            progress_metric(
                "Pitch Accuracy",
                state["pitch_accuracy"],
                "Percentage of voiced frames sung within ±50 cents of the reference.",
            )

            progress_metric(
                "Note-Level Pitch Correctness",
                state["note_pitch_acc"],
                "Percentage of reference notes whose median pitch was within ±50 cents.",
            )

            error_metric(
                "Average Pitch Error",
                state["mace"],
                good_value=15,
                bad_value=100,
                description="Lower is better. Around 100 cents is about one semitone.",
            )

            error_metric(
                "Pitch RMSE",
                state["pitch_rmse"],
                good_value=20,
                bad_value=120,
                description="Lower is better. This penalizes larger pitch mistakes more heavily.",
            )

        # Correct notes
        if state["correct_notes"] is not None and state["total_notes"] is not None:
            ui.label("Note Accuracy").classes("text-4xl font-bold mt-12")

            with ui.card().classes("w-full p-6 rounded-2xl shadow-md"):
                ui.label(
                    f"{state['correct_notes']} out of {state['total_notes']} notes were sung correctly."
                ).classes("text-2xl font-bold")

                note_ratio = state["correct_notes"] / state["total_notes"]
                ui.linear_progress(value=note_ratio).classes("w-full mt-4 h-4")

        # Main graph: only one graph visible by default
        if state["pitch_vs_reference_url"] is not None:
            ui.label("Pitch vs Reference").classes("text-4xl font-bold mt-12")

            with ui.card().classes("w-full p-6 rounded-2xl shadow-lg"):
                ui.label(
                    "This chart compares the detected singing pitch against the reference melody."
                ).classes("text-xl text-gray-400 mb-4")

                ui.image(state["pitch_vs_reference_url"]).classes(
                    "w-full max-w-6xl mx-auto rounded-xl"
                )

        # Feedback
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
                feedback.append("Pitch accuracy needs work. Many sung frames are outside the ±50 cent range.")

        if state["mace"] is not None:
            feedback.append(f"Average pitch deviation is {state['mace']:.2f} cents.")

        if state["note_pitch_acc"] is not None:
            feedback.append(f"Note-level pitch correctness is {state['note_pitch_acc']:.2f}%.")

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

        # Advanced graphs hidden by default
        with ui.expansion("Advanced Graphs", icon="analytics").classes("w-full mt-12"):
            ui.label(
                "These are detailed analysis plots generated by the pitch pipeline."
            ).classes("text-lg text-gray-400 mb-4")

            if state["pitch_plot_url"] is not None:
                ui.label("Pitch Extraction Visualization").classes("text-2xl font-bold mt-4")
                ui.image(state["pitch_plot_url"]).classes(
                    "w-full max-w-6xl mx-auto rounded-xl mt-2"
                )

            if state["cent_error_url"] is not None:
                ui.label("Cent Error Over Time").classes("text-2xl font-bold mt-6")
                ui.image(state["cent_error_url"]).classes(
                    "w-full max-w-6xl mx-auto rounded-xl mt-2"
                )

            if state["pitch_summary_url"] is not None:
                ui.label("Pitch Accuracy Summary").classes("text-2xl font-bold mt-6")
                ui.image(state["pitch_summary_url"]).classes(
                    "w-full max-w-5xl mx-auto rounded-xl mt-2"
                )

            if state["note_error_url"] is not None:
                ui.label("Note-Level Pitch Errors").classes("text-2xl font-bold mt-6")
                ui.image(state["note_error_url"]).classes(
                    "w-full max-w-6xl mx-auto rounded-xl mt-2"
                )

        # Debug output hidden by default
        with ui.expansion("Debug Output", icon="terminal").classes("w-full mt-4"):
            if state["last_inference_stdout"]:
                ui.label("inference.py output").classes("text-xl font-bold")
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

def run_command(command, cwd):
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }

    return subprocess.run(
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

    # If source and destination are the same file, do not copy.
    if source_musicxml.resolve() != PITCH_SCORE_REFERENCE_PATH.resolve():
        shutil.copyfile(source_musicxml, PITCH_SCORE_REFERENCE_PATH)

    print("Selected song:", selected_song)
    print("Using MusicXML:", source_musicxml)
    print("Pitch scorer reference:", PITCH_SCORE_REFERENCE_PATH)

    return True


def run_pitch_inference(uploaded_path: Path) -> bool:
    analysis_status_label.set_text("Running pitch extraction...")
    state["last_inference_stdout"] = ""
    state["last_inference_stderr"] = ""
    state["pitch_plot_url"] = None

    if not PITCH_MODEL_DIR.exists():
        analysis_status_label.set_text("PitchModel_Kim folder not found.")
        ui.notify(f"Missing folder: {PITCH_MODEL_DIR}", color="negative")
        return False

    if not INFERENCE_SCRIPT.exists():
        analysis_status_label.set_text("inference.py not found.")
        ui.notify(f"Missing file: {INFERENCE_SCRIPT}", color="negative")
        return False

    uploaded_path = uploaded_path.resolve()

    command = [
        sys.executable,
        "inference.py",
        "--audio",
        str(uploaded_path),
        "--visualize",
    ]

    print("Running inference command:")
    print(" ".join(command))
    print("Working directory:", PITCH_MODEL_DIR)

    result = run_command(command, PITCH_MODEL_DIR)

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

    if PITCH_PLOT_PATH.exists():
        state["pitch_plot_url"] = f"/pitch_outputs/pitch_data_vad_pitch.png?t={int(time.time())}"
    else:
        state["pitch_plot_url"] = None
        ui.notify("Pitch extraction plot was not found.", color="warning")

    analysis_status_label.set_text("Pitch extraction complete.")
    ui.notify("Pitch extraction complete.", color="positive")
    return True


def run_pitch_score() -> bool:
    analysis_status_label.set_text("Running pitch scoring...")
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
    state["pitch_summary_url"] = None
    state["note_error_url"] = None

    if not PITCH_SCORE_SCRIPT.exists():
        analysis_status_label.set_text("pitch_score.py not found.")
        ui.notify(f"Missing file: {PITCH_SCORE_SCRIPT}", color="negative")
        return False

    if not prepare_selected_musicxml():
        return False

    pitch_data_path = PITCH_MODEL_DIR / "pitch_data.json"

    if not PITCH_SCORE_REFERENCE_PATH.exists():
        analysis_status_label.set_text("test.musicxml not found.")
        ui.notify("Missing test.musicxml in PitchModel_Kim.", color="negative")
        return False

    if not pitch_data_path.exists():
        analysis_status_label.set_text("pitch_data.json not found.")
        ui.notify("Missing pitch_data.json. Run inference first.", color="negative")
        return False

    command = [
        sys.executable,
        "pitch_score.py",
    ]

    print("Running pitch score command:")
    print(" ".join(command))
    print("Working directory:", PITCH_MODEL_DIR)

    result = run_command(command, PITCH_MODEL_DIR)

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

    timestamp = int(time.time())

    if PITCH_VS_REFERENCE_PATH.exists():
        state["pitch_vs_reference_url"] = f"/pitch_outputs/pitch_vs_reference.png?t={timestamp}"

    if CENT_ERROR_PATH.exists():
        state["cent_error_url"] = f"/pitch_outputs/cent_error_over_time.png?t={timestamp}"

    if PITCH_SUMMARY_PATH.exists():
        state["pitch_summary_url"] = f"/pitch_outputs/pitch_accuracy_summary.png?t={timestamp}"

    if NOTE_ERROR_PATH.exists():
        state["note_error_url"] = f"/pitch_outputs/note_level_pitch_errors.png?t={timestamp}"

    analysis_status_label.set_text("Pitch scoring complete.")
    ui.notify("Pitch scoring complete.", color="positive")
    return True


# ----------------------------
# App actions
# ----------------------------

def analyze_singing():
    print("Analyze clicked")
    print("Selected mode:", state["selected_mode"])
    print("Selected song:", state["selected_song"])
    print("Uploaded ready:", state["uploaded_file_ready"])
    print("Uploaded path:", state["uploaded_audio_path"])

    if state["selected_mode"] == "sing":
        if not state["recording_ready"]:
            ui.notify("Please sing directly first.", color="negative")
            return

        ui.notify("Sing Directly mode is still a placeholder.", color="warning")
        show_results()
        return

    elif state["selected_mode"] == "upload":
        uploaded_path = state["uploaded_audio_path"]

        if not state["uploaded_file_ready"] or uploaded_path is None:
            ui.notify("Please upload one audio file first.", color="negative")
            return

        if not uploaded_path.exists():
            ui.notify("The uploaded file was not found on disk.", color="negative")
            upload_status_label.set_text("Uploaded file is missing.")
            upload_path_label.set_text(f"Missing path: {uploaded_path}")
            return

        upload_status_label.set_text(f"Ready to analyze: {uploaded_path.name}")

        success = run_pitch_inference(uploaded_path)
        if not success:
            return

        success = run_pitch_score()
        if not success:
            return

        show_results()

    else:
        ui.notify("Please choose a singing input method first.", color="negative")
        return


def choose_mode(mode):
    state["selected_mode"] = mode
    input_area.clear()
    results_area.clear()

    if mode == "upload":
        state["uploaded_file_ready"] = False
        state["uploaded_audio_path"] = None

        state["last_inference_stdout"] = ""
        state["last_inference_stderr"] = ""
        state["last_score_stdout"] = ""
        state["last_score_stderr"] = ""

        state["pitch_plot_url"] = None
        state["pitch_vs_reference_url"] = None
        state["cent_error_url"] = None
        state["pitch_summary_url"] = None
        state["note_error_url"] = None

        upload_status_label.set_text("")
        upload_path_label.set_text("")
        analysis_status_label.set_text("")
        remove_file_area.clear()

    with input_area:
        if mode == "sing":
            ui.label("Sing Directly").classes("text-3xl font-bold mt-6")
            ui.label(
                "Press play/start and sing along with the selected song."
            ).classes("text-xl text-gray-400")

            with ui.row().classes("gap-4 mt-4"):
                ui.button("▶ Play / Start Singing", on_click=start_singing).classes(
                    "text-lg p-4"
                )
                ui.button("■ Stop Singing", on_click=stop_singing).classes(
                    "text-lg p-4"
                )

            ui.label(
                "Note: direct recording is only a placeholder for now."
            ).classes("text-lg text-gray-500 mt-2")

        elif mode == "upload":
            ui.label("Upload Audio File").classes("text-3xl font-bold mt-6")
            ui.label(
                "Upload exactly one WAV, MP3, M4A, or FLAC file."
            ).classes("text-xl text-gray-400")

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

            upload_status_label.set_text("No file uploaded yet.")
            upload_path_label.set_text("")
            analysis_status_label.set_text("")
            remove_file_area.clear()

        ui.button("Analyze Singing", on_click=analyze_singing).classes(
            "mt-8 text-xl p-5"
        )


def start_singing():
    state["recording_ready"] = False
    ui.notify("Singing started. Sing now!")


def stop_singing():
    state["recording_ready"] = True
    ui.notify("Singing stopped. Recording is ready for analysis.")


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

        state["pitch_plot_url"] = None
        state["pitch_vs_reference_url"] = None
        state["cent_error_url"] = None
        state["pitch_summary_url"] = None
        state["note_error_url"] = None

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

    state["last_inference_stdout"] = ""
    state["last_inference_stderr"] = ""
    state["last_score_stdout"] = ""
    state["last_score_stderr"] = ""

    state["pitch_plot_url"] = None
    state["pitch_vs_reference_url"] = None
    state["cent_error_url"] = None
    state["pitch_summary_url"] = None
    state["note_error_url"] = None

    upload_status_label.set_text("No file uploaded yet.")
    upload_path_label.set_text("")
    analysis_status_label.set_text("")
    remove_file_area.clear()
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
    ui.button("Toggle Theme", on_click=dark.toggle).classes("text-lg p-4")

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
    ).classes("w-96 text-xl mt-4")

    ui.label("Choose Input Method").classes("text-4xl font-bold mt-8")

    with ui.row().classes("gap-6 mt-4"):
        ui.button("Sing Directly", on_click=lambda: choose_mode("sing")).classes(
            "text-xl p-5"
        )
        ui.button("Upload Audio File", on_click=lambda: choose_mode("upload")).classes(
            "text-xl p-5"
        )

    input_area = ui.column().classes("w-full")


upload_status_label = ui.label("").classes("text-xl text-gray-400 mt-4 px-4")
upload_path_label = ui.label("").classes("text-lg text-blue-400 mt-2 px-4")
analysis_status_label = ui.label("").classes("text-xl text-green-400 mt-2 px-4")
remove_file_area = ui.row().classes("px-4")

results_area = ui.column().classes("w-full px-4 pb-12")

ui.run()