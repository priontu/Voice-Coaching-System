def analyze_audio(song_name: str, input_mode: str) -> dict:
    return {
        "final": 84,
        "pitch": 88,
        "timing": 79,
        "duration": 82,
        "lyrics": 86,
        "feedback": [
            f"You selected: {song_name}",
            f"Input mode: {input_mode}",
            "Your pitch was mostly accurate, but a few notes were slightly sharp.",
            "Your timing was a little late in the middle phrase.",
            "Some note endings were cut slightly short.",
        ],
    }