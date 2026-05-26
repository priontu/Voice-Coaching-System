import json
import torch
import torchcrepe
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Config
AUDIO_PATH = "test.wav"

SAMPLE_RATE = 16000
HOP_LENGTH = 160  # 160 samples at 16 kHz = 10 ms


# Frequency in Hz → MIDI note
def hz_to_midi(f0_hz):

    return 69 + 12 * np.log2(f0_hz / 440.0)


def main():

    # 1. Load audio
    audio, sr = librosa.load(
        AUDIO_PATH,
        sr=SAMPLE_RATE,
        mono=True
    )

    audio_tensor = torch.tensor(audio).unsqueeze(0)

    print("Audio loaded")
    print("Sample rate:", sr)
    print("Audio shape:", audio.shape)

    # 2. Extract pitch / F0
    print("\nExtracting pitch...")

    f0 = torchcrepe.predict(
        audio_tensor,
        SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        fmin=50,
        fmax=1000,
        model="full",
        batch_size=1024,
        device="cpu"
    )

    f0 = f0.squeeze().numpy()

    # Masking silent/unvoiced frames
    voiced = f0[f0 > 0]

    if len(voiced) == 0:
        print("No voiced pitch detected.")
        return

    # Basic statistics
    min_pitch = np.min(voiced)
    max_pitch = np.max(voiced)
    avg_pitch = np.mean(voiced)
    median_pitch = np.median(voiced)

    print("\nPitch Extraction Results")
    print("------------------------")

    print("F0 shape:", f0.shape)
    print("Voiced frames:", len(voiced))

    print(f"Min pitch: {min_pitch:.2f} Hz")
    print(f"Max pitch: {max_pitch:.2f} Hz")
    print(f"Average pitch: {avg_pitch:.2f} Hz")
    print(f"Median pitch: {median_pitch:.2f} Hz")

    print("\nApprox MIDI Notes")
    print("-----------------")

    print(f"Min MIDI: {hz_to_midi(min_pitch):.2f}")
    print(f"Max MIDI: {hz_to_midi(max_pitch):.2f}")
    print(f"Average MIDI: {hz_to_midi(avg_pitch):.2f}")
    print(f"Median MIDI: {hz_to_midi(median_pitch):.2f}")

    # Create time axis
    times = np.arange(len(f0)) * (HOP_LENGTH / SAMPLE_RATE)

    # Save pitch data to JSON
    print("\nSaving pitch data...")

    pitch_frames = []

    for t, pitch in zip(times, f0):

        pitch_frames.append({
            "time": float(t),
            "f0": float(pitch),
            "midi": float(hz_to_midi(pitch)) if pitch > 0 else None
        })

    output_data = {
        "audio_path": AUDIO_PATH,
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "num_frames": len(f0),
        "frames": pitch_frames
    }

    with open("pitch_data.json", "w") as f:
        json.dump(output_data, f, indent=2)

    print("Saved pitch data to pitch_data.json")

    # Plot pitch contour (for visualization purposes)
    print("\nGenerating pitch contour plot...")

    plt.figure(figsize=(12, 4))

    plt.plot(times, f0)

    plt.title("Extracted Singing Pitch Contour")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("pitch_contour.png", dpi=200)

    plt.show()

    print("Saved plot as pitch_contour.png")


if __name__ == "__main__":
    main()