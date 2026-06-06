from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "WestDoor/voice-coaching-system-weights"
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

for filename in ["best.ckpt", "best.pth"]:
    print(f"Downloading {filename}...")
    hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=OUTPUT_DIR,
    )

print("Done. Weights downloaded to models/")
