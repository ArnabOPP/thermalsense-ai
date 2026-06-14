"""
ThermalSense AI — Railway startup script
Downloads model files and feature matrices from Hugging Face before starting the API.
"""
import os
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "arnabinthegame05/thermalsense-ai-models"
ROOT = Path(__file__).resolve().parent

print("Downloading model files from Hugging Face...")
snapshot_download(
    repo_id=REPO_ID,
    repo_type="model",
    local_dir=str(ROOT),
    local_dir_use_symlinks=False,
)
print("Download complete. Starting API...")

# Start the actual API
import uvicorn
from backend.main import app
port = int(os.environ.get("PORT", 8000))
uvicorn.run(app, host="0.0.0.0", port=port)