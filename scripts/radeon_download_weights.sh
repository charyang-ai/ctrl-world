#!/bin/bash
# Container side. Downloads the three checkpoints the replay reads, about 42 GB.
# Run radeon_install_deps.sh first: it provides huggingface_hub.
# Set HF_ENDPOINT=https://hf-mirror.com to use the mirror, and HF_TOKEN if the
# Hugging Face site asks you to log in or accept a license.
set -euo pipefail
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
export HF_HUB_DISABLE_XET=1
mkdir -p /workspace/weights/svd /workspace/weights/clip /workspace/weights/ctrl-world

/workspace/venv/bin/python - << 'PY'
from huggingface_hub import snapshot_download

snapshot_download("openai/clip-vit-base-patch32", local_dir="/workspace/weights/clip")
snapshot_download("stabilityai/stable-video-diffusion-img2vid", local_dir="/workspace/weights/svd")
snapshot_download(
    "yjguo/Ctrl-World",
    local_dir="/workspace/weights/ctrl-world",
    allow_patterns=["checkpoint-10000.pt", "config.json", "README.md"],
)
print("WEIGHTS_OK")
PY
