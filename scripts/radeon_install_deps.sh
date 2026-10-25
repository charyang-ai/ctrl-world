#!/bin/bash
# Container side. Installs everything except torch into /workspace/venv and
# keeps the image's ROCm PyTorch in /opt/venv.
#
# /workspace/venv is built on /usr/bin/python3.12, so --system-site-packages
# exposes the Ubuntu system packages and not /opt/venv. accelerate depends on
# torch, which makes pip pull a CUDA torch from PyPI. This script removes it
# afterwards. run_replay_radeon.sh then uses /opt/venv/bin/python and puts
# /workspace/venv on PYTHONPATH.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
INDEX=${PIP_INDEX_URL:-https://pypi.org/simple}

apt-get update
apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 python3-venv python3-pip

python3 -m venv --system-site-packages /workspace/venv
/workspace/venv/bin/python -m pip install -i "$INDEX" -U pip setuptools wheel
/workspace/venv/bin/python -m pip install -i "$INDEX" -r /workspace/src/requirements-rocm.txt

/workspace/venv/bin/python - << 'PY'
import subprocess
import sys

freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
names = [line.split("==", 1)[0] for line in freeze.splitlines() if "==" in line]
drop = [
    n for n in names
    if n.lower() in {"torch", "triton"} or n.lower().startswith(("nvidia-", "cuda-"))
]
print("UNINSTALL", " ".join(drop))
if drop:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", *drop])
PY

PYTHONPATH=/workspace/venv/lib/python3.12/site-packages /opt/venv/bin/python - << 'PY'
import torch
assert torch.version.hip, f"expected ROCm torch, got {torch.__version__}"
import accelerate, av, cv2, diffusers, einops, mediapy, transformers
print("torch", torch.__version__, "hip", torch.version.hip, "devices", torch.cuda.device_count())
print("diffusers", diffusers.__version__, "transformers", transformers.__version__)
print("DEPS_OK")
PY
