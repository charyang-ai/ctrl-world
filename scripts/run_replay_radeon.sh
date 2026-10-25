#!/bin/bash
# Official Ctrl-World path (1): replay recorded DROID actions inside the world model.
# One W7900. The image keeps ROCm PyTorch in /opt/venv. A project venv holds the
# other wheels; PYTHONPATH exposes them without letting pip replace torch.
set -euo pipefail
cd /workspace/src
export PYTHONPATH=/workspace/venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}
PY=/opt/venv/bin/python
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export WANDB_DISABLED=true
export SWANLAB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export MIOPEN_USER_DB_PATH=/workspace/weights/miopen
export MIOPEN_CUSTOM_CACHE_DIR=/workspace/weights/miopen
mkdir -p /workspace/results /workspace/weights/miopen /workspace/src/synthetic_traj
LOG=/workspace/results/replay.log
echo "REPLAY_START $(date -Is)" | tee "$LOG"
"$PY" - << 'PY' | tee -a "$LOG"
import torch
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("file", torch.__file__)
print("cuda_is_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("total_mem_gb", round(props.total_memory / 1024**3, 2))
assert getattr(torch.version, "hip", None), "expected the image ROCm torch"
PY
set +e
"$PY" -u scripts/rollout_replay_traj.py \
  --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info \
  --dataset_names droid_subset \
  --svd_model_path /workspace/weights/svd \
  --clip_model_path /workspace/weights/clip \
  --ckpt_path /workspace/weights/ctrl-world/checkpoint-10000.pt \
  --task_type replay \
  2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e
echo "$status" > /workspace/results/replay.exit
echo "REPLAY_EXIT $status $(date -Is)" | tee -a "$LOG"
"$PY" - << 'PY'
import os, glob
out = "/workspace/results/summary.txt"
lines = []
paths = sorted(glob.glob("/workspace/src/synthetic_traj/**/*.mp4", recursive=True))
lines.append(f"mp4_count {len(paths)}")
for p in paths:
    lines.append(f"{os.path.getsize(p)} {p}")
text = "\n".join(lines) + "\n"
open(out, "w", encoding="utf-8").write(text)
print(text, end="")
PY
exit "$status"
