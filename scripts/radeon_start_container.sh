#!/bin/bash
# Host side. Creates the single-GPU container used for the validation run.
#
#   bash scripts/radeon_start_container.sh /path/to/workdir
#
# The workdir is mounted at /workspace. Clone this repo to <workdir>/src first.
# Device groups are added by number: the image's /etc/group has no "render"
# entry, so `--group-add render` fails inside docker.
set -euo pipefail
HOST_DIR=${1:?usage: radeon_start_container.sh HOST_DIR}
NAME=${NAME:-radeon-ctrl-world}
IMAGE=${IMAGE:-rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0}
VIDEO_GID=$(getent group video | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3)
mkdir -p "$HOST_DIR/weights" "$HOST_DIR/results"

docker run -d --name "$NAME" \
  --entrypoint bash \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add "$VIDEO_GID" \
  --group-add "$RENDER_GID" \
  --ipc=host \
  --shm-size=32g \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0 \
  -e ROCR_VISIBLE_DEVICES=0 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v "$HOST_DIR":/workspace \
  -w /workspace \
  "$IMAGE" \
  -lc 'sleep infinity'

docker exec "$NAME" /opt/venv/bin/python -c \
  'import torch; print("torch", torch.__version__, "hip", torch.version.hip, "devices", torch.cuda.device_count())'
