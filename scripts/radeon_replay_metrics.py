"""PSNR of replay videos against the decoded ground truth.

rollout_replay_traj.py saves each frame as the decoded ground truth (top half)
over the world-model prediction (bottom half), with the three DROID views side
by side. "static" scores a model that predicts no motion: it repeats the first
ground-truth frame. Frame 0 is skipped because the model is given that frame.

    python scripts/radeon_replay_metrics.py run2/*.mp4 [--other run1/*.mp4]

With --other, the predictions of two runs are also compared, matched by
trajectory id. The replay sets no seed, so this shows how much the sampling
noise alone moves the output.
"""
import argparse
import re

import av
import numpy as np

VIEWS = ("exterior_1", "exterior_2", "wrist")


def read_frames(path):
    with av.open(path) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def split(video):
    t, h2, w3, _ = video.shape
    h, w = h2 // 2, w3 // len(VIEWS)
    views = lambda x: x.reshape(t, h, len(VIEWS), w, 3).transpose(0, 2, 1, 3, 4)
    return views(video[:, :h]), views(video[:, h:])  # (T, V, H, W, 3) each


def psnr(a, b):
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean(axis=(-3, -2, -1))
    return 10 * np.log10(255.0**2 / np.maximum(mse, 1e-10))


def traj_id(path):
    return re.search(r"_traj_(\d+)_", path).group(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+")
    p.add_argument("--other", nargs="*", default=[])
    args = p.parse_args()
    other = {traj_id(v): v for v in args.other}

    rows = []
    for path in sorted(args.videos, key=lambda v: int(traj_id(v))):
        gt, pred = split(read_frames(path))
        model = psnr(pred[1:], gt[1:])  # (T-1, V)
        static = psnr(gt[:1], gt[1:])
        line = f"traj {traj_id(path)} frames {len(gt)}"
        for v, name in enumerate(VIEWS):
            line += f" | {name} model {model[:, v].mean():.2f} static {static[:, v].mean():.2f}"
        tail = model[-10:].mean()
        line += f" | all model {model.mean():.2f} static {static.mean():.2f} model_last10 {tail:.2f}"
        if traj_id(path) in other:
            _, pred_other = split(read_frames(other[traj_id(path)]))
            n = min(len(pred), len(pred_other))
            line += f" | run_vs_run {psnr(pred[1:n], pred_other[1:n]).mean():.2f}"
        rows.append((model.mean(), static.mean()))
        print(line)
    m = np.array(rows)
    print(f"mean over {len(m)} trajectories: model {m[:, 0].mean():.2f} dB, static {m[:, 1].mean():.2f} dB")


if __name__ == "__main__":
    main()
