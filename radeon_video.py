"""PyAV stand-in for decord's VideoReader.

The rollout scripts import `VideoReader` and `cpu` from here. Frames come back
as a uint8 array of shape (T, H, W, 3), the same thing decord's
`VideoReader.get_batch(...).asnumpy()` returns.

It was written on the assumption that decord had no CPython 3.12 wheel. That
assumption was wrong: decord 0.6.0 ships a py3-none manylinux2010 wheel that
installs and runs in the ROCm 7.14 image. On all 12 validation videos in
dataset_example/droid_subset, this reader and decord return identical frames
(max abs difference 0), so either one feeds the replay the same input.
"""
from __future__ import annotations

import numpy as np


def cpu(index=0):
    return index


class _Batch:
    def __init__(self, array: np.ndarray):
        self._array = array

    def asnumpy(self) -> np.ndarray:
        return self._array

    def numpy(self) -> np.ndarray:
        return self._array


class VideoReader:
    def __init__(self, uri, ctx=None, num_threads=0):
        self._frames = _read_rgb(str(uri))

    def __len__(self) -> int:
        return int(self._frames.shape[0])

    def get_batch(self, indices):
        idx = np.asarray(list(indices), dtype=np.int64)
        return _Batch(self._frames[idx])


def _read_rgb(uri: str) -> np.ndarray:
    errors = []
    try:
        import av

        frames = []
        with av.open(uri) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
        if frames:
            return np.stack(frames, axis=0)
        errors.append("pyav decoded zero frames")
    except Exception as exc:  # noqa: BLE001 - fall through to OpenCV
        errors.append(f"pyav: {exc}")

    try:
        import cv2

        cap = cv2.VideoCapture(uri)
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        if frames:
            return np.stack(frames, axis=0)
        errors.append("opencv decoded zero frames")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"opencv: {exc}")

    raise RuntimeError(f"could not read {uri}: {errors}")
