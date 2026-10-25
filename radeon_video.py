"""Video reader used in place of decord.

The ROCm PyTorch image is CPython 3.12. Published decord wheels stop at
3.11, so `import decord` is not a dependency we can pin. This module keeps
the two names the rollout scripts import, `VideoReader` and `cpu`. Frames
come back as a uint8 array of shape (T, H, W, 3), which is what
`VideoReader.get_batch(...).asnumpy()` returns.
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
