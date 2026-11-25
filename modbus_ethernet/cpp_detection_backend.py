import ctypes as ct
from pathlib import Path

import numpy as np


class NativeDetection:
    def __init__(self):
        lib = ct.CDLL(str(Path(__file__).with_name("cpp") / "libjelly_detection.so"))
        self.detect = lib.jelly_detect
        p, i = ct.c_void_p, ct.c_int
        self.detect.argtypes = [p, i, i, i, i, p, i, i, i, p, p, p, p]
        self.detect.restype = i
        self.scratch = np.empty(0, np.int32)

    def mask(self, image, bg, x0, threshold, step, touch):
        rows, cols, channels = image.shape
        width = bg.shape[0]
        if (channels != 3 or image.dtype != np.uint8 or bg.shape != (width, 3)
                or bg.dtype != np.int16 or not 0 <= x0 < x0 + width <= cols or step < 1):
            raise ValueError("invalid BGR block or detector ROI")
        image = np.ascontiguousarray(image)
        bg = np.ascontiguousarray(bg)
        samples = (width + 7) // 8
        if self.scratch.size < samples:
            self.scratch = np.empty(samples, np.int32)
        hit = np.empty((rows, width), np.bool_)
        touched = np.empty(rows, np.bool_)
        dense = np.empty(rows, np.bool_)
        occupied = self.detect(image.ctypes.data, rows, image.strides[0], x0, width,
                               bg.ctypes.data, threshold, step, touch, self.scratch.ctypes.data,
                               hit.ctypes.data, touched.ctypes.data, dense.ctypes.data)
        return hit, touched, dense, bool(occupied)
