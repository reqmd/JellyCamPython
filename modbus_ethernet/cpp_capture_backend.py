"""Linux KSJ capture into an independently owned array for every SDK call."""
import ctypes as ct
from pathlib import Path

import numpy as np


class NativeCapture:
    def __init__(self, sdk):
        path = Path(__file__).resolve().parent / "cpp" / "libjelly_capture.so"
        self.lib = ct.CDLL(str(path))
        self.capture_rgb = sdk.KSJ_CaptureRgbData
        self.read = self.lib.jelly_capture_rgb
        self.read.argtypes = (ct.c_void_p, ct.c_int, ct.c_void_p)
        self.read.restype = ct.c_int
        self.flip = self.lib.jelly_flip_rows
        self.flip.argtypes = (ct.c_void_p, ct.c_size_t, ct.c_size_t)
        self.flip.restype = None

    def acquire(self, index, shape):
        image = np.empty(shape, dtype=np.uint8)
        rc = self.read(self.capture_rgb, index, image.ctypes.data)
        return rc, image

    def chronological(self, image, reverse):
        if reverse:
            self.flip(image.ctypes.data, image.shape[0], image.strides[0])
        return image
