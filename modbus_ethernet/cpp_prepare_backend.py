"""Native OpenCV resize/pad/BGR-to-RGB; output owned by NumPy."""
import ctypes as C
from pathlib import Path

import numpy as np


class CppPrepare:
    def __init__(self):
        self.lib = C.CDLL(str(Path(__file__).with_name('cpp') / 'libjelly_prepare.so'))
        self.lib.jelly_prepare_rgb.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_size_t,
                                              C.c_void_p, C.c_int, C.c_int,C.c_void_p,C.c_size_t]
        self.lib.jelly_prepare_rgb.restype = C.c_int

    def __call__(self, crop_bgr, width, height):
        crop = np.ascontiguousarray(crop_bgr, dtype=np.uint8)
        if crop.ndim != 3 or crop.shape[2] != 3 or not all(crop.shape):
            raise ValueError('expected nonempty BGR crop')
        output = np.empty((1,height,width,3), dtype=np.uint8)
        error = C.create_string_buffer(256)
        rc = self.lib.jelly_prepare_rgb(crop.ctypes.data, crop.shape[1], crop.shape[0],
                                        crop.strides[0], output.ctypes.data, width, height,
                                        error,len(error))
        if rc: raise RuntimeError(error.value.decode(errors='replace'))
        return output
