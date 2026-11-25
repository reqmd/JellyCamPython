"""ctypes bridge for the persistent RKNN C++ context; one instance per NPU worker."""

import ctypes as C
from pathlib import Path

import numpy as np


class CppRKNN:
    def __init__(self, model, core, library=None):
        path = Path(library) if library else Path(__file__).with_name("cpp") / "libjelly_rknn.so"
        if library is None and not path.is_file():
            # Позволяет проверить вырезание до пересборки прежней библиотеки.
            path = Path(__file__).with_name("libjelly_rknn.so")
        if not path.is_file():
            raise RuntimeError(f"C++ NPU library not found: {path}")
        self.lib = C.CDLL(str(path))  # ctypes.CDLL releases the GIL during RKNN calls.
        self.lib.jelly_rknn_create.argtypes = [C.c_char_p, C.c_int, C.POINTER(C.c_void_p),
                                               C.POINTER(C.c_size_t), C.POINTER(C.c_size_t),
                                               C.c_void_p, C.c_size_t]
        self.lib.jelly_rknn_create.restype = C.c_int
        self.lib.jelly_rknn_infer.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t,
                                              C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]
        self.lib.jelly_rknn_infer.restype = C.c_int
        self.lib.jelly_rknn_destroy.argtypes = [C.c_void_p]
        self.lib.jelly_rknn_destroy.restype = None
        self.handle = C.c_void_p()
        n_input, n_output = C.c_size_t(), C.c_size_t()
        error = C.create_string_buffer(256)
        rc = self.lib.jelly_rknn_create(str(model).encode(), int(core),
                                        C.byref(self.handle), C.byref(n_input),
                                        C.byref(n_output), error, len(error))
        if rc != 0:
            raise RuntimeError(f"C++ RKNN init: {error.value.decode(errors='replace')}")
        self.input_bytes = n_input.value
        self.output_count = n_output.value

    def inference(self, inputs):
        if len(inputs) != 1:
            raise ValueError("C++ RKNN expects exactly one input")
        inp = inputs[0]
        if inp.dtype != np.uint8 or not inp.flags.c_contiguous or inp.nbytes != self.input_bytes:
            raise ValueError(f"C++ RKNN expects contiguous uint8 input ({self.input_bytes} bytes)")
        logits = np.empty(self.output_count, dtype=np.float32)
        error = C.create_string_buffer(256)
        rc = self.lib.jelly_rknn_infer(self.handle, inp.ctypes.data, inp.nbytes,
                                       logits.ctypes.data, logits.size, error, len(error))
        if rc != 0:
            raise RuntimeError(f"C++ RKNN inference: {error.value.decode(errors='replace')}")
        return [logits]

    def close(self):
        handle = getattr(self, "handle", None)
        if handle and handle.value:
            self.lib.jelly_rknn_destroy(handle)
            self.handle = C.c_void_p()

    def __del__(self):
        self.close()
