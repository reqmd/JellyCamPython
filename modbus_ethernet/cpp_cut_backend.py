"""C++ STREAM crop backend. Input snapshots stay owned by Python during the call."""

import ctypes as C
from pathlib import Path
from time import perf_counter_ns

import numpy as np


class _Item(C.Structure):
    _fields_ = [("x0", C.c_int), ("x1", C.c_int),
                ("row0", C.c_int64), ("row1", C.c_int64),
                ("area", C.c_int), ("width", C.c_int), ("height", C.c_int),
                ("offset", C.c_size_t), ("bytes", C.c_size_t)]


class _Stats(C.Structure):
    _fields_ = [("count", C.c_int), ("at_edge", C.c_int), ("invalid", C.c_int),
                ("mask_ms", C.c_double), ("morph_ms", C.c_double),
                ("components_ms", C.c_double), ("crop_ms", C.c_double),
                ("output_bytes", C.c_size_t)]


class CppCut:
    def __init__(self):
        library = Path(__file__).with_name("cpp") / "libjelly_cut.so"
        if not library.exists():
            raise RuntimeError(f"нет C++ библиотеки вырезания: {library}")
        self.lib = C.CDLL(str(library))
        self.lib.jelly_cut.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_size_t,
            C.c_int, C.c_void_p, C.c_size_t, C.c_int64, C.c_int64,
            C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
            C.c_int, C.c_int, C.c_int, C.POINTER(C.c_void_p),
            C.POINTER(_Stats), C.c_void_p, C.c_size_t]
        self.lib.jelly_cut.restype = C.c_int
        self.lib.jelly_cut_export.argtypes = [C.c_void_p, C.POINTER(_Item),
            C.c_size_t, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]
        self.lib.jelly_cut_export.restype = C.c_int
        self.lib.jelly_cut_destroy.argtypes = [C.c_void_p]
        self.lib.jelly_cut_destroy.restype = None

    def extract(self, snap, *, left, right, halo, x0, x1, edge, min_area,
                median_kernel, open_kernel, close_kernel):
        start_ns = perf_counter_ns()
        rows = np.ascontiguousarray(snap['rows'], dtype=np.uint8)
        runs = np.ascontiguousarray(snap['track'].runs, dtype=np.int64).reshape(-1, 3)
        if rows.ndim != 3 or rows.shape[2] != 3:
            raise ValueError("ожидается снимок BGR [строки, столбцы, 3]")
        if not runs.size:
            runs = np.zeros((1, 3), dtype=np.int64)
            n_runs = 0
        else:
            n_runs = len(runs)
        handle, stats = C.c_void_p(), _Stats()
        error = C.create_string_buffer(256)
        call_ns = perf_counter_ns()
        rc = self.lib.jelly_cut(rows.ctypes.data, rows.shape[0], rows.shape[1],
            rows.strides[0], int(snap.get('x_offset', 0)), runs.ctypes.data,
            n_runs, snap['lo'], snap['hi'], left, right, halo, x0, x1,
            edge, min_area, median_kernel, open_kernel, close_kernel,
            C.byref(handle), C.byref(stats), error, len(error))
        export_start_ns = perf_counter_ns()
        if rc != 0:
            raise RuntimeError(f"C++ вырезание: {error.value.decode(errors='replace')}")
        profile = None
        try:
            # One owned NumPy buffer for every crop from this snapshot. Views
            # keep this buffer alive after the native result is destroyed.
            pixels = np.empty(max(1, stats.output_bytes), dtype=np.uint8)
            items = (_Item * max(1, stats.count))()
            rc = self.lib.jelly_cut_export(handle, items, stats.count,
                pixels.ctypes.data, stats.output_bytes, error, len(error))
            if rc != 0:
                raise RuntimeError(f"C++ вырезание: {error.value.decode(errors='replace')}")
            objects_start_ns = perf_counter_ns()
            found = []
            for index in range(stats.count):
                item = items[index]
                if item.bytes != item.width*item.height*3:
                    raise RuntimeError("C++ вырезание: неверный размер crop")
                if item.offset > stats.output_bytes or item.bytes > stats.output_bytes-item.offset:
                    raise RuntimeError("C++ вырезание: crop вне выходного буфера")
                crop = pixels[item.offset:item.offset+item.bytes].reshape(
                    item.height,item.width,3)
                found.append((item.x0,item.x1,item.row0,item.row1,item.area,crop))
            end_ns = perf_counter_ns()
            profile = {"mask_ms":stats.mask_ms,"morph_ms":stats.morph_ms,
                       "components_ms":stats.components_ms,"crop_ms":stats.crop_ms,
                       "bridge_prep_ms":(call_ns-start_ns)/1e6,
                       "bridge_call_ms":(export_start_ns-call_ns)/1e6,
                       "bridge_export_ms":(objects_start_ns-export_start_ns)/1e6,
                       "bridge_objects_ms":(end_ns-objects_start_ns)/1e6}
            return found, stats.at_edge, stats.invalid, profile
        finally:
            cleanup_ns = perf_counter_ns()
            self.lib.jelly_cut_destroy(handle)
            if profile is not None:
                profile['bridge_cleanup_ms'] = (
                    perf_counter_ns()-cleanup_ns)/1e6
