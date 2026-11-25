"""C++ implementation of the STREAM row tracker, with the Python track contract."""
import ctypes as C
from pathlib import Path

import numpy as np


class _TrackInfo(C.Structure):
    _fields_ = [('id', C.c_int64), ('lo', C.c_int64), ('last', C.c_int64),
                ('x0', C.c_int32), ('x1', C.c_int32), ('area', C.c_int64),
                ('invalid', C.c_int32), ('run_offset', C.c_size_t),
                ('run_count', C.c_size_t)]


class _Stats(C.Structure):
    _fields_ = [('count', C.c_size_t), ('run_count', C.c_size_t),
                ('active', C.c_size_t), ('merged', C.c_int64),
                ('indexed_rows', C.c_int64), ('linear_rows', C.c_int64),
                ('index_candidates', C.c_int64), ('linear_candidates', C.c_int64)]


class CppTracker:
    def __init__(self, gap, link, min_run, max_rows, use_spatial_index=True):
        library = Path(__file__).with_name('cpp') / 'libjelly_tracker.so'
        self.lib = C.CDLL(str(library))
        self.lib.jelly_tracker_create.argtypes = [C.c_int] * 5 + [C.POINTER(C.c_void_p), C.c_void_p, C.c_size_t]
        self.lib.jelly_tracker_create.restype = C.c_int
        self.lib.jelly_tracker_feed.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int,
                                                C.c_int64, C.c_int, C.POINTER(_Stats),
                                                C.c_void_p, C.c_size_t]
        self.lib.jelly_tracker_feed.restype = C.c_int
        self.lib.jelly_tracker_export.argtypes = [C.c_void_p, C.POINTER(_TrackInfo), C.c_size_t,
                                                  C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]
        self.lib.jelly_tracker_export.restype = C.c_int
        self.lib.jelly_tracker_reset.argtypes = [C.c_void_p]
        self.lib.jelly_tracker_reset.restype = C.c_size_t
        self.lib.jelly_tracker_destroy.argtypes = [C.c_void_p]
        self.lib.jelly_tracker_destroy.restype = None
        self.handle = C.c_void_p()
        err = C.create_string_buffer(256)
        rc = self.lib.jelly_tracker_create(int(gap), int(link), int(min_run), int(max_rows),
                                           int(use_spatial_index), C.byref(self.handle),err,len(err))
        if rc: raise RuntimeError(err.value.decode(errors='replace'))
        self.active_count = self.merged = self.indexed_rows = self.linear_rows = 0
        self.index_candidates = self.linear_candidates = 0

    @property
    def active(self):
        # Only len() and truthiness are used by the camera outside the Python tracker.
        return range(self.active_count)

    def reset(self):
        n = self.lib.jelly_tracker_reset(self.handle)
        self.active_count = 0
        return n

    def feed(self, hit, row0, x0, track_type):
        hit = np.ascontiguousarray(hit.view(np.uint8) if hit.dtype == np.bool_ else hit, dtype=np.uint8)
        if hit.ndim != 2: raise ValueError('tracker expects a two-dimensional mask')
        stats, err = _Stats(), C.create_string_buffer(256)
        rc = self.lib.jelly_tracker_feed(self.handle, hit.ctypes.data, *hit.shape,
                                         row0, x0, C.byref(stats), err, len(err))
        if rc: raise RuntimeError(err.value.decode(errors='replace'))
        self.active_count = stats.active
        for key in ('merged','indexed_rows','linear_rows','index_candidates','linear_candidates'):
            setattr(self, key, getattr(stats,key))
        if not stats.count:
            return []
        infos = (_TrackInfo * max(1, stats.count))()
        runs = np.empty(max(1,stats.run_count*3), dtype=np.int64)
        rc = self.lib.jelly_tracker_export(self.handle, infos, stats.count,
                                           runs.ctypes.data, stats.run_count*3, err, len(err))
        if rc: raise RuntimeError(err.value.decode(errors='replace'))
        result=[]
        for info in list(infos)[:stats.count]:
            if info.run_offset > stats.run_count or info.run_count > stats.run_count-info.run_offset:
                raise RuntimeError('tracker run buffer out of bounds')
            run_list = [tuple(r) for r in runs[info.run_offset*3:
                             (info.run_offset+info.run_count)*3].reshape(-1,3).tolist()]
            result.append(track_type(info.id,info.lo,info.last,info.x0,info.x1,
                                     info.area,run_list,invalid=bool(info.invalid)))
        return result

    def close(self):
        handle = getattr(self, 'handle', None)
        if handle and handle.value:
            self.lib.jelly_tracker_destroy(handle)
            self.handle = C.c_void_p()

    def __del__(self):
        self.close()
