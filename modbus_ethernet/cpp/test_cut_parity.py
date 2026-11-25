"""Run on Orange Pi after `make`: Python and C++ segmentation on identical tracks."""

import sys
import threading
import gc
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stream_detector import _Track, make_stream_detector


class Base:
    pass


def detector(backend):
    cfg = dict(X0=0, X1=100, GAP_ROWS=3, MIN_ROWS=6, MIN_AREA=10,
               TH=30, EDGE_OLD=2, MEDIAN_KERNEL=3, COL_STEP=4,
               TOUCH_PIXELS=2, _row_median=None, CLOSE_K=11, OPEN_K=3,
               CUT_BACKEND=backend, ROWS_PER_SEC=4000)
    cls = make_stream_detector(Base, cfg)
    obj = object.__new__(cls)
    obj.ker_open = np.ones((3,3),np.uint8)
    obj.ker_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    obj._stream_counts_lock = threading.Lock()
    obj.stream_invalid = obj.stream_emitted = obj.stream_empty = obj.rejected = 0
    return obj


def snapshot(seed, runs):
    lo, hi = 1000, 1048
    x0 = min(a for _,a,_ in runs)
    x1 = max(b for _,_,b in runs)
    halo = 12
    left = max(0,x0-halo)
    right = min(100,x1+halo)
    rng = np.random.default_rng(seed)
    rows = rng.integers(0,256,(hi-lo,right-left,3),dtype=np.uint8)
    track = _Track(seed, lo, hi-1, x0, x1, runs=runs)
    return dict(track=track, lo=lo, hi=hi, rows=rows, x_offset=left,
                timings=[(i/4000,0.2) for i in range(lo,hi)],
                diag_rec=None, diag_entry=None)


def compare(name, snap):
    py, cpp = detector('python'), detector('cpp')
    a, edge_a = py.extract_snapshot(dict(snap))
    b, edge_b = cpp.extract_snapshot(dict(snap))
    gc.collect()  # Crop must own its bytes after the C++ result has been freed.
    assert edge_a==edge_b and len(a)==len(b), (name,edge_a,edge_b,len(a),len(b))
    for i,(x,y) in enumerate(zip(a,b)):
        keys = ('x0','x1','row0','row1','area','t_front','t_last','cam_lag_ms')
        assert all(x[k]==y[k] for k in keys), (name,i,[(k,x[k],y[k]) for k in keys])
        assert np.array_equal(x['crop'],y['crop']), (name,i,'crop mismatch')
    assert (py.stream_invalid,py.stream_empty)==(cpp.stream_invalid,cpp.stream_empty)
    print(f"{name}: OK, crops={len(a)}, edge={edge_a}")


def main():
    compare('single',snapshot(1,[(r,28,47) for r in range(1012,1029)]))
    compare('two',snapshot(2,[(r,25,38) for r in range(1010,1032)] +
                           [(r,60,76) for r in range(1010,1032)]))
    compare('hole',snapshot(3,[(r,25,61) if r in (1010,1030) else
                              (r,25,34) if r%2 else (r,52,61)
                              for r in range(1010,1031)]))
    compare('edge',snapshot(4,[(r,0,11) for r in range(1012,1030)]))
    for seed in range(5,205):
        rng = np.random.default_rng(seed)
        runs = []
        for _ in range(int(rng.integers(1,4))):
            a = int(rng.integers(0,77))
            w = int(rng.integers(4,min(25,101-a)))
            lo = int(rng.integers(1004,1020))
            h = int(rng.integers(5,18))
            runs.extend((r,a,a+w) for r in range(lo,lo+h))
        compare(f'random-{seed}',snapshot(seed,runs))
    print('Python/C++ parity confirmed on synthetic tracks')


if __name__ == '__main__':
    main()
