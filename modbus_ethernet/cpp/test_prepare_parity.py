"""Compare C++ input preparation against the exact server OpenCV pipeline."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cpp_prepare_backend import CppPrepare

native=CppPrepare()
rng=np.random.default_rng(19)
for width,height in ((32,32),(32,24),(32,16)):
    for w,h in ((1,1),(2,3),(32,16),(63,17),(17,63),(48,16),(89,43),(13,31)) + tuple(
            (int(rng.integers(1,150)),int(rng.integers(1,150))) for _ in range(100)):
        crop=rng.integers(0,256,(h,w,3),dtype=np.uint8)
        scale=min(width/w,height/h)
        nw=min(width,max(1,round(w*scale)))
        nh=min(height,max(1,round(h*scale)))
        resized=cv2.resize(crop,(nw,nh),interpolation=cv2.INTER_LINEAR)
        canvas=np.zeros((height,width,3),dtype=np.uint8)
        y0,x0=(height-nh)//2,(width-nw)//2
        canvas[y0:y0+nh,x0:x0+nw]=resized
        expected=cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)[None,...]
        actual=native(crop,width,height)
        assert np.array_equal(expected,actual),(width,height,w,h,'pixel mismatch')
print('NPU input: exact Python/C++ parity on 324 synthetic crops')
