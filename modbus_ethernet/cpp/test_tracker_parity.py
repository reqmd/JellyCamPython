"""Compare native and Python STREAM trackers on deterministic masks and merges."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stream_detector import _Tracker, _Track
from cpp_tracker_backend import CppTracker


def compare(seed):
    rng=np.random.default_rng(seed)
    py=_Tracker(3,10,3,96,use_spatial_index=True)
    cpp=CppTracker(3,10,3,96,use_spatial_index=True)
    width=160
    for batch in range(180):
        mask=np.zeros((4,width),dtype=np.uint8)
        for row in mask:
            for _ in range(int(rng.integers(0,7))):
                a=int(rng.integers(0,width-30));b=a+int(rng.integers(1,29))
                row[a:b]=1
        base=1000+batch*4
        expected=[]
        for i,row in enumerate(mask):
            transitions=np.flatnonzero(np.diff(np.pad(row.astype(bool),(1,1)).astype(np.int8)))
            intervals=zip(transitions[::2].tolist(),transitions[1::2].tolist())
            expected.extend(py.step(base+i,intervals))
        got=cpp.feed(mask,base,0,_Track)
        def key(t):
            return (t.ident,t.lo,t.last,t.x0,t.x1,t.area,t.invalid,t.runs)
        assert [key(t) for t in expected]==[key(t) for t in got],(seed,batch)
        assert len(py.active)==len(cpp.active) and py.merged==cpp.merged,(seed,batch,'state')
        if batch==90:
            assert py.reset()==cpp.reset()
    cpp.close()


for seed in range(20):
    compare(seed)
print('STREAM tracker: Python/C++ output, ordering, merges and reset match')
