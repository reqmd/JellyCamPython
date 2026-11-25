"""Replay one real camera mask sequence through Python and C++ STREAM trackers."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stream_detector import _Track, _Tracker
from cpp_tracker_backend import CppTracker


def signature(track):
    return (track.ident, track.lo, track.last, track.x0, track.x1,
            track.area, track.invalid, track.runs)


def replay(filename):
    with np.load(filename, allow_pickle=False) as saved:
        if int(saved['version']) != 1:
            raise ValueError('неизвестная версия записи масок')
        meta = {key: int(saved[key]) for key in
                ('x0', 'x1', 'gap', 'link', 'min_run', 'max_rows', 'index')}
        masks = saved['masks']
        rows = saved['row0']
        resets = saved['reset_before']
        width = meta['x1'] - meta['x0']
        if (masks.ndim != 3 or masks.dtype != np.uint8 or
                masks.shape[0] != len(rows) or len(rows) != len(resets) or
                masks.shape[2] != (width+7)//8 or width <= 0 or
                len(rows) == 0 or len(rows) > 60000):
            raise ValueError('неверная форма или размер сохранённых масок')
        print(f'Файл: {filename}; порций={len(rows)}, строк/порция={masks.shape[1]}, '
              f'ROI={meta["x0"]}:{meta["x1"]}, GAP={meta["gap"]}, '
              f'LINK={meta["link"]}, MIN_RUN={meta["min_run"]}')
        args = (meta['gap'], meta['link'], meta['min_run'], meta['max_rows'])
        py = _Tracker(*args, use_spatial_index=bool(meta['index']))
        cpp = CppTracker(*args, use_spatial_index=bool(meta['index']))
        completed = invalid = resets_n = 0
        try:
            for block, (packed, row0, reset) in enumerate(zip(masks, rows, resets)):
                if reset:
                    a, b = py.reset(), cpp.reset()
                    if a != b:
                        raise AssertionError(f'порция {block}: сброс: Python={a}, C++={b}')
                    resets_n += 1
                hit = np.unpackbits(packed, axis=1, count=width)
                # The live detector skips an entirely empty piece only when
                # it has no active tracks; preserve that condition here.
                if not hit.any() and not py.active and not cpp.active:
                    continue
                for i, row in enumerate(hit):
                    padded = np.pad(row.astype(bool), (1,1), constant_values=False)
                    ends = np.flatnonzero(padded[1:] != padded[:-1]) + meta['x0']
                    intervals = zip(ends[::2].tolist(), ends[1::2].tolist())
                    expected = py.step(int(row0)+i, intervals)
                    actual = cpp.feed(row[None,:], int(row0)+i, meta['x0'], _Track)
                    if len(expected) != len(actual):
                        raise AssertionError(f'порция {block}, строка {i}: '
                                             f'завершено Python={len(expected)}, C++={len(actual)}')
                    for j, (a,b) in enumerate(zip(expected,actual)):
                        if signature(a) != signature(b):
                            raise AssertionError(f'порция {block}, строка {i}, трек {j}: '
                                                 f'Python={signature(a)!r}; C++={signature(b)!r}')
                        invalid += int(a.invalid)
                    completed += len(expected)
                    if len(py.active) != len(cpp.active) or py.merged != cpp.merged:
                        raise AssertionError(f'порция {block}, строка {i}: '
                                             'число активных треков или объединений не совпало')
        finally:
            cpp.close()
        print(f'СОВПАДЕНИЕ: завершено={completed}, длинных невалидных={invalid}, '
              f'объединений={py.merged}, сбросов={resets_n}, '
              f'активных на конце={len(py.active)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file',type=Path,help='stream_masks_*.npz из mask_captures')
    args = parser.parse_args()
    try:
        replay(args.file)
    except (AssertionError,ValueError,RuntimeError,OSError) as exc:
        print(f'НЕСОВПАДЕНИЕ/ОШИБКА: {exc}',file=sys.stderr)
        sys.exit(1)
