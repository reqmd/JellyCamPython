"""Bounded capture of consecutive STREAM masks for stateful tracker replay."""
import os
import threading
from datetime import datetime
from pathlib import Path

import numpy as np


class MaskCapture:
    def __init__(self, directory, max_blocks, *, x0, x1, gap, link, min_run,
                 max_rows, index):
        directory = Path(directory)
        self.directory = directory if directory.is_absolute() else Path(__file__).resolve().parent / directory
        self.max_blocks = min(60000, max(1, int(max_blocks)))
        self.meta = dict(version=1, x0=x0, x1=x1, gap=gap, link=link,
                         min_run=min_run, max_rows=max_rows, index=int(index))
        self.masks = []
        self.row0 = []
        self.resets = []
        self.pending_reset = True
        self.finished = False
        self.path = self.directory / f"stream_masks_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.npz"

    def mark_reset(self):
        self.pending_reset = True

    def add(self, hit, row0):
        if self.finished:
            return
        if hit.ndim != 2 or hit.shape[1] != self.meta['x1']-self.meta['x0']:
            raise ValueError('mask capture geometry changed')
        # All camera portions, including empty masks, must be kept in order.
        self.masks.append(np.packbits(hit, axis=1))
        self.row0.append(row0)
        self.resets.append(self.pending_reset)
        self.pending_reset = False
        if len(self.masks) >= self.max_blocks:
            self.finish()

    def finish(self):
        if self.finished or not self.masks:
            return
        self.finished = True
        mask_parts, row_parts, reset_parts = self.masks, self.row0, self.resets
        meta = self.meta.copy()
        self.masks, self.row0, self.resets = [], [], []

        def write():
            tmp = self.path.with_suffix('.npz.tmp')
            try:
                masks = np.stack(mask_parts)
                rows = np.asarray(row_parts, dtype=np.int64)
                resets = np.asarray(reset_parts, dtype=np.uint8)
                self.directory.mkdir(parents=True, exist_ok=True)
                with tmp.open('wb') as out:
                    np.savez(out, masks=masks, row0=rows, reset_before=resets,
                             **meta)
                os.replace(tmp, self.path)
                print(f'  STREAM маски сохранены: {self.path} '
                      f'порций={len(rows)} строк={masks.shape[1]}',flush=True)
            except Exception as exc:
                print(f'  STREAM не удалось сохранить маски: {exc!r}',flush=True)
                try: tmp.unlink(missing_ok=True)
                except OSError: pass

        threading.Thread(target=write, name='stream-mask-writer',daemon=True).start()
