"""Bounded, opt-in acquisition diagnostics. No image I/O during sorting."""
import json
import math
import os
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from valve_config import CONFIG, HERE

CFG = dict(enabled=False, directory='diagnostics', pre_ms=80, post_ms=220,
           max_ms=500, max_fragments=24, max_memory_mb=384,
           trigger_threshold=25, trigger_pixels=5, auto_trigger=True, manual_ms=1500,
           manual_file='/tmp/jellycam-diagnostic-trigger')
CFG.update(CONFIG.get('diagnostics') or {})
_current = None


def current():
    return _current


def start(session, camera, params):
    global _current
    if not CFG['enabled']:
        return
    try:
        det = getattr(camera, 'detector', None)
        if det is None:
            print('  DIAG: камера не готова, запись не запущена')
            return
        _current = Recorder(session, det.full_bg, params, CFG)
        print(f'  DIAG: запись включена | авто={bool(CFG["auto_trigger"])} | ручной touch {CFG["manual_file"]}')
    except Exception as e:
        print(f'  DIAG: не удалось запустить: {e!r}')


def stop():
    global _current
    rec, _current = _current, None
    if rec is not None:
        try:
            rec.save()
        except Exception as e:
            print(f'  DIAG ERROR: сохранение: {e!r}')


def poll_manual():
    rec = _current
    if rec is None:
        return
    try:
        path = Path(rec.cfg['manual_file'])
        if path.exists():
            path.unlink()
            rec.manual()
    except Exception as e:
        rec.error(e)


class Recorder:
    def __init__(self, session, background, params, cfg):
        self.session = session
        self.cfg = dict(cfg)
        self.params = dict(params)
        self.bg = background.copy()
        self.x0, self.x1 = params['X0'], params['X1']
        self.sample_bg = self.bg[self.x0:self.x1:4].astype(np.int16)
        self.lock = threading.Lock()
        self.active = True
        self.pre = deque()
        self.index = {}  # seq -> entry; only entries retained in pre/fragment
        self.fragments = []
        self.fragment = None
        self.bytes = 0
        self.cap = int(cfg['max_memory_mb'] * 1024**2)
        self.manual_requested = False
        self.cooldown_until = 0.0
        self.events = []
        self.event_limit = 100000
        self.event_dropped = 0
        self.capture_skipped = 0
        self.errors = 0
        self.started = time.monotonic()
        self.wall = time.strftime('%Y%m%d_%H%M%S')
        self.stamp = time.time_ns()

    def error(self, e):
        with self.lock:
            self.errors += 1
            n = self.errors
        if n == 1:
            print(f'  DIAG ERROR: {e!r}')

    def manual(self):
        with self.lock:
            if self.active:
                self.manual_requested = True

    def _trigger(self, piece):
        # Independent of STREAM: no run linking, morphology or MIN_AREA.
        # Remove row-wide brightness shift, but use an independently sampled mask.
        d = piece[:, self.x0:self.x1:4].astype(np.int16) - self.sample_bg
        k = d.shape[1] // 2
        d -= np.partition(d, k, axis=1)[:, k:k+1]
        np.abs(d, out=d)
        return bool(np.any((d.max(axis=2) > self.cfg['trigger_threshold']).sum(axis=1)
                           >= self.cfg['trigger_pixels']))

    def capture(self, seq, piece, t_end, lag, host_end):
        """Called before detector queue; owns immutable processed frame reference."""
        trigger = self.cfg['auto_trigger'] and self._trigger(piece)
        with self.lock:
            if not self.active:
                return
            manual = self.manual_requested
            self.manual_requested = False
            # Retain a short pre-roll, including pieces that never reach detector.
            entry = dict(session=self.session, seq=int(seq), t_end=float(t_end), host_end=float(host_end),
                         lag_ms=float(lag), image=piece, row0=None, mask=None,
                         tracks=[], objects=[])
            self.pre.append(entry)
            self.index[seq] = entry
            pre_s = self.cfg['pre_ms'] / 1000
            while self.pre and (host_end - self.pre[0]['host_end'] > pre_s
                                or len(self.pre) > 512):
                old = self.pre.popleft()
                if not old.get('retained'):
                    self.index.pop(old['seq'], None)
            if self.fragment is not None and host_end >= self.fragment['end']:
                self.fragment = None
                self.cooldown_until = host_end + .10
            if manual or (trigger and self.fragment is None and host_end >= self.cooldown_until):
                if self.fragment is None and len(self.fragments) < self.cfg['max_fragments'] and self.bytes < self.cap:
                    duration = self.cfg['manual_ms'] if manual else self.cfg['max_ms']
                    self.fragment = dict(id=len(self.fragments)+1, trigger='manual' if manual else 'auto',
                                         start=host_end, end=host_end+duration/1000,
                                         hard_end=host_end+duration/1000, entries=[])
                    if not manual:
                        self.fragment['end'] = host_end + self.cfg['post_ms']/1000
                    self.fragments.append(self.fragment)
                    for p in self.pre:
                        self._retain(p)
                    if manual:
                        print(f'  DIAG: фрагмент {self.fragment["id"]} начат, бросайте зёрна')
                elif manual and self.fragment is not None:
                    # Do not cut an existing auto fragment; extend it for manual shot.
                    self.fragment['trigger'] = 'manual'
                    self.fragment['hard_end'] = self.fragment['end'] = host_end + self.cfg['manual_ms']/1000
                    print(f'  DIAG: фрагмент {self.fragment["id"]} продлён, бросайте зёрна')
                elif manual:
                    print('  DIAG: лимит памяти/фрагментов, остановите сортировку')
            if self.fragment is not None:
                self._retain(entry)
                if trigger and self.fragment['trigger'] == 'auto':
                    self.fragment['end'] = min(self.fragment['hard_end'], host_end+self.cfg['post_ms']/1000)

    def _retain(self, e):
        if e.get('last_fragment') == self.fragment['id']:
            return
        # Conservative reserve for mask + metadata, plus original BGR data.
        cost = e['image'].nbytes + e['image'].shape[0]*(self.x1-self.x0) + 1024
        if self.bytes + cost > self.cap:
            self.capture_skipped += 1
            self.fragment['truncated'] = True
            return
        e['retained'] = True
        e['last_fragment'] = self.fragment['id']
        self.bytes += cost
        self.fragment['entries'].append(e)

    def detector_piece(self, seq, row0):
        with self.lock:
            e = self.index.get(seq)
            if self.active and e is not None and e.get('session') == self.session:
                e['row0'] = int(row0)
                return e
        return None

    def mask(self, e, hit):
        with self.lock:
            if self.active and e is not None and e.get('session') == self.session:
                e['mask'] = hit  # new, immutable bool array from STREAM

    def track(self, e, record):
        with self.lock:
            if self.active and e is not None and e.get('session') == self.session:
                if len(e['tracks']) < 512:
                    e['tracks'].append(record)
                else:
                    self.event_dropped += 1

    def decision(self, e, record):
        with self.lock:
            if self.active and e is not None and e.get('session') == self.session:
                e['objects'].append(record)

    def spi(self, record):
        with self.lock:
            if self.active:
                if len(self.events) < self.event_limit:
                    self.events.append(record)
                else:
                    self.event_dropped += 1

    def save(self):
        with self.lock:
            self.active = False
            fragments = self.fragments
            if self.fragment is not None:
                self.fragment['ended_by_stop'] = True
        root = Path(self.cfg['directory'])
        if not root.is_absolute():
            root = Path(HERE) / root
        root = root / f'{self.wall}_s{self.session}_{self.stamp%1000000000:09d}'
        root.mkdir(parents=True, exist_ok=False)
        np.save(root/'background.npy', self.bg)
        manifest = dict(session=self.session, config=self.cfg, params=self.params,
                        started_monotonic=self.started, started_wall_ns=self.stamp,
                        fragments=[], bytes_reserved=self.bytes,
                        skipped_pieces=self.capture_skipped, diagnostic_errors=self.errors,
                        dropped_events=self.event_dropped, spi=self.events,
                        note='BGR corrected, chronological rows; x absolute; row0 detector index; null means not processed/mapped. Mask is pre run-filter STREAM; post-morph components in tracks.')
        for f in fragments:
            es = f['entries']
            if not es:
                continue
            prefix = f'fragment_{f["id"]:02d}'
            images = np.concatenate([e['image'] for e in es])
            mask = np.concatenate([e['mask'].astype(np.uint8)*255 if e['mask'] is not None
                                   else np.zeros((len(e['image']),self.x1-self.x0),np.uint8) for e in es])
            if not cv2.imwrite(str(root/(prefix+'.png')), images, [cv2.IMWRITE_PNG_COMPRESSION,1]):
                raise IOError(prefix+' color PNG')
            if not cv2.imwrite(str(root/(prefix+'_mask.png')), mask, [cv2.IMWRITE_PNG_COMPRESSION,1]):
                raise IOError(prefix+' mask PNG')
            offset = 0
            records = []
            for e in es:
                records.append(dict(seq=e['seq'], image_row=offset, rows=len(e['image']),
                                    row0=e['row0'], t_end=e['t_end'], host_end=e['host_end'], lag_ms=e['lag_ms'],
                                    mask_available=e['mask'] is not None, tracks=e['tracks'], objects=e['objects']))
                offset += len(e['image'])
            info = {k:v for k,v in f.items() if k!='entries'}
            info.update(image=prefix+'.png', mask=prefix+'_mask.png', rows=offset, pieces=records)
            (root/(prefix+'.json')).write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
            manifest['fragments'].append(dict(id=f['id'], trigger=f['trigger'], rows=offset, metadata=prefix+'.json', truncated=f.get('truncated',False)))
        (root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'  DIAG: сохранено {len(manifest["fragments"])} фрагментов | {root} | '
              f'пропущено порций={self.capture_skipped}, ошибок={self.errors}')
