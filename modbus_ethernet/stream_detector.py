"""Experimental independent line-scan tracking for line_camera.py.

Use make_stream_detector(ObjectDetector, globals()) in LineCamera._detector.
Capture mode delegates to the original detector. Sorting uses full-resolution
background subtraction, row runs, and local morphology on completed tracks.
"""
from dataclasses import dataclass, field
from time import perf_counter_ns

import cv2
import numpy as np


@dataclass
class _Track:
    ident: int
    lo: int
    last: int
    x0: int
    x1: int
    area: int = 0
    runs: list = field(default_factory=list)
    frontier: list = field(default_factory=list)
    new_frontier: list = field(default_factory=list)
    invalid: bool = False


class _Tracker:
    def __init__(self, gap, link, min_run, max_rows):
        self.gap = max(1, gap)
        self.link = max(0, link)
        self.min_run = max(1, min_run)
        self.max_rows = max(1, max_rows)
        self.active = []
        self.next_id = 0
        self.merged = 0  # диагностический счётчик, не влияет на связывание

    def reset(self):
        n = len(self.active)
        self.active.clear()
        return n

    def _near(self, a, b, intervals):
        return any(a <= d + self.link and c <= b + self.link
                   for c, d in intervals)

    def _merge(self, a, b):
        a.lo = min(a.lo, b.lo)
        a.last = max(a.last, b.last)
        a.x0, a.x1 = min(a.x0, b.x0), max(a.x1, b.x1)
        a.area += b.area
        a.frontier.extend(b.frontier)
        a.new_frontier.extend(b.new_frontier)
        a.invalid = a.invalid or b.invalid
        if a.invalid:
            a.runs.clear()
        else:
            a.runs.extend(b.runs)
        self.active.remove(b)
        self.merged += 1

    def step(self, row, intervals):
        """Intervals are half-open in absolute sensor X; row is absolute Y."""
        # Group nearby intervals in the same row, but preserve their raw runs.
        groups = []
        for a, b in intervals:
            if b - a < self.min_run:
                continue
            if groups and a <= groups[-1][1] + self.link:
                groups[-1][1] = b
                groups[-1][2].append((a, b))
            else:
                groups.append([a, b, [(a, b)]])

        for t in self.active:
            t.new_frontier.clear()

        for a, b, raw in groups:
            matches = [t for t in self.active
                       if self._near(a, b, t.frontier)
                       or self._near(a, b, t.new_frontier)]
            if matches:
                t = matches[0]
                for other in matches[1:]:
                    self._merge(t, other)
            else:
                self.next_id += 1
                t = _Track(self.next_id, row, row, a, b)
                self.active.append(t)
            t.x0, t.x1 = min(t.x0, a), max(t.x1, b)
            t.new_frontier.append((a, b))
            t.area += sum(d - c for c, d in raw)
            if row - t.lo + 1 > self.max_rows:
                t.invalid = True
                t.runs.clear()
            if not t.invalid:
                t.runs.extend((row, c, d) for c, d in raw)

        done, keep = [], []
        for t in self.active:
            if t.new_frontier:
                t.frontier = t.new_frontier[:]
                t.last = row
            if row - t.last >= self.gap:
                done.append(t)
            else:
                keep.append(t)
        self.active = keep
        return done


def make_stream_detector(base, cfg):
    """Return a subclass, using the calling module's actual configuration."""
    x0, x1 = int(cfg['X0']), int(cfg['X1'])
    gap = int(cfg['GAP_ROWS'])
    min_rows, min_area = int(cfg['MIN_ROWS']), int(cfg['MIN_AREA'])
    threshold = int(cfg['TH'])
    edge = int(cfg['EDGE_OLD'])
    median_kernel = int(cfg['MEDIAN_KERNEL'])
    col_step = int(cfg['COL_STEP'])
    touch_pixels = int(cfg['TOUCH_PIXELS'])
    row_median = cfg['_row_median']
    close_k, open_k = int(cfg['CLOSE_K']), int(cfg['OPEN_K'])
    # Conservative horizontal association: CLOSE may join nearby fragments.
    # Can be overridden in line_camera.py for an A/B test.
    link = int(cfg.get('STREAM_LINK_X', close_k - 1))
    min_run = int(cfg.get('STREAM_MIN_RUN', 3))
    # OPEN + CLOSE have a finite spatial support. Give the mask a zero halo.
    halo = 2 * (open_k // 2 + close_k // 2)

    class StreamingObjectDetector(base):
        def __init__(self, full_bg, piece_rows, bg_rows=None):
            super().__init__(full_bg, piece_rows, bg_rows)
            self._tracker = _Tracker(gap, link, min_run,
                                     self.buf_pieces * self.H - 2 * halo - self.H)
            self._stream_mode = False
            self._current_track = None
            self._stream_bg_src = None
            self.stream_finished = 0
            self.stream_emitted = 0
            self.stream_invalid = 0
            self.stream_resets = 0
            self.stream_empty = 0
            self.stream_mask_ns = 0
            self.stream_track_ns = 0
            self.stream_pieces = 0
            self._stream_busy_rows = 0
            self._stream_bg_warned = False

        def reset_stream(self):
            self.stream_resets += self._tracker.reset()
            self._current_track = None
            self._stream_busy_rows = 0
            self._stream_bg_warned = False

        def note_gap(self):
            self.reset_stream()
            super().note_gap()

        def _change_mode(self, mode):
            self.reset_stream()
            # An object crossing a mode change is deliberately discarded.
            self.run_lo = self.run_hi = None
            self.run_px = self.gap = 0
            self.run_edge = False
            self.pending.clear()
            self.busy_streak = 0
            self._stream_mode = mode
            self._stream_bg_src = None
            if mode:
                print(f"  камера: STREAM tracking ON, GAP={gap}, "
                      f"LINK_X={link}, MIN_RUN={min_run}", flush=True)

        def feed(self, piece, t_end=None, lag=0.0):
            mode = not self.frames_enabled
            if mode != self._stream_mode:
                self._change_mode(mode)
            if not mode:
                return super().feed(piece, t_end, lag)

            # Exact full-resolution mask: same arithmetic as old extract_objects.
            t0 = perf_counter_ns()
            if self._stream_bg_src is not self.full_bg:
                self._stream_bg_src = self.full_bg
                self._stream_bg = self.full_bg[x0:x1].astype(np.int16)
            d = piece[:, x0:x1].astype(np.int16) - self._stream_bg
            d -= row_median(d[:, ::8])
            np.abs(d, out=d)
            hit = np.maximum(np.maximum(d[..., 0], d[..., 1]), d[..., 2]) > threshold
            rec = getattr(self, '_diag_rec', None)
            entry = getattr(self, '_diag_entry', None)
            if rec is not None and entry is not None:
                try:
                    rec.mask(entry, hit)
                except Exception as e:
                    rec.error(e)
            counts = hit[:, ::col_step].sum(axis=1) * col_step
            # Maintain the original history (including timestamps) so capture
            # mode and the existing time-to-valve contract remain unchanged.
            base_row = self.total
            self.pieces.append(piece)
            self.touch.append(counts >= touch_pixels)
            self.times.append((base_row, len(piece), t_end, lag))
            self.total += len(piece)
            occupied = bool(hit.any())
            if not occupied:
                self.idle += 1
            self.stream_mask_ns += perf_counter_ns() - t0
            self.stream_pieces += 1
            if not occupied and not self._tracker.active:
                self._stream_busy_rows = 0
                return []

            # Do not learn a new background from a dense product stream.
            # Warn once if almost the entire band stays foreground for a while.
            track_t0 = perf_counter_ns()
            for fraction in hit.mean(axis=1):
                self._stream_busy_rows = (self._stream_busy_rows + 1
                                          if fraction > .75 else 0)
            if (self._stream_busy_rows > self.buf_pieces * self.H
                    and not self._stream_bg_warned):
                self._stream_bg_warned = True
                print('  камера: STREAM: длительно занята почти вся полоса; '
                      'проверьте фон на пустой ленте', flush=True)

            # Find transitions for all rows at once. No Python loop over pixels.
            padded = np.pad(hit, ((0, 0), (1, 1)), constant_values=False)
            changes = padded[:, 1:] != padded[:, :-1]
            self.stream_track_ns += perf_counter_ns() - track_t0
            for i, transitions in enumerate(changes):
                t0 = perf_counter_ns()
                ends = np.flatnonzero(transitions) + x0
                intervals = zip(ends[::2].tolist(), ends[1::2].tolist())
                completed = self._tracker.step(base_row + i, intervals)
                self.stream_track_ns += perf_counter_ns() - t0
                for track in completed:
                    self.stream_finished += 1
                    if track.invalid or track.last - track.lo + 1 < min_rows:
                        self.rejected += 1
                        self.stream_invalid += 1
                        self._diag_track(track, 'too_long' if track.invalid else 'short')
                        continue
                    # No global MIN_ROW_PIXELS/approximate MIN_AREA prefilter:
                    # the exact post-morphology component area is authoritative.
                    if self.on_run is not None:
                        self._current_track = track
                        try:
                            self.on_run(track.lo, track.last + 1)
                        finally:
                            self._current_track = None
            return []

        def _diag_track(self, track, reason, **extra):
            rec = getattr(self, '_diag_rec', None)
            entry = getattr(self, '_diag_entry', None)
            if rec is not None and entry is not None:
                try:
                    rec.track(entry, dict(track_id=track.ident, row0=track.lo,
                        row1=track.last+1, x0=track.x0, x1=track.x1,
                        raw_area=track.area, reason=reason, **extra))
                except Exception as e:
                    rec.error(e)

        def extract_objects(self, lo, hi):
            track = self._current_track
            if track is None:
                return super().extract_objects(lo, hi)
            # Only pixels belonging to this tracked group enter its mask.
            # Thus a different track in the local rectangle cannot be emitted
            # again. Nearby tracks are conservatively merged by _Tracker.
            first = self.total - sum(p.shape[0] for p in self.pieces)
            if lo < first or hi > self.total:
                self.rejected += 1
                self.stream_invalid += 1
                self._diag_track(track, 'history_missing')
                return [], 0
            left, right = max(x0, track.x0 - halo), min(x1, track.x1 + halo)
            start = lo - halo
            stop = hi + halo
            mask = np.zeros((stop - start, right - left), np.uint8)
            for row, a, b in track.runs:
                mask[row - start, a - left:b - left] = 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.ker_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.ker_close)
            n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            found, at_edge = [], 0
            diagnostic = getattr(self, '_diag_entry', None) is not None
            components = [] if diagnostic else None
            # Actual color pixels must be present; halo is mask-only, never
            # fabricated RGB. Read the real track rows once, as before.
            rows = self._rows(lo, hi - lo) if n > 1 else None
            for i in range(1, n):
                x, y, w, h, area = (int(v) for v in st[i])
                a, b = left + x, left + x + w
                r0, r1 = start + y, start + y + h
                component = None
                if diagnostic:
                    component = dict(x0=a,x1=b,row0=r0,row1=r1,area=area,reason='emitted')
                    components.append(component)
                if area < min_area:
                    if component is not None: component['reason'] = 'area'
                    continue
                if a < x0 + edge or b > x1 - edge:
                    at_edge += 1
                    if component is not None: component['reason'] = 'edge'
                    continue
                if r0 < lo or r1 > hi:
                    # No crop outside real tracked rows; count explicitly.
                    self.stream_invalid += 1
                    if component is not None: component['reason'] = 'outside_rows'
                    continue
                m = (lab[y:y+h, x:x+w] == i).astype(np.uint8)
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filled = np.zeros_like(m)
                cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
                crop = rows[r0-lo:r1-lo, a:b].copy()
                if median_kernel > 1:
                    crop = cv2.medianBlur(crop, median_kernel)
                crop[filled == 0] = 0
                found.append(dict(x0=a, x1=b, row0=r0, row1=r1,
                                  area=area, crop=crop,
                                  t_front=self.row_time(r0),
                                  cam_lag_ms=self.row_lag(r0),
                                  track_id=track.ident))
            if diagnostic:
                # RLE preserves the exact local post-morph mask, without PNG I/O.
                try:
                    transitions = np.diff(np.pad(mask != 0, ((0,0),(1,1))).astype(np.int8), axis=1)
                    ys, xs = np.nonzero(transitions)
                    post = None
                    if len(xs) <= 20000:
                        post = [[int(y), int(a), int(b)] for y,a,b in zip(ys[::2], xs[::2], xs[1::2])]
                    self._diag_track(track, 'segmented' if found else 'empty',
                        components=components, mask_origin=[start,left], mask_shape=list(mask.shape),
                        post_runs=post, raw_runs=[list(r) for r in track.runs] if len(track.runs) <= 10000 else None, emitted=len(found))
                except Exception as e:
                    self._diag_rec.error(e)
            self.stream_emitted += len(found)
            if not found:
                self.stream_empty += 1
                self.rejected += 1
            return found, at_edge

        def stream_status(self):
            n = max(1, self.stream_pieces)
            return (f'STREAM: active={len(self._tracker.active)}, '
                    f'finished={self.stream_finished}, emitted={self.stream_emitted}, '
                    f'empty={self.stream_empty}, invalid={self.stream_invalid}, '
                    f'resets={self.stream_resets}, '
                    f'mask={self.stream_mask_ns/n/1e6:.3f} ms/piece, '
                    f'track={self.stream_track_ns/n/1e6:.3f} ms/piece')

    return StreamingObjectDetector
