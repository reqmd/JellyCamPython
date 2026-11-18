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
    def __init__(self, gap, link, min_run, max_rows, use_spatial_index=True):
        self.gap = max(1, gap)
        self.link = max(0, link)
        self.min_run = max(1, min_run)
        self.max_rows = max(1, max_rows)
        self.active = []
        self.next_id = 0
        self.merged = 0  # диагностический счётчик, не влияет на связывание
        self.use_spatial_index = bool(use_spatial_index)
        self.indexed_rows = 0
        self.linear_rows = 0
        self.index_candidates = 0
        self.linear_candidates = 0

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
        """Match raw horizontal runs, preserving the original merge order.

        When many tracks or groups exist, use a conservative spatial index.
        Track lifetime X bounds are a *superset* of all frontier intervals,
        so it cannot exclude a real match. The exact `_near` test and the
        original active-list order still decide every association.
        """
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

        indexed = self.use_spatial_index and len(self.active) + len(groups) >= 24
        if indexed:
            self.indexed_rows += 1
            bucket_width = 64
            buckets = {}
            indexed_span = {}
            alive = {t.ident: t for t in self.active}
            order = {t.ident: i for i, t in enumerate(self.active)}
            next_order = len(order)

            def index_track(t):
                # Include the link distance. Expanding track bounds must
                # update buckets immediately for later groups on this row.
                a_bin = (t.x0 - self.link) // bucket_width
                b_bin = (t.x1 + self.link) // bucket_width
                existing = indexed_span.setdefault(t.ident, set())
                for k in range(a_bin, b_bin+1):
                    if k not in existing:
                        buckets.setdefault(k, []).append(t)
                        existing.add(k)

            for t in self.active:
                index_track(t)
        else:
            self.linear_rows += 1

        for a, b, raw in groups:
            if indexed:
                ids = set()
                for k in range(a // bucket_width, b // bucket_width + 1):
                    for candidate in buckets.get(k, ()):
                        if candidate.ident in alive:
                            ids.add(candidate.ident)
                candidates = [alive[k] for k in sorted(ids, key=order.__getitem__)]
                self.index_candidates += len(candidates)
            else:
                candidates = self.active
                self.linear_candidates += len(candidates)
            matches = [t for t in candidates
                       if self._near(a, b, t.frontier)
                       or self._near(a, b, t.new_frontier)]
            if matches:
                t = matches[0]
                for other in matches[1:]:
                    self._merge(t, other)
                    if indexed:
                        alive.pop(other.ident, None)
            else:
                self.next_id += 1
                t = _Track(self.next_id, row, row, a, b)
                self.active.append(t)
                if indexed:
                    alive[t.ident] = t
                    order[t.ident] = next_order
                    next_order += 1
            t.x0, t.x1 = min(t.x0, a), max(t.x1, b)
            t.new_frontier.append((a, b))
            t.area += sum(d - c for c, d in raw)
            if row - t.lo + 1 > self.max_rows:
                t.invalid = True
                t.runs.clear()
            if not t.invalid:
                t.runs.extend((row, c, d) for c, d in raw)
            if indexed:
                index_track(t)

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
    track_prefilter = bool(cfg.get('STREAM_TRACK_PREFILTER', False))
    snapshot_band = bool(cfg.get('STREAM_SNAPSHOT_BAND', False))
    tracker_index = bool(cfg.get('STREAM_TRACK_INDEX', False))
    # OPEN + CLOSE have a finite spatial support. Give the mask a zero halo.
    halo = 2 * (open_k // 2 + close_k // 2)

    class StreamingObjectDetector(base):
        def __init__(self, full_bg, piece_rows, bg_rows=None):
            super().__init__(full_bg, piece_rows, bg_rows)
            self._tracker = _Tracker(gap, link, min_run,
                                     self.buf_pieces * self.H - 2 * halo - self.H,
                                     use_spatial_index=tracker_index)
            self._stream_mode = False
            self._current_track = None
            self._stream_bg_src = None
            self.stream_finished = 0
            self.stream_emitted = 0
            self.stream_invalid = 0
            self.stream_prefilter_area = 0
            self.stream_prefilter_edge = 0
            self.stream_resets = 0
            self.stream_empty = 0
            self.stream_mask_ns = 0
            self.stream_track_ns = 0
            self.stream_pieces = 0
            self.stream_mask_max_ns = 0
            self.stream_track_max_ns = 0
            self.stream_onrun_n = 0
            self.stream_onrun_sum_ns = 0
            self.stream_onrun_max_ns = 0
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
            self._store_piece(piece, counts >= touch_pixels, t_end, lag)
            occupied = bool(hit.any())
            if not occupied:
                self.idle += 1
            elapsed_mask_ns = perf_counter_ns() - t0
            self.stream_mask_ns += elapsed_mask_ns
            self.stream_mask_max_ns = max(self.stream_mask_max_ns, elapsed_mask_ns)
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
            piece_track_ns = perf_counter_ns() - track_t0
            self.stream_track_ns += piece_track_ns
            for i, transitions in enumerate(changes):
                t0 = perf_counter_ns()
                ends = np.flatnonzero(transitions) + x0
                intervals = zip(ends[::2].tolist(), ends[1::2].tolist())
                completed = self._tracker.step(base_row + i, intervals)
                row_track_ns = perf_counter_ns() - t0
                self.stream_track_ns += row_track_ns
                piece_track_ns += row_track_ns
                for track in completed:
                    self.stream_finished += 1
                    if track.invalid or track.last - track.lo + 1 < min_rows:
                        self.rejected += 1
                        self.stream_invalid += 1
                        self._diag_track(track, 'too_long' if track.invalid else 'short')
                        continue
                    if track_prefilter:
                        # OPEN is anti-extensive. CLOSE with a symmetric flat
                        # kernel stays within the raw mask bounding rectangle
                        # (the mask has a zero halo). Consequently no output
                        # component can exceed this *bounding box area*.
                        # Do NOT use raw pixel count: CLOSE may fill holes.
                        max_area = (track.x1 - track.x0) * (track.last - track.lo + 1)
                        if max_area < min_area:
                            self.stream_prefilter_area += 1
                            self.stream_empty += 1
                            self.rejected += 1
                            self._diag_track(track, 'prefilter_area', max_area=max_area)
                            continue
                        # If the ENTIRE raw bounding box is within the
                        # forbidden side strip, no post-morph component can
                        # pass the unchanged edge test either.
                        if track.x1 <= x0 + edge or track.x0 >= x1 - edge:
                            self.stream_prefilter_edge += 1
                            self.stream_empty += 1
                            self.rejected += 1
                            self._diag_track(track, 'prefilter_edge')
                            continue
                    if self.on_run is not None:
                        self._current_track = track
                        onrun_start_ns = perf_counter_ns()
                        try:
                            self.on_run(track.lo, track.last + 1)
                        finally:
                            onrun_ns = perf_counter_ns() - onrun_start_ns
                            self.stream_onrun_n += 1
                            self.stream_onrun_sum_ns += onrun_ns
                            self.stream_onrun_max_ns = max(self.stream_onrun_max_ns,
                                                           onrun_ns)
                            self._current_track = None
            self.stream_track_max_ns = max(self.stream_track_max_ns, piece_track_ns)
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

        def snapshot_track(self, lo, hi):
            """Own real RGB rows and capture times before history advances.

            Called in the detector thread, while `_current_track` refers to
            the completed TRACK. Cropping itself can happen on another thread.
            """
            track = self._current_track
            if track is None:
                return None
            if lo < self._history_first or hi > self.total:
                self.rejected += 1
                self.stream_invalid += 1
                self._diag_track(track, 'history_missing')
                return None
            left, right = max(x0, track.x0 - halo), min(x1, track.x1 + halo)
            offset = left if snapshot_band else 0
            rows = (self._rows_band(lo, hi-lo, left, right)
                    if snapshot_band else self._rows(lo, hi-lo))
            return dict(track=track, lo=lo, hi=hi, x_offset=offset,
                        rows=rows,
                        timings=(self._snapshot_timings(lo, hi) if snapshot_band
                                 else tuple((self.row_time(r), self.row_lag(r))
                                            for r in range(lo, hi))),
                        diag_rec=getattr(self, '_diag_rec', None),
                        diag_entry=getattr(self, '_diag_entry', None))

        def extract_objects(self, lo, hi):
            track = self._current_track
            if track is None:
                return super().extract_objects(lo, hi)
            snap = self.snapshot_track(lo, hi)
            if snap is None:
                return [], 0
            return self.extract_snapshot(snap)

        def extract_snapshot(self, snap):
            """Pure crop over an immutable snapshot, safe in a worker thread.

            Do not refer to current TRACK, camera history, or current diagnostic
            piece: those change while the detector reads new camera blocks.
            """
            track = snap['track']
            lo, hi = snap['lo'], snap['hi']
            left, right = max(x0, track.x0 - halo), min(x1, track.x1 + halo)
            start = lo - halo
            stop = hi + halo
            mask = np.zeros((stop-start, right-left), np.uint8)
            for row, a, b in track.runs:
                mask[row-start, a-left:b-left] = 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.ker_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.ker_close)
            n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            found, at_edge = [], 0
            rec, entry = snap['diag_rec'], snap['diag_entry']
            diagnostic = rec is not None and entry is not None
            components = [] if diagnostic else None
            rows = snap['rows']
            for i in range(1, n):
                x, y, w, h, area = (int(v) for v in st[i])
                a, b = left+x, left+x+w
                r0, r1 = start+y, start+y+h
                component = None
                if diagnostic:
                    component = dict(x0=a,x1=b,row0=r0,row1=r1,area=area,reason='emitted')
                    components.append(component)
                if area < min_area:
                    if component is not None: component['reason'] = 'area'
                    continue
                if a < x0+edge or b > x1-edge:
                    at_edge += 1
                    if component is not None: component['reason'] = 'edge'
                    continue
                if r0 < lo or r1 > hi:
                    self.stream_invalid += 1
                    if component is not None: component['reason'] = 'outside_rows'
                    continue
                m = (lab[y:y+h, x:x+w] == i).astype(np.uint8)
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filled = np.zeros_like(m)
                cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
                crop = rows[r0-lo:r1-lo,
                            a-snap.get('x_offset', 0):b-snap.get('x_offset', 0)].copy()
                if median_kernel > 1:
                    crop = cv2.medianBlur(crop, median_kernel)
                crop[filled == 0] = 0
                t_front, lag = snap['timings'][r0-lo]
                t_last, _ = snap['timings'][r1-1-lo]
                found.append(dict(x0=a,x1=b,row0=r0,row1=r1,
                                  area=area,crop=crop,t_front=t_front,
                                  t_last=t_last,cam_lag_ms=lag,
                                  track_id=track.ident))
            if diagnostic:
                try:
                    transitions = np.diff(np.pad(mask != 0, ((0,0),(1,1))).astype(np.int8), axis=1)
                    ys, xs = np.nonzero(transitions)
                    post = None
                    if len(xs) <= 20000:
                        post = [[int(y), int(a), int(b)] for y,a,b in zip(ys[::2], xs[::2], xs[1::2])]
                    rec.track(entry, dict(track_id=track.ident, row0=track.lo,
                        row1=track.last+1, x0=track.x0, x1=track.x1,
                        raw_area=track.area, reason='segmented' if found else 'empty',
                        components=components, mask_origin=[start,left],
                        mask_shape=list(mask.shape), post_runs=post,
                        raw_runs=[list(r) for r in track.runs] if len(track.runs) <= 10000 else None,
                        emitted=len(found)))
                except Exception as e:
                    rec.error(e)
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
