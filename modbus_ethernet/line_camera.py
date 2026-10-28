"""
Съёмка объекта линейной камерой CatchBEST Jelly4.

Детекция повторяет рабочую версию capture_simple.py: скользящее окно из
последних порций, маска отличия от фона, морфология и связные компоненты.
Объект считается снятым, когда он целиком помещается в окно и не касается
его краёв.

Отличие от capture_simple.py одно: вместо вырезки по габаритам объекта
отдаётся кадр постоянной высоты FRAME_HEIGHT во всю ширину сенсора, где
объект стоит по центру, а вокруг - настоящий фон из того же потока.
Приложение ждёт именно 2048 пикселей в строке и берёт высоту из регистра
состояния, поэтому кадр должен быть предсказуемого размера.

Перед запуском нужна переменная окружения, иначе плагины камеры
не найдутся (читается при старте процесса, изнутри Python уже поздно):

    LD_LIBRARY_PATH=/opt/ksjapi/arm64 python3 dsp_protocol.py

Отдельно этот файл запускается для проверки камеры без сервера:

    LD_LIBRARY_PATH=/opt/ksjapi/arm64 python3 line_camera.py
    python3 line_camera.py --selftest      # детекция без железа
"""

import os
import queue
import sys
import threading
import time
import warnings
from collections import deque
from ctypes import (CDLL, byref, c_float, c_int, c_uint, c_ushort,
                    create_string_buffer)

import cv2
import numpy as np

if sys.platform == "win32":
    from ctypes import WinDLL
    SDK_DIR = (r"C:\Program Files (x86)\CatchBEST\IndustryCamera"
               r"\Applications\KSJShow\x64")
    LIB = "KSJApi64.dll"
else:
    SDK_DIR = "/opt/ksjapi/arm64"
    LIB = "libksjapi.so"

# --------------------------------------------------------------------------
# Камера
# --------------------------------------------------------------------------

IDX = 0
SENSOR_WIDTH = 2048
SENSOR_ROWS = 2             # физических ряда у сенсора Awaiba
LINES = 64                  # считываний сенсора в порции
EXPOSURE_MS = 0.1           # выдержка строки, мс
TRIGGER_MODE = 3            # 3 = фиксированная частота строк
# Считываний в секунду. Больше 4000 эта камера не держит: при 8000 порция
# приходит не раз в 8 мс, а раз в 16 - камера отдаёт каждый второй кадр, и
# кусок ленты между порциями пропадает целиком. Предметы на стыках порций
# от этого и резались. Проверено gapcheck.py --sweep: при 4000 и 3000 цикл
# совпадает с расчётом, при 6000 и 8000 он ровно вдвое длиннее.
FIXED_RATE = 4000.0
GAIN = 200                  # усиление сегментов сенсора

# Полоса, по которой ищутся предметы: вся освещённая лента, а не её
# середина. Строки кадра идут на всю ширину сенсора, поэтому и решать,
# есть ли в строке предмет, надо по всей ленте: предмет у края, которого
# детектор не видит, при сборке кадра режется пополам. На снимке лента
# освещена от x = 405 до 1854 (синий канал выше 80), берём с запасом.
X0, X1 = 420, 1840

# Строки внутри порции камера отдаёт в обратном порядке, сверху самая
# свежая. Порции переворачиваются сразу при захвате, см. _grabber.
FLIP_PIECES = True
# Перевернуть готовый кадр целиком перед отправкой. Это нужно, если в
# приложении предметы должны ехать в другую сторону. Целостность кадра
# от этого не зависит: переворот всего кадра - то же, что ваша идея
# переставить фрагменты в обратном порядке.
FLIP_OUTPUT = False
# Считать каждый второй столбец: предметы шире 20 пикселей, а работы
# вдвое меньше - полоса стала почти вдвое шире прежней.
COL_STEP = 2
BW = X1 - X0
ROWS_PER_SEC = FIXED_RATE * SENSOR_ROWS

# --------------------------------------------------------------------------
# Детекция
# --------------------------------------------------------------------------

BG_ROWS = 512               # строк пустого фона для калибровки
BG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "background.npy")
# Сами строки пустой ленты, а не только их медиана: ими заменяются соседи,
# которые торчат за край кадра. Это настоящий фон с настоящим шумом.
# Порядок строк в порции меняет и порядок строк фона, поэтому файл свой
# для каждого режима: строки, снятые до переворота, после него легли бы
# не той чётностью рядов сенсора.
BG_ROWS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "background_rows_flip.npy" if FLIP_PIECES
                            else "background_rows.npy")

# Построчный отбор. Строка считается занятой, если от фона отличается
# больше MIN_ROW_PIXELS пикселей. Серия занятых строк прерывается, когда
# подряд прошло GAP_ROWS пустых - запас нужен, чтобы тонкая перемычка
# предмета не разрывала его надвое. Серия становится предметом, если она
# не короче MIN_ROWS строк, набрала MIN_AREA пикселей и не задела края
# полосы (EDGE_OLD столбцов с каждой стороны).
MIN_ROW_PIXELS = 10
GAP_ROWS = 12
MIN_ROWS = 6
TH = 15                     # порог отличия от фона
MIN_AREA = 400              # пикселей в серии
EDGE_OLD = 3                # столбцов у краёв полосы

# Кадр для приложения: всегда такой высоты, объект по центру.
FRAME_HEIGHT = 512

# Сколько пустых строк оставить вплотную к объекту сверху и снизу,
# прежде чем добирать кадр фоном. Подробности в _pick_rows.
# Строка "задета", если от фона отличается хотя бы TOUCH_PIXELS пикселей -
# это мягче, чем порог занятости MIN_ROW_PIXELS, и ловит кончики предметов,
# где пикселей мало. Задетые строки дополнительно расширяются на
# TOUCH_DILATE строк в обе стороны. По ним решается, торчит ли сосед за
# край кадра: кончик соседа не должен остаться в кадре полоской.
TOUCH_PIXELS = 2
TOUCH_DILATE = 3

# Доля занятых строк, выше которой фон считается негодным: на пустой
# ленте маска обязана молчать, и занятость под сотню означает, что
# background.npy разошёлся с тем, что видит камера.
BG_SANE_FRACTION = 0.75

# На сколько уровней сохранённый фон может разойтись с текущей
# картинкой, прежде чем его пересобрать. Порог TH = 15, так что
# расхождение в треть порога уже заметно портит маску.
BG_MAX_DRIFT = 5.0
# Запас строк в кольцевом буфере сверх окна - из него режется кадр.
BUFFER_ROWS = 2048

QUEUE_SIZE = 64

# Порций, выбрасываемых сразу после открытия камеры: первые кадры она
# отдаёт неустоявшимися. Одной хватает с запасом, берём четыре - это
# меньше 40 мс при LINES = 64.
SKIP_PIECES = 4

# Ядро медианы, снимающей решётку с готового кадра. Только нечётное:
# 1 отключает фильтр, 3 - рабочее значение, 5 уже портит детализацию
# вдоль хода ленты. Подробности в take().
MEDIAN_KERNEL = 3

# --------------------------------------------------------------------------
# График яркости строки - то, что приложение читает из блоков 106 и 20366.
#
# Камера отдаёт байты 0..255, а приложение рисует график в шкале до 16384
# (так же, как её рисовал генератор до подключения камеры). Поэтому
# значения домножаются на 64: 255 * 64 = 16320.
# --------------------------------------------------------------------------

GRAPH_SCALE = 64            # множитель под шкалу приложения
GRAPH_MAX = 16383           # потолок после домножения
GRAPH_ALPHA = 0.05          # вес новой строки в усреднённом графике
GRAPH_PERIOD_SEC = 0.05     # как часто обновлять график, 20 раз в секунду


class CameraError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Детектор. Работает порциями, без построчных циклов: при 16000 строк/с
# цикл по строкам в Python не успевал бы.
# --------------------------------------------------------------------------

class ObjectDetector:
    """
    Построчный детектор.

    Для каждой строки считается одно число - сколько в полосе ленты
    пикселей отличается от фона. Строка занята, если их больше
    MIN_ROW_PIXELS. Подряд идущие занятые строки образуют серию; серия
    заканчивается, когда подряд прошло GAP_ROWS пустых строк.

    Серия и есть предмет - со всеми, кто проехал с ним бок о бок: они
    попадут в кадр целиком, потому что серия покрывает их все. Разрезать
    нечего, и границы порций ни на что не влияют.

    Ни скользящего окна, ни морфологии, ни связных компонент, ни
    отсеивания повторов: предмет виден ровно один раз - в тот момент,
    когда он закончился.
    """

    def __init__(self, full_bg, piece_rows, bg_rows=None):
        self.full_bg = np.asarray(full_bg, np.uint8)         # (W, 3)
        self.band_bg = self.full_bg[X0:X1:COL_STEP].astype(np.int16)
        # Записанные строки пустой ленты. Без них соседей заменяем
        # медианным фоном - ровнее настоящего, но годится.
        self.bg_rows = (None if bg_rows is None or len(bg_rows) < 2
                        else np.asarray(bg_rows, np.uint8))
        self.H = piece_rows
        self.buf_pieces = max(2, -(-BUFFER_ROWS // piece_rows))

        self.pieces = deque(maxlen=self.buf_pieces)
        self.touch = deque(maxlen=self.buf_pieces)   # задетые строки
        self.total = 0                  # сколько строк прошло всего

        self.run_lo = None              # начало текущей серии
        self.run_hi = None              # конец (за последней занятой)
        self.run_px = 0                 # пикселей в серии
        self.run_edge = False           # серия задела край полосы
        self.gap = 0                    # пустых строк после серии
        self.pending = []               # серии, ждущие строк снизу

        self.gaps = 0                   # разрывов потока
        self.idle = 0                   # пустых порций
        self.replaced = 0               # соседей, убранных с края кадра
        self.rejected = 0               # серий, не прошедших отбор
        self.busy_streak = 0        # порций подряд, занятых целиком
        self.rebuilds = 0           # пересборок фона на ходу
        self.bg_warned = False
        self.short_buffer_warned = False

    def _rebuild_background(self):
        """
        Пересобрать фон по последним строкам, не останавливая ленту.

        Если лента "занята" дольше, чем весь буфер, предметов таких не
        бывает: картинка разошлась с фоном - например, после сбоя захвата
        камера сменила яркость. Без пересборки серия занятых строк не
        кончается никогда, и ни один предмет больше не выходит.

        Медиана по каждому столбцу за 2048 строк отбрасывает предметы:
        в любом столбце они занимают меньше половины строк. Поэтому пустая
        лента для этого не нужна. Не сработает только при сплошном слое
        предметов - тогда фон надо снимать на пустой ленте, как при старте.
        """
        stack = np.concatenate(self.pieces)
        new = np.median(stack[::2], axis=0).astype(np.uint8)
        shift = np.abs(new[X0:X1].astype(np.int16)
                       - self.full_bg[X0:X1].astype(np.int16)).mean()
        self.full_bg = new
        self.band_bg = new[X0:X1:COL_STEP].astype(np.int16)
        # Записанные строки фона сняты при прежней картинке - больше не
        # годятся; соседей на краю кадра заменяем медианой, подогнанной
        # по яркости.
        self.bg_rows = None
        # Пересчитать задетые строки в буфере уже по новому фону и
        # забыть незаконченную серию - она была ложной.
        self.touch = deque((self.row_stats(p)[0] >= TOUCH_PIXELS
                            for p in self.pieces), maxlen=self.buf_pieces)
        self.run_lo = self.run_hi = None
        self.run_px, self.run_edge, self.gap = 0, False, 0
        self.pending.clear()
        self.busy_streak = 0
        self.rebuilds += 1
        print(f"  камера: лента {self.buf_pieces * self.H} строк подряд "
              f"считалась занятой - фон разошёлся с картинкой на "
              f"{shift:.1f} уровня. Фон пересобран по последним строкам "
              f"(пересборка {self.rebuilds}), детекция продолжается")

    def note_gap(self):
        """Потеряна порция. Текущую серию закрываем: в ней дыра."""
        self.gaps += 1
        self._close(force=True)

    # --- приём порции ---------------------------------------------------

    def row_stats(self, piece):
        """
        На каждую строку: сколько пикселей отличается от фона и задет ли
        край полосы. Вычитание медианы по строке убирает общий дрейф
        подсветки - без него яркость уводит строку целиком за порог.
        """
        d = piece[:, X0:X1:COL_STEP].astype(np.int16)
        d -= self.band_bg
        # Медиана строки через partition и вычисления на месте - на треть
        # быстрее np.median с копиями, результат тот же до строки.
        s = d[:, ::4]
        k = s.shape[1] // 2
        np.subtract(d, np.partition(s, k, axis=1)[:, k:k + 1], out=d)
        hit = np.abs(d, out=d) > TH
        hit = hit.any(axis=2)                        # (строки, ширина полосы)
        counts = hit.sum(axis=1) * COL_STEP          # в пикселях сенсора
        e = max(1, EDGE_OLD // COL_STEP)
        edge = hit[:, :e].any(axis=1) | hit[:, -e:].any(axis=1)
        return counts, edge

    def feed(self, piece):
        """Принимает порцию строк, возвращает готовые кадры."""
        counts, edge = self.row_stats(piece)
        busy = counts >= MIN_ROW_PIXELS

        self.pieces.append(piece)
        self.touch.append(counts >= TOUCH_PIXELS)
        base = self.total
        self.total += piece.shape[0]

        if not busy.any():
            self.idle += 1

        # Одна порция может быть занята целиком - крупный предмет её
        # закрывает. А вот если лента "занята" подряд дольше, чем буфер,
        # предметов таких не бывает: это разошёлся фон, и маска срабатывает
        # на пустой ленте. Тогда не работает ни отбор, ни сборка кадра.
        self.busy_streak = (self.busy_streak + 1
                            if busy.mean() > BG_SANE_FRACTION else 0)
        if self.busy_streak > self.buf_pieces:
            self._rebuild_background()
            return []

        for i in range(piece.shape[0]):
            if busy[i]:
                if self.run_lo is None:
                    self.run_lo = base + i
                self.run_hi = base + i + 1
                self.run_px += int(counts[i])
                self.run_edge |= bool(edge[i])
                self.gap = 0
            elif self.run_lo is not None:
                self.gap += 1
                if self.gap >= GAP_ROWS:
                    self._close()

        return self._ready()

    def _close(self, force=False):
        """Серия кончилась - решаем, предмет это или шум."""
        if self.run_lo is None:
            return
        lo, hi, px, edge = self.run_lo, self.run_hi, self.run_px, self.run_edge
        self.run_lo = self.run_hi = None
        self.run_px = 0
        self.run_edge = False
        self.gap = 0
        if force or edge or hi - lo < MIN_ROWS or px < MIN_AREA:
            self.rejected += 1
            return
        self.pending.append((lo, hi))

    # --- выдача кадра ---------------------------------------------------

    def _ready(self):
        """Отдаёт кадры, под которые уже набрались строки снизу."""
        out = []
        while self.pending:
            lo, hi = self.pending[0]
            centre = (lo + hi) // 2
            if self.total < centre + FRAME_HEIGHT // 2:
                break                       # строк снизу ещё мало
            self.pending.pop(0)
            out.append(self._window(lo, hi))
        return out

    def _rows(self, start, n):
        """n строк потока подряд, начиная с абсолютной строки start."""
        out = np.repeat(self.full_bg[None, :, :], n, axis=0)
        pos = self.total - sum(p.shape[0] for p in self.pieces)
        for piece in self.pieces:
            a, b = pos, pos + piece.shape[0]
            pos = b
            lo, hi = max(a, start), min(b, start + n)
            if lo < hi:
                out[lo - start:hi - start] = piece[lo - a:hi - a]
        return out

    def _bg_fill(self, start, n):
        """
        n строк пустого фона на место абсолютных строк start.. Строка фона
        берётся той же чётности: два ряда сенсора отличаются по отклику,
        и при сдвиге чётности на стыке проступила бы решётка.
        """
        if self.bg_rows is None:
            return np.repeat(self.full_bg[None, :, :], n, axis=0)
        k = self.bg_rows.shape[0] & ~1               # чётное число строк
        idx = (np.arange(start, start + n)) % k
        return self.bg_rows[idx]

    def _matched_fill(self, real, start):
        """
        Пустой фон на место строк real, подогнанный по яркости.

        Строки фона сняты при калибровке, а подсветка с тех пор могла
        уплыть или мерцает. Тогда подложка выделяется полосой. Но в
        заменяемых строках чужой предмет занимает лишь часть ширины,
        остальное там - настоящая лента, снятая в тот же миг. Медиана
        отношения "как есть / фон" по ленте показывает, во сколько раз
        ярче была подсветка, и подложка умножается на это число построчно.
        """
        fill = self._bg_fill(start, real.shape[0]).astype(np.float32)
        r = real[:, X0:X1:8].astype(np.float32)
        f = fill[:, X0:X1:8]
        ok = f > 20
        ratio = np.where(ok, r / np.maximum(f, 1), np.nan)
        with warnings.catch_warnings():
            # Строка без светлых столбцов даёт пустую медиану - это не
            # ошибка, такую строку просто не подгоняем (множитель 1).
            warnings.simplefilter("ignore", RuntimeWarning)
            gain = np.nanmedian(ratio, axis=1, keepdims=True)  # (строки,1,3)
        gain = np.nan_to_num(gain, nan=1.0)
        return np.clip(fill * gain, 0, 255).astype(np.uint8)

    def _window(self, lo, hi):
        """
        Кадр FRAME_HEIGHT строк - сплошной кусок потока вокруг серии.

        Строки внутри кадра больше не пропускаются: именно пропуски
        разрезали предметы, которых детектор не видел, - от них оставалась
        половина, а линия стыка была видна. Теперь кадр непрерывен.

        Соседи, которые торчат за верхний или нижний край кадра, целиком
        заменяются записанным пустым фоном: сосед либо весь в кадре, либо
        его нет совсем, половинок не бывает. Сам объект не трогается,
        даже если по задетым строкам он слился с соседом.
        """
        touch = np.concatenate(self.touch)
        first = self.total - touch.size
        centre = (lo + hi) // 2
        start = centre - FRAME_HEIGHT // 2
        start = max(first, min(start, self.total - FRAME_HEIGHT)) & ~1
        frame = self._rows(start, FRAME_HEIGHT)

        t = touch[start - first:start - first + FRAME_HEIGHT]
        if t.size < FRAME_HEIGHT:
            t = np.concatenate([t, np.zeros(FRAME_HEIGHT - t.size, bool)])
        if TOUCH_DILATE:
            k = 2 * TOUCH_DILATE + 1
            t = np.convolve(t.astype(np.uint8), np.ones(k, np.uint8),
                            "same") > 0

        edges = np.diff(np.concatenate([[0], t.astype(np.int8), [0]]))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)

        # Строки самого объекта с запасом на кончики - их не трогаем.
        guard = TOUCH_DILATE + 2
        keep_lo = lo - start - guard
        keep_hi = hi - start + guard

        for a, b in zip(starts, ends):
            if a > 0 and b < FRAME_HEIGHT:
                continue                        # целиком в кадре
            parts = []
            if a < keep_lo:
                parts.append((a, min(b, keep_lo)))
            if b > keep_hi:
                parts.append((max(a, keep_hi), b))
            for p0, p1 in parts:
                if p0 < p1:
                    frame[p0:p1] = self._matched_fill(frame[p0:p1],
                                                      start + p0)
            if parts:
                self.replaced += 1
        return frame


# --------------------------------------------------------------------------
# Камера
# --------------------------------------------------------------------------

class LineCamera:
    def __init__(self):
        self.ksj = None
        self.width = self.height = self.channels = 0
        self.frames = queue.Queue(maxsize=QUEUE_SIZE)
        self.errors = {}
        self.grabbed = 0
        self.dropped = 0
        self.gap_sum = self.gap_max = self.call_sum = 0.0
        self.gap_n = 0
        self.rate_warned = False
        self.recoveries = 0
        self.objects = 0
        self.detector = None

        self._latest = None
        self._latest_id = 0
        self._wanted_after = 0
        self._lock = threading.Lock()
        self._running = False
        self._threads = []

        # График: плоские массивы R,G,B по пикселям, готовые к отправке.
        self._live_graph = None
        self._avg_graph = None
        self._avg_acc = None            # накопитель усреднения, float32
        self._graph_at = 0.0            # когда обновляли последний раз

    def _update_graph(self, piece):
        """
        Готовит строку для графика. Вызывается не чаще GRAPH_PERIOD_SEC,
        чтобы не отнимать время у детекции.
        """
        now = time.monotonic()
        if now - self._graph_at < GRAPH_PERIOD_SEC:
            return
        self._graph_at = now

        # Камера отдаёт BGR, приложение ждёт R, G, B - разворачиваем каналы.
        line = piece[0][:, ::-1].astype(np.float32)

        if self._avg_acc is None:
            self._avg_acc = line.copy()
        else:
            self._avg_acc += GRAPH_ALPHA * (line - self._avg_acc)

        def pack(arr):
            v = np.clip(arr * GRAPH_SCALE, 0, GRAPH_MAX)
            return v.reshape(-1).astype(np.uint16)

        self._live_graph = pack(line)
        self._avg_graph = pack(self._avg_acc)

    # --- подключение ----------------------------------------------------

    def _load(self):
        path = os.path.join(SDK_DIR, LIB)
        if not os.path.exists(path):
            raise CameraError(f"нет библиотеки {path}")
        try:
            if sys.platform == "win32":
                os.add_dll_directory(SDK_DIR)
                ksj = WinDLL(path)
            else:
                ksj = CDLL(path)
        except OSError as e:
            raise CameraError(f"не загружается {path}: {e}")

        ksj.KSJ_ExposureTimeSet.argtypes = (c_int, c_float)
        ksj.KSJ_SetFixedFrameRateEx.argtypes = (c_int, c_float)
        ksj.KSJ_CaptureSetFieldOfViewEx.argtypes = (c_int,) * 7 + (c_ushort,)
        ksj.KSJ_CaptureSetTimeOut.argtypes = (c_int, c_uint)
        return ksj

    def _apply_settings(self, ksj):
        """
        Все настройки съёмки. Вызывается при открытии и повторно после
        ошибки захвата: SDK на сбое передачи может переинициализировать
        устройство, и тогда выдержка, усиление и чёрный уровень молча
        возвращаются к заводским - картинка меняется, и детекция слепнет.
        """
        ksj.KSJ_TriggerModeSet(IDX, TRIGGER_MODE)
        ksj.KSJ_SetFixedFrameRateEx(IDX, FIXED_RATE)
        ksj.KSJ_ExposureTimeSet(IDX, EXPOSURE_MS)
        ksj.KSJ_CaptureSetFieldOfViewEx(IDX, 0, 0, SENSOR_WIDTH, SENSOR_ROWS,
                                        0, 0, LINES)
        ksj.KSJ_CaptureSetTimeOut(IDX, int(LINES / FIXED_RATE * 2000 + 5000))

        # Усиление и чёрный уровень по сегментам сенсора. Без этого кадр
        # может оказаться чёрным, а ряды - отличаться по яркости.
        segs = c_int()
        if ksj.KSJ_AWAIBA_GetSegmentNum(IDX, byref(segs)) >= 0:
            for sgm in range(segs.value):
                ksj.KSJ_AWAIBA_SetGain(IDX, sgm, GAIN)
                ksj.KSJ_AWAIBA_AutoBlackLevel(IDX, sgm)

    def open(self):
        ksj = self._load()
        ksj.KSJ_Init()

        if ksj.KSJ_DeviceGetCount() < 1:
            ksj.KSJ_UnInit()
            raise CameraError("камера не найдена (питание, USB3-порт, udev)")

        # В восьмой версии библиотеки устройство открывается явно,
        # в шестой это происходило само. Вызов безвреден для обеих.
        if hasattr(ksj, "KSJ_Open"):
            ksj.KSJ_Open(IDX)

        self._apply_settings(ksj)
        piece_sec = LINES / FIXED_RATE

        w, h, bits = c_int(), c_int(), c_int()
        ksj.KSJ_CaptureGetSizeEx(IDX, byref(w), byref(h), byref(bits))
        self.width, self.height = w.value, h.value
        self.channels = bits.value // 8
        if self.width <= 0 or self.height <= 0:
            ksj.KSJ_UnInit()
            raise CameraError(f"камера вернула размер {self.width}x{self.height}")

        self.ksj = ksj
        self.piece_sec = piece_sec
        print(f"  камера: порция {self.width}x{self.height} "
              f"({piece_sec*1000:.1f} мс), {ROWS_PER_SEC:.0f} строк/с")

    def close(self):
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads = []
        if self.ksj is not None:
            if hasattr(self.ksj, "KSJ_Close"):
                self.ksj.KSJ_Close(IDX)
            self.ksj.KSJ_UnInit()
            self.ksj = None

    # --- потоки ---------------------------------------------------------

    def _check_rate(self):
        """
        Держит ли камера заданную частоту.

        Если порция приходит заметно реже, чем по расчёту, камера не
        успевает и отдаёт не каждый кадр: строки между порциями теряются, а
        предметы на стыках режутся. Снаружи этого не видно никак - картинка
        выглядит нормально, пока через стык не проедет предмет. Поэтому
        проверяем сами и говорим прямо.
        """
        if self.rate_warned or self.gap_n < 50 or self.gap_n % 50:
            return
        cycle = (self.call_sum + self.gap_sum) / self.gap_n
        expect = self.piece_sec * 1000
        if cycle > expect * 1.3:
            real = LINES / cycle * 1000
            print(f"  камера: порция приходит раз в {cycle:.1f} мс вместо "
                  f"{expect:.1f} - камера не держит FIXED_RATE = "
                  f"{FIXED_RATE:.0f} и пропускает кадры, предметы на стыках "
                  f"порций будут резаться. Реально выходит {real:.0f} "
                  f"считываний в секунду: снизьте FIXED_RATE до этого "
                  f"значения или ниже (проверка: gapcheck.py --sweep)")
            self.rate_warned = True

    def _grabber(self):
        """Только забирает порции и кладёт в очередь, без обработки."""
        buf = create_string_buffer(self.width * self.height * self.channels)
        seq = 0
        skip = SKIP_PIECES
        last_end = None
        failed = False          # была ошибка, ждём первую удачную порцию
        level_ok = None         # яркость ленты до сбоя
        level_check = False     # сравнить яркость после восстановления
        while self._running:
            t0 = time.perf_counter()
            if last_end is not None and not skip:
                # Время между возвратом из захвата и следующим захватом.
                # Если камера снимает только во время вызова, строки за это
                # время теряются, и предметы на стыках порций режутся.
                g = (t0 - last_end) * 1000
                self.gap_sum += g
                self.gap_n += 1
                self.gap_max = max(self.gap_max, g)
            rc = self.ksj.KSJ_CaptureRgbData(IDX, buf)
            last_end = time.perf_counter()
            if not skip:
                self.call_sum += (last_end - t0) * 1000
                self._check_rate()
            if rc < 0:
                self.errors[rc] = self.errors.get(rc, 0) + 1
                if self.errors[rc] in (1, 100, 1000):
                    print(f"  камера: ошибка захвата {rc} "
                          f"(всего {self.errors[rc]})")
                failed = True
                continue
            if failed:
                # Первая удачная порция после сбоя. SDK мог переоткрыть
                # устройство с заводскими настройками - ставим свои заново,
                # снова пропускаем порции на прогрев и сообщаем детектору
                # о разрыве потока, чтобы он не склеил строки до и после.
                failed = False
                self.recoveries += 1
                self._apply_settings(self.ksj)
                skip = SKIP_PIECES
                seq += 1
                level_check = True
                print(f"  камера: после ошибки захвата настройки применены "
                      f"заново (восстановление {self.recoveries})")
                continue
            # Первые порции после открытия камера отдаёт неустоявшимися:
            # в сыром дампе видно, как на 23-й паре строк чётные столбцы
            # разом проседают на 40 процентов. Если фон уже лежит в файле,
            # сбор фона порции не забирает, и этот мусор уходит прямо
            # детектору - он видит огромное отличие от фона и находит
            # несуществующий объект. Поэтому выбрасываем их здесь, до всех
            # потребителей сразу.
            if skip:
                skip -= 1
                if skip == 0:
                    print(f"  камера: пропущено {SKIP_PIECES} порций "
                          f"на прогрев")
                continue

            self.grabbed += 1
            seq += 1
            piece = np.frombuffer(buf, np.uint8).reshape(
                self.height, self.width, self.channels)
            # Камера отдаёт строки порции в обратном порядке: сверху самая
            # свежая. Без переворота время внутри порции идёт снизу вверх,
            # а от порции к порции - сверху вниз, и предмет на стыке
            # распадается на две половинки в 256 строках друг от друга.
            # Перевернув порцию, получаем сплошной поток сверху вниз.
            piece = (piece[::-1] if FLIP_PIECES else piece).copy()

            # Яркость ленты - чтобы после сбоя было видно, что изменилось.
            level = piece[::16, X0:X1:16].reshape(-1, 3).mean(axis=0)
            if level_check:
                level_check = False
                was = ("?" if level_ok is None
                       else " ".join(f"{v:.0f}" for v in level_ok))
                now = " ".join(f"{v:.0f}" for v in level)
                print(f"  камера: яркость ленты B G R до сбоя {was}, "
                      f"после восстановления {now}")
            else:
                level_ok = level
            try:
                # Номер порции нужен детектору: по пропуску в нумерации он
                # понимает, что поток разорвался, и сбрасывает окно.
                self.frames.put_nowait((seq, piece))
            except queue.Full:
                self.dropped += 1
                if self.dropped in (1, 10, 100, 1000):
                    print(f"  камера: детекция не успевает, порция потеряна "
                          f"(всего {self.dropped}); "
                          f"увеличьте LINES или уменьшите FIXED_RATE")

    def _detector(self, background):
        det = ObjectDetector(background, self.height,
                             getattr(self, "bg_rows", None))
        self.detector = det
        print(f"  буфер {det.buf_pieces} порций = "
              f"{det.buf_pieces * self.height} строк "
              f"({det.buf_pieces * self.height / ROWS_PER_SEC * 1000:.0f} мс), "
              f"серия рвётся на {GAP_ROWS} пустых строках")
        expected = None
        while self._running:
            try:
                seq, piece = self.frames.get(timeout=1.0)
            except queue.Empty:
                continue

            if expected is not None and seq != expected:
                det.note_gap()
                if det.gaps in (1, 10, 100):
                    print(f"  камера: разрыв потока, пропущено "
                          f"{seq - expected} порций подряд "
                          f"(всего разрывов {det.gaps}); "
                          f"от пяти подряд объект теряется")
            expected = seq + 1
            self._update_graph(piece)

            for frame in det.feed(piece):
                self.objects += 1
                with self._lock:
                    self._latest = frame
                    self._latest_id = self.objects
                print(f"  камера: объект {self.objects}, "
                      f"кадр {frame.shape[1]}x{frame.shape[0]}")

    # --- фон ------------------------------------------------------------

    def _collect_background(self):
        """
        Медиана строк пустой ленты, на всю ширину сенсора.
        Полоса поиска X0..X1 получается из неё срезом, поэтому
        рассогласования ширины быть не может.
        """
        want = (self.width, self.channels)
        if os.path.exists(BG_FILE):
            bg = np.load(BG_FILE)
            if bg.shape != want:
                print(f"  фон в {BG_FILE} имеет форму {bg.shape}, "
                      f"ожидалась {want}; пересобираю")
            else:
                # Форма сошлась - проверяем, что фон ещё и похож на то, что
                # камера видит сейчас. Усиление, выдержка и свет меняются, а
                # файл остаётся; разошедшийся фон ломает и детекцию, и сборку
                # кадра, причём молча. Дешевле пересобрать, чем ловить потом.
                # В очереди лежат пары (номер порции, строки) - номер нужен
                # детектору, чтобы замечать разрывы потока.
                _, piece = self.frames.get(timeout=self.piece_sec * 6 + 10)
                now = np.median(piece, axis=0).astype(np.int16)
                diff = np.abs(now - bg.astype(np.int16))[X0:X1].mean()
                if diff <= BG_MAX_DRIFT:
                    print(f"  фон загружен из {BG_FILE} "
                          f"(расхождение {diff:.1f} уровня)")
                    self.bg_rows = None
                    if os.path.exists(BG_ROWS_FILE):
                        r = np.load(BG_ROWS_FILE)
                        if r.ndim == 3 and r.shape[1:] == want:
                            self.bg_rows = r
                    if self.bg_rows is None:
                        print(f"  строк фона ({os.path.basename(BG_ROWS_FILE)})"
                              f" нет - соседей буду заменять медианным "
                              f"фоном; чтобы были настоящие строки, удалите "
                              f"{os.path.basename(BG_FILE)} и пересоберите")
                    return bg.astype(np.uint8)
                print(f"  фон из {BG_FILE} разошёлся с картинкой на "
                      f"{diff:.1f} уровня (порог {BG_MAX_DRIFT}); пересобираю")

        print(f"  держите ленту пустой, собираю {BG_ROWS} строк фона...")
        rows = 0
        cal = []
        while rows < BG_ROWS:
            try:
                _, piece = self.frames.get(timeout=self.piece_sec * 6 + 10)
            except queue.Empty:
                raise CameraError("камера не отдаёт данные при сборе фона")
            cal.append(piece)
            rows += piece.shape[0]
        rows_bg = np.vstack(cal)[:BG_ROWS]
        bg = np.median(rows_bg, axis=0).astype(np.uint8)
        np.save(BG_FILE, bg)
        np.save(BG_ROWS_FILE, rows_bg)
        self.bg_rows = rows_bg
        print(f"  фон снят: {np.round(bg[X0:X1].mean(axis=0), 1)}, "
              f"сохранён в {BG_FILE}")
        return bg

    # --- публичный интерфейс --------------------------------------------

    def start(self):
        self.open()
        self._running = True

        t = threading.Thread(target=self._grabber, daemon=True,
                             name="захват")
        t.start()
        self._threads.append(t)

        background = self._collect_background()

        t = threading.Thread(target=self._detector, args=(background,),
                             daemon=True, name="детекция")
        t.start()
        self._threads.append(t)
        print("  камера готова, можно подавать предметы")

    def arm(self):
        """Ждём следующий объект: снятое раньше не считается."""
        with self._lock:
            self._wanted_after = self._latest_id

    def take(self):
        """Кадр с объектом (FRAME_HEIGHT, 2048, 3) в BGR или None."""
        with self._lock:
            if self._latest_id <= self._wanted_after:
                return None
            self._wanted_after = self._latest_id
            return self._latest

    def live_graph(self):
        return self._live_graph

    def average_graph(self):
        return self._avg_graph

    def status(self):
        d = self.detector
        gaps = d.gaps if d else 0
        idle = d.idle if d else 0
        # Отсеяно - серий, не прошедших отбор: короткие, мелкие или задевшие
        # край полосы. Соседей убрано - сколько раз предмет, торчавший за
        # край кадра, был заменён пустым фоном вместо того, чтобы резаться.
        extra = ""
        if d and d.rejected:
            extra += f", отсеяно {d.rejected}"
        if d and d.replaced:
            extra += f", соседей убрано с края {d.replaced}"
        if d and d.rebuilds:
            extra += f", фон пересобран {d.rebuilds} раз"
        if self.recoveries:
            extra += f", восстановлений после ошибки {self.recoveries}"
        if self.gap_n:
            extra += (f", захват {self.call_sum / self.gap_n:.2f} мс, "
                      f"между захватами сред {self.gap_sum / self.gap_n:.3f}"
                      f" / макс {self.gap_max:.2f} мс")
        # Поток, умерший на исключении, иначе ничем себя не выдаёт: сервер
        # отвечает, а снимка нет. Трассировка его падения - в логе.
        dead = [t.name for t in self._threads if not t.is_alive()]
        if dead and self._running:
            extra += f", ПОТОК УМЕР: {', '.join(dead)} (трассировка в логе)"
        return (f"порций {self.grabbed}, потеряно {self.dropped}, "
                f"разрывов {gaps}, пустых порций {idle}, "
                f"объектов {self.objects}{extra}, "
                f"ошибки {self.errors or 'нет'}")


# --------------------------------------------------------------------------
# Один экземпляр на программу
# --------------------------------------------------------------------------

_camera = None
_start_error = None
_thread = None


def _startup():
    """Открытие камеры и сбор фона. Занимает секунды, поэтому в потоке."""
    global _camera, _start_error
    try:
        cam = LineCamera()
        cam.start()
        _camera = cam
    except CameraError as e:
        _start_error = str(e)
        print(f"  камера недоступна: {e}")
    except Exception as e:                   # неожиданность не должна ронять
        _start_error = repr(e)               # сервер вместе с собой
        print(f"  камера не запустилась: {e!r}")


def begin():
    """
    Запускает камеру в фоне и сразу возвращается.

    Запускать нужно при старте программы, а не из обработчика запроса:
    открытие камеры и сбор фона занимают секунды, и если делать это внутри
    обработчика, сервер на всё это время перестаёт отвечать приложению.
    """
    global _thread
    if _thread is not None:
        return
    _thread = threading.Thread(target=_startup, daemon=True)
    _thread.start()


def ready() -> bool:
    """Камера поднялась и снимает."""
    return _camera is not None


def failed() -> bool:
    """Камера не поднялась и больше не поднимется."""
    return _start_error is not None


def arm():
    """Ждём следующий объект. Пока камера не готова - ничего не делает."""
    if _camera is not None:
        _camera.arm()


def take():
    """
    Кадр с объектом или None.

    Здесь же снимается решётка: у двух рядов сенсора разная спектральная
    чувствительность, из-за чего соседние строки отличаются по зелёному
    примерно на 20 уровней. Настройками камеры это не лечится - усиление и
    чёрный уровень действуют на весь кадр и отношение рядов не меняют.
    Медиана ядром 3 убирает 93 процента решётки за 0.7 мс, теряя около
    7 процентов резкости; ядро 5 режет вертикальную детализацию втрое
    сильнее ради лишних полпроцента решётки и потому не берётся.

    Фильтр стоит именно здесь, а не в capture.py: через эту функцию идут
    оба пути - и сервер, и прямой запуск line_camera.py, - поэтому кадр,
    сохранённый локально, совпадает с тем, что получает приложение.
    Поток, по которому работает детектор, фильтр не трогает: там решётку
    и так снимает построчное вычитание медианы в _mask.
    """
    if _camera is None:
        return None
    frame = _camera.take()
    if frame is None:
        return None
    if FLIP_OUTPUT:
        frame = np.ascontiguousarray(frame[::-1])
    if MEDIAN_KERNEL <= 1:
        return frame
    return cv2.medianBlur(frame, MEDIAN_KERNEL)


def live_graph():
    """Текущая строка для графика: R,G,B подряд по пикселям. None, если нет."""
    return None if _camera is None else _camera.live_graph()


def average_graph():
    """Усреднённая строка для графика. None, если камеры нет."""
    return None if _camera is None else _camera.average_graph()


def status():
    if _camera is None:
        if _start_error:
            return _start_error
        return "запускается" if _thread is not None else "не запущена"
    return _camera.status()


def stop():
    global _camera
    if _camera is not None:
        _camera.close()
        _camera = None


# --------------------------------------------------------------------------
# Проверки
# --------------------------------------------------------------------------

def selftest():
    """Проверяет детекцию, высоту кадра и центрирование без камеры."""
    rng = np.random.default_rng(0)
    w, ch = SENSOR_WIDTH, 3
    piece_rows = LINES * SENSOR_ROWS
    full_bg = np.full((w, ch), 90, np.uint8)

    def stream(det, specs):
        """specs: список (строк, есть ли объект). Возвращает кадры."""
        out = []
        for rows, obj, *rest in specs:
            drift = rest[0] if rest else 0
            done = 0
            while done < rows:
                n = min(piece_rows, rows - done)
                p = rng.integers(85, 96, size=(n, w, ch), dtype=np.int16)
                p += drift                      # общий дрейф подсветки
                if obj:
                    p[:, X0 + 150:X0 + 450] = rng.integers(
                        180, 230, size=(n, 300, ch))
                p = np.clip(p, 0, 255).astype(np.uint8)
                if p.shape[0] < piece_rows:     # порция всегда целая
                    pad = np.repeat(p[-1:], piece_rows - p.shape[0], axis=0)
                    p = np.vstack([p, pad])
                out.extend(det.feed(p))
                done += piece_rows
        return out

    def object_rows(frame):
        band = frame[:, X0:X1].astype(np.int16)
        d = band - full_bg[X0:X1].astype(np.int16)
        d -= np.median(d[:, ::8], axis=1, keepdims=True).astype(np.int16)
        fg = (np.abs(d) > TH).any(axis=2).sum(axis=1)
        rows = np.flatnonzero(fg >= 20)
        return (int(rows[0]), int(rows[-1])) if rows.size else None

    ok = True
    print(f"порция {piece_rows} строк, кадр {FRAME_HEIGHT} строк, "
          f"разрыв серии {GAP_ROWS} строк\n")

    for length in (64, 128, 256):
        det = ObjectDetector(full_bg, piece_rows)
        found = stream(det, [(512, False), (length, True), (1536, False)])
        if len(found) != 1:
            print(f"  объект {length:4d} строк: кадров {len(found)}, ожидался 1")
            ok = False
            continue
        frame = found[0]
        span = object_rows(frame)
        if span is None:
            print(f"  объект {length:4d} строк: в кадре его нет")
            ok = False
            continue
        top, bottom = span
        offset = (top + bottom) / 2 - FRAME_HEIGHT / 2
        good = (frame.shape == (FRAME_HEIGHT, w, ch) and abs(offset) <= 24)
        ok = ok and good
        print(f"  объект {length:4d} строк -> кадр {frame.shape[0]}x"
              f"{frame.shape[1]}, строки {top}..{bottom}, "
              f"смещение центра {offset:+.0f}  {'ок' if good else 'ПЛОХО'}")

    # Главная проверка: дрейф подсветки не должен приниматься за объект.
    det = ObjectDetector(full_bg, piece_rows)
    drifted = stream(det, [(512, False, 0), (1024, False, 25),
                           (1024, False, -25), (512, False, 0)])
    print(f"\n  дрейф подсветки на +-25 уровней: кадров {len(drifted)} "
          f"(ожидалось 0)")
    ok = ok and len(drifted) == 0

    # Шум без объекта тоже не должен срабатывать.
    det = ObjectDetector(full_bg, piece_rows)
    noise = stream(det, [(4096, False)])
    print(f"  чистый фон 4096 строк: кадров {len(noise)} (ожидалось 0)")
    ok = ok and len(noise) == 0

    # Два объекта подряд не должны потеряться и не должны задвоиться.
    det = ObjectDetector(full_bg, piece_rows)
    two = stream(det, [(512, False), (128, True), (768, False),
                       (128, True), (1536, False)])
    print(f"  два объекта подряд: кадров {len(two)} (ожидалось 2)")
    ok = ok and len(two) == 2

    print("\nдетектор:", "работает" if ok else "НЕ РАБОТАЕТ")
    return ok


def show_graph():
    """
    Живой график яркости строки, как в graph.py. Нужен дисплей.

    Берёт ту же строку, что уходит приложению, поэтому годится для
    настройки освещения и усиления: что видно здесь, то увидит и
    приложение в блоке 106.
    """
    import matplotlib.pyplot as plt

    begin()
    while not ready() and not failed():
        time.sleep(0.1)
    if failed():
        raise SystemExit(1)

    x = np.arange(SENSOR_WIDTH)
    plt.ion()
    fig, ax = plt.subplots()
    lines = [ax.plot(x, np.zeros(SENSOR_WIDTH), c + '-', label=n)[0]
             for c, n in (('r', 'R'), ('g', 'G'), ('b', 'B'))]
    ax.set_ylim(0, GRAPH_MAX)
    ax.set_xlim(0, SENSOR_WIDTH)
    ax.set_xlabel('Линия пикселей')
    ax.set_ylabel('Яркость')
    ax.legend()

    while plt.fignum_exists(fig.number):
        flat = live_graph()
        if flat is None:
            plt.pause(0.05)
            continue
        rgb = flat.reshape(SENSOR_WIDTH, 3)
        for i, ln in enumerate(lines):
            ln.set_ydata(rgb[:, i])
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.02)
    stop()


def main():
    if "--selftest" in sys.argv:
        raise SystemExit(0 if selftest() else 1)
    if "--graph" in sys.argv:
        show_graph()
        return

    begin()
    while not ready() and not failed():
        time.sleep(0.1)
    if failed():
        raise SystemExit(1)
    print("жду предметы, Ctrl+C для выхода")
    arm()
    saved = 0
    try:
        while True:
            frame = take()
            if frame is not None:
                name = "frame_%03d.png" % saved
                cv2.imwrite(name, frame)
                saved += 1
                print(f"  сохранено {name}  [{status()}]")
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n" + status())
    finally:
        stop()


if __name__ == "__main__":
    main()