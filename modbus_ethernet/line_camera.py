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
FIXED_RATE = 8000.0         # считываний в секунду
GAIN = 200                  # усиление сегментов сенсора

X0, X1 = 560, 1300          # полоса ленты, по ней ищем объект
BW = X1 - X0
ROWS_PER_SEC = FIXED_RATE * SENSOR_ROWS

# --------------------------------------------------------------------------
# Детекция
# --------------------------------------------------------------------------

BG_ROWS = 512               # строк пустого фона для калибровки
BG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "background.npy")

WIN_ROWS_MIN = 512          # окно в строках, число порций считается само
TH = 15                     # порог отличия от фона
MIN_AREA = 400              # площадь компоненты, пикселей
EDGE_NEW = 48               # отступ от нижнего (свежего) края окна
EDGE_OLD = 3                # отступ от верхнего (старого) края
OPEN_K, CLOSE_K = 3, 11     # морфология: убрать шум, залить дыры

# Кадр для приложения: всегда такой высоты, объект по центру.
FRAME_HEIGHT = 512
# Запас строк в кольцевом буфере сверх окна - из него режется кадр.
BUFFER_ROWS = 2048

QUEUE_SIZE = 64


class CameraError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Детектор. Работает порциями, без построчных циклов: при 16000 строк/с
# цикл по строкам в Python не успевал бы.
# --------------------------------------------------------------------------

class ObjectDetector:
    """Скользящее окно по порциям, кадр постоянной высоты на выходе."""

    def __init__(self, full_bg, piece_rows):
        self.full_bg = np.asarray(full_bg, np.uint8)        # (W, 3)
        self.band_bg = self.full_bg[X0:X1].astype(np.int16)  # (BW, 3)
        self.H = piece_rows

        self.win_pieces = max(2, -(-WIN_ROWS_MIN // piece_rows))
        self.win_rows = self.win_pieces * piece_rows
        self.buf_pieces = max(self.win_pieces * 2,
                              -(-BUFFER_ROWS // piece_rows))

        self.pieces = deque(maxlen=self.buf_pieces)   # полные порции
        self.masks = deque(maxlen=self.win_pieces)    # маски полосы
        self.has_fg = deque(maxlen=self.win_pieces)

        self.ker_open = np.ones((OPEN_K, OPEN_K), np.uint8)
        self.ker_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (CLOSE_K, CLOSE_K))
        self.total = 0              # сколько строк прошло всего
        self.seen = deque(maxlen=200)   # уже отданные объекты
        self.pending = []           # ждут строк снизу для центрирования
        self.idle = 0               # порций, пропущенных как пустые
        self.short_buffer_warned = False

    # --- маска порции ---------------------------------------------------

    def _mask(self, piece):
        """
        Маска отличия от фона для полосы ленты.

        Вторая строка - вычитание медианы по самой строке. Она убирает
        общие колебания подсветки: без неё дрейф яркости поднимает строку
        целиком выше порога, и фон принимается за объект.
        """
        belt = piece[:, X0:X1]
        d = belt.astype(np.int16) - self.band_bg
        d -= np.median(d[:, ::8], axis=1, keepdims=True).astype(np.int16)
        return (np.abs(d) > TH).any(axis=2).astype(np.uint8) * 255

    # --- приём порции ---------------------------------------------------

    def feed(self, piece):
        """Принимает порцию строк, возвращает готовые кадры."""
        self.pieces.append(piece)
        self.total += piece.shape[0]
        m = self._mask(piece)
        self.masks.append(m)
        self.has_fg.append(bool(cv2.countNonZero(m)))

        if len(self.masks) == self.win_pieces and any(self.has_fg):
            self._scan()
        else:
            self.idle += 1

        return self._ready()

    def _scan(self):
        """Ищет объекты в окне и ставит их в очередь на центрирование."""
        mask = np.vstack(self.masks)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.ker_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.ker_close)

        n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
        win_top = self.total - self.win_rows        # абсолютный номер строки

        for i in range(1, n):
            x, y, bw, bh, area = st[i]
            if area < MIN_AREA:
                continue
            # Объект должен целиком помещаться в окно. Снизу приходят свежие
            # строки, и объект там ещё доезжает, поэтому отступ большой.
            if y + bh > self.win_rows - EDGE_NEW:
                continue
            if y < EDGE_OLD:
                continue
            if x < EDGE_OLD or x + bw > BW - EDGE_OLD:
                continue

            abs_row = win_top + y + bh // 2
            col = x + bw // 2
            # Один объект виден в нескольких окнах подряд - берём раз.
            if any(abs(abs_row - r) < max(bh, 8) and abs(col - c) < 20
                   for r, c in self.seen):
                continue
            self.seen.append((abs_row, col))
            self.pending.append((abs_row, bh, area))

    # --- выдача кадра ---------------------------------------------------

    def _ready(self):
        """Отдаёт кадры, под которые уже набрались строки снизу."""
        out = []
        while self.pending:
            centre, bh, area = self.pending[0]
            lo = centre - FRAME_HEIGHT // 2
            if self.total < lo + FRAME_HEIGHT:
                break                       # строк снизу ещё мало
            self.pending.pop(0)
            out.append(self._window(lo, bh, area))
        return out

    def _window(self, lo, bh, area):
        """Вырезает FRAME_HEIGHT строк начиная с lo. Чего нет - ровный фон."""
        first = self.total - sum(p.shape[0] for p in self.pieces)
        if lo < first and not self.short_buffer_warned:
            print(f"  камера: буфер короток, часть кадра заполнена фоном "
                  f"(увеличьте BUFFER_ROWS)")
            self.short_buffer_warned = True

        frame = np.repeat(self.full_bg[None, :, :], FRAME_HEIGHT, axis=0)
        pos = first
        for piece in self.pieces:
            a, b = pos, pos + piece.shape[0]        # что покрывает порция
            pos = b
            src_lo, src_hi = max(a, lo), min(b, lo + FRAME_HEIGHT)
            if src_lo >= src_hi:
                continue
            frame[src_lo - lo:src_hi - lo] = piece[src_lo - a:src_hi - a]
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
        self.objects = 0

        self._latest = None
        self._latest_id = 0
        self._wanted_after = 0
        self._lock = threading.Lock()
        self._running = False
        self._threads = []

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

        ksj.KSJ_TriggerModeSet(IDX, TRIGGER_MODE)
        ksj.KSJ_SetFixedFrameRateEx(IDX, FIXED_RATE)
        ksj.KSJ_ExposureTimeSet(IDX, EXPOSURE_MS)
        ksj.KSJ_CaptureSetFieldOfViewEx(IDX, 0, 0, SENSOR_WIDTH, SENSOR_ROWS,
                                        0, 0, LINES)
        piece_sec = LINES / FIXED_RATE
        ksj.KSJ_CaptureSetTimeOut(IDX, int(piece_sec * 2000 + 5000))

        # Усиление и чёрный уровень по сегментам сенсора. Без этого кадр
        # может оказаться чёрным, а ряды - отличаться по яркости.
        segs = c_int()
        if ksj.KSJ_AWAIBA_GetSegmentNum(IDX, byref(segs)) >= 0:
            for sgm in range(segs.value):
                ksj.KSJ_AWAIBA_SetGain(IDX, sgm, GAIN)
                ksj.KSJ_AWAIBA_AutoBlackLevel(IDX, sgm)

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

    def _grabber(self):
        """Только забирает порции и кладёт в очередь, без обработки."""
        buf = create_string_buffer(self.width * self.height * self.channels)
        while self._running:
            rc = self.ksj.KSJ_CaptureRgbData(IDX, buf)
            if rc < 0:
                self.errors[rc] = self.errors.get(rc, 0) + 1
                if self.errors[rc] in (1, 100, 1000):
                    print(f"  камера: ошибка захвата {rc} "
                          f"(всего {self.errors[rc]})")
                continue
            self.grabbed += 1
            piece = np.frombuffer(buf, np.uint8).reshape(
                self.height, self.width, self.channels).copy()
            try:
                self.frames.put_nowait(piece)
            except queue.Full:
                self.errors["очередь"] = self.errors.get("очередь", 0) + 1

    def _detector(self, background):
        det = ObjectDetector(background, self.height)
        print(f"  окно {det.win_pieces} порций = {det.win_rows} строк "
              f"({det.win_rows / ROWS_PER_SEC * 1000:.0f} мс), "
              f"буфер {det.buf_pieces} порций")
        while self._running:
            try:
                piece = self.frames.get(timeout=1.0)
            except queue.Empty:
                continue
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
            if bg.shape == want:
                print(f"  фон загружен из {BG_FILE}")
                return bg.astype(np.uint8)
            print(f"  фон в {BG_FILE} имеет форму {bg.shape}, "
                  f"ожидалась {want}; пересобираю")

        print(f"  держите ленту пустой, собираю {BG_ROWS} строк фона...")
        rows = 0
        cal = []
        while rows < BG_ROWS:
            try:
                piece = self.frames.get(timeout=self.piece_sec * 6 + 10)
            except queue.Empty:
                raise CameraError("камера не отдаёт данные при сборе фона")
            cal.append(piece)
            rows += piece.shape[0]
        bg = np.median(np.vstack(cal)[:BG_ROWS], axis=0).astype(np.uint8)
        np.save(BG_FILE, bg)
        print(f"  фон снят: {np.round(bg[X0:X1].mean(axis=0), 1)}, "
              f"сохранён в {BG_FILE}")
        return bg

    # --- публичный интерфейс --------------------------------------------

    def start(self):
        self.open()
        self._running = True

        t = threading.Thread(target=self._grabber, daemon=True)
        t.start()
        self._threads.append(t)

        background = self._collect_background()

        t = threading.Thread(target=self._detector, args=(background,),
                             daemon=True)
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

    def status(self):
        return (f"порций {self.grabbed}, объектов {self.objects}, "
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
    """Кадр с объектом или None."""
    if _camera is None:
        return None
    return _camera.take()


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
    det0 = ObjectDetector(full_bg, piece_rows)
    print(f"порция {piece_rows} строк, окно {det0.win_rows} строк, "
          f"кадр {FRAME_HEIGHT} строк\n")

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


def main():
    if "--selftest" in sys.argv:
        raise SystemExit(0 if selftest() else 1)

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