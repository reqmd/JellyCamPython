"""
Съёмка объекта линейной камерой CatchBEST Jelly4.

Камера непрерывно снимает ленту в фоновом потоке. Когда мимо проезжает
предмет, он обнаруживается по отличию от фона, строки накапливаются, и
готовый кадр кладётся в слот. Сервер забирает его из слота - не блокируя
себя ожиданием, чтобы продолжать отвечать на опрос приложения.

Перед запуском нужна переменная окружения, иначе плагины камеры
не найдутся (читается при старте процесса, изнутри Python уже поздно):

    LD_LIBRARY_PATH=/opt/ksjapi/arm64 python3 dsp_protocol.py

Отдельно этот файл запускается для проверки камеры без сервера:

    LD_LIBRARY_PATH=/opt/ksjapi/arm64 python3 line_camera.py
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

# --------------------------------------------------------------------------
# Настройки камеры. Значения подобраны под конкретную установку:
# FIXED_RATE привязан к скорости ленты, X0/X1 - к положению камеры.
# --------------------------------------------------------------------------

SDK_DIR = "/opt/ksjapi/arm64"
LIB = "libksjapi.so"
IDX = 0                     # индекс камеры

SENSOR_WIDTH = 2048         # пикселей в строке
SENSOR_ROWS = 2             # физических рядов у сенсора Awaiba
LINES = 128                 # считываний сенсора в одной порции
EXPOSURE_MS = 0.75          # выдержка строки, мс
TRIGGER_MODE = 3            # 3 = фиксированная частота строк
FIXED_RATE = 134.0          # считываний в секунду, под скорость ленты

# --------------------------------------------------------------------------
# Детекция объекта. Полоса X0..X1 - та часть кадра, где едет лента.
# Отдаётся при этом кадр целиком, во всю ширину сенсора: приложение
# ждёт именно 2048 пикселей.
# --------------------------------------------------------------------------

X0, X1 = 560, 1300          # полоса, по которой ищем объект
BG_ROWS = 512               # строк пустого фона для калибровки
BG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "background.npy")

TH = 12                     # порог отличия от фона
MIN_FG_PIXELS = 12          # пикселей в строке, чтобы считать объект начавшимся
KEEP_PIX = 4                # пикселей, чтобы продолжать (тонкие места)
GAP_ROWS = 30               # пустых строк подряд = объект кончился
MIN_ROWS = 15               # короче - считаем шумом
MAX_ROWS = 1000             # длиннее - обрываем

# Кадр всегда одной высоты, объект стоит в нём по центру, вокруг - настоящий
# фон из того же потока. Поэтому после объекта надо дождаться ещё строк,
# прежде чем вырезать окно.
FRAME_HEIGHT = 512

# Кольцевой буфер последних строк, из него и вырезается окно. Нужен запас:
# половина длины объекта плюс половина кадра плюс пауза в конце.
BUFFER_ROWS = 1024

QUEUE_SIZE = 64


class CameraError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Конечный автомат детекции. Вынесен отдельно, чтобы его можно было
# проверить без камеры - см. функцию selftest() внизу файла.
# --------------------------------------------------------------------------

class ObjectDetector:
    """
    Находит объекты в потоке строк и отдаёт кадры фиксированной высоты.

    Строки копятся в кольцевом буфере. Когда объект закончился, известны
    его первая и последняя строка, и из буфера вырезается окно высотой
    FRAME_HEIGHT так, чтобы середина объекта пришлась на середину кадра.
    Вокруг объекта оказывается настоящий фон из того же потока, а не
    подрисованный.

    Отсюда задержка: пока не наберутся строки ниже объекта, вырезать
    нечего. Поэтому кадр отдаётся не сразу после объекта, а через
    (FRAME_HEIGHT / 2 - длина объекта / 2) строк.
    """

    def __init__(self, band_bg, full_bg):
        self.band_bg = band_bg.astype(np.int16)     # фон полосы поиска
        self.full_bg = full_bg.astype(np.uint8)     # фон строки целиком
        self.buf = deque(maxlen=BUFFER_ROWS)
        self.total = 0              # сколько строк прошло всего
        self.collecting = False
        self.start = None           # первая строка объекта
        self.last_fg = None         # последняя строка с объектом
        self.gap = 0
        self.pending = []           # объекты, ждущие строк снизу
        self.short_buffer_warned = False

    def _foreground(self, row):
        """Сколько пикселей строки отличается от фона."""
        band = row[X0:X1].astype(np.int16)
        diff = np.abs(band - self.band_bg)
        return int((diff > TH).any(axis=1).sum())

    def feed(self, piece):
        """Принимает порцию строк, возвращает готовые кадры."""
        done = []
        for i in range(piece.shape[0]):
            row = piece[i]
            self.buf.append(row)
            self.total += 1
            idx = self.total - 1
            fg = self._foreground(row)

            if not self.collecting:
                if fg >= MIN_FG_PIXELS:
                    self.collecting = True
                    self.start = self.last_fg = idx
                    self.gap = 0
            else:
                if fg >= KEEP_PIX:
                    self.last_fg = idx
                    self.gap = 0
                else:
                    self.gap += 1
                if self.gap >= GAP_ROWS or idx - self.start >= MAX_ROWS:
                    length = self.last_fg - self.start + 1
                    if length >= MIN_ROWS:
                        self.pending.append((self.start, self.last_fg))
                    self.collecting = False
                    self.start = self.last_fg = None
                    self.gap = 0

            done.extend(self._ready())
        return done

    def _ready(self):
        """Отдаёт кадры, для которых уже набрались строки снизу."""
        out = []
        while self.pending:
            start, end = self.pending[0]
            centre = (start + end + 1) // 2
            lo = centre - FRAME_HEIGHT // 2
            if self.total < lo + FRAME_HEIGHT:
                break                           # строк снизу ещё мало
            self.pending.pop(0)
            out.append(self._window(lo, end - start + 1))
        return out

    def _window(self, lo, obj_rows):
        """Вырезает FRAME_HEIGHT строк начиная с lo. Чего нет - фон."""
        first = self.total - len(self.buf)       # индекс самой старой строки
        if lo < first and not self.short_buffer_warned:
            print(f"  камера: буфер короток для объекта в {obj_rows} строк, "
                  f"часть кадра заполнена фоном "
                  f"(увеличьте BUFFER_ROWS)")
            self.short_buffer_warned = True

        rows = []
        for a in range(lo, lo + FRAME_HEIGHT):
            if first <= a < self.total:
                rows.append(self.buf[a - first])
            else:
                rows.append(self.full_bg)        # за краями потока - ровный фон
        return np.stack(rows)


# --------------------------------------------------------------------------
# Камера
# --------------------------------------------------------------------------

class LineCamera:
    """Фоновая съёмка с детекцией объектов."""

    def __init__(self):
        self.ksj = None
        self.width = self.height = self.channels = 0
        self.frames = queue.Queue(maxsize=QUEUE_SIZE)
        self.errors = {}
        self.grabbed = 0
        self.objects = 0

        self._latest = None         # последний снятый объект
        self._latest_id = 0
        self._wanted_after = 0      # какой номер объекта считать новым
        self._lock = threading.Lock()
        self._running = False
        self._threads = []

    # --- подключение ---------------------------------------------------

    def _load(self):
        path = os.path.join(SDK_DIR, LIB)
        if not os.path.exists(path):
            raise CameraError(f"нет библиотеки {path}")
        try:
            ksj = CDLL(path)
        except OSError as e:
            raise CameraError(f"не загружается {path}: {e}")

        ksj.KSJ_ExposureTimeSet.argtypes = (c_int, c_float)
        ksj.KSJ_SetFixedFrameRateEx.argtypes = (c_int, c_float)
        ksj.KSJ_CaptureSetFieldOfViewEx.argtypes = (c_int,) * 7 + (c_ushort,)
        ksj.KSJ_CaptureSetTimeOut.argtypes = (c_int, c_uint)
        return ksj

    def open(self):
        """Открывает и настраивает камеру. Бросает CameraError при неудаче."""
        ksj = self._load()
        ksj.KSJ_Init()

        count = ksj.KSJ_DeviceGetCount()
        if count < 1:
            ksj.KSJ_UnInit()
            raise CameraError("камера не найдена (проверьте питание, "
                              "USB3-порт и правило udev)")

        # В восьмой версии библиотеки устройство нужно открывать явно,
        # в шестой это происходило само. Вызов безвреден для обеих.
        if hasattr(ksj, "KSJ_Open"):
            ksj.KSJ_Open(IDX)

        ksj.KSJ_TriggerModeSet(IDX, TRIGGER_MODE)
        if TRIGGER_MODE == 3:
            ksj.KSJ_SetFixedFrameRateEx(IDX, FIXED_RATE)
        ksj.KSJ_ExposureTimeSet(IDX, EXPOSURE_MS)
        ksj.KSJ_CaptureSetFieldOfViewEx(IDX, 0, 0, SENSOR_WIDTH, SENSOR_ROWS,
                                        0, 0, LINES)

        # Порция набирается LINES / FIXED_RATE секунд, таймаут берём с запасом.
        piece_sec = LINES / FIXED_RATE
        ksj.KSJ_CaptureSetTimeOut(IDX, int(piece_sec * 2000 + 5000))

        w, h, bits = c_int(), c_int(), c_int()
        ksj.KSJ_CaptureGetSizeEx(IDX, byref(w), byref(h), byref(bits))
        self.width, self.height = w.value, h.value
        self.channels = bits.value // 8
        if self.width <= 0 or self.height <= 0:
            ksj.KSJ_UnInit()
            raise CameraError(f"камера вернула размер {self.width}x{self.height}")

        self.ksj = ksj
        self.piece_sec = piece_sec
        print(f"  камера: порция {self.width}x{self.height}x{self.channels}, "
              f"{FIXED_RATE/LINES:.1f} порций/с, "
              f"{FIXED_RATE*self.height/LINES:.0f} строк/с")

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

    # --- потоки --------------------------------------------------------

    def _grabber(self):
        """Только забирает порции и кладёт в очередь, без обработки."""
        buf = create_string_buffer(self.width * self.height * self.channels)
        while self._running:
            rc = self.ksj.KSJ_CaptureRgbData(IDX, buf)
            if rc < 0:
                self.errors[rc] = self.errors.get(rc, 0) + 1
                if self.errors[rc] in (1, 10, 100):
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
        det = ObjectDetector(background[X0:X1], background)
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

    # --- фон -----------------------------------------------------------

    def _collect_background(self):
        """
        Медиана строк пустой ленты, на всю ширину сенсора.
        Полоса поиска X0..X1 получается срезом из неё, поэтому
        рассогласования ширины, как раньше, уже не случится.
        """
        want = (self.width, self.channels)
        if os.path.exists(BG_FILE):
            bg = np.load(BG_FILE)
            if bg.shape == want:
                print(f"  фон загружен из {BG_FILE}")
                return bg.astype(np.uint8)
            print(f"  фон в {BG_FILE} имеет форму {bg.shape}, "
                  f"ожидалась {want}; пересобираю")

        print(f"  держите ленту пустой, собираю {BG_ROWS} строк фона "
              f"(примерно {BG_ROWS / (FIXED_RATE * SENSOR_ROWS):.0f} с)...")
        rows = []
        while len(rows) < BG_ROWS:
            try:
                piece = self.frames.get(timeout=self.piece_sec * 3 + 10)
            except queue.Empty:
                raise CameraError("камера не отдаёт данные при сборе фона")
            rows.extend(piece)
        bg = np.median(np.stack(rows[:BG_ROWS]), axis=0).astype(np.uint8)
        np.save(BG_FILE, bg)
        print(f"  фон собран и сохранён в {BG_FILE}")
        return bg

    # --- публичный интерфейс -------------------------------------------

    def start(self):
        """Открывает камеру, собирает фон и запускает фоновую съёмку."""
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
        """Ждём следующий объект: всё, что снято раньше, не считается."""
        with self._lock:
            self._wanted_after = self._latest_id

    def take(self):
        """Снятый объект (строки, 2048, 3) в BGR или None, если ещё нет."""
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


def ensure_started():
    """Поднимает камеру при первом обращении. False, если не удалось."""
    global _camera, _start_error
    if _camera is not None:
        return True
    if _start_error is not None:
        return False
    try:
        cam = LineCamera()
        cam.start()
        _camera = cam
        return True
    except CameraError as e:
        _start_error = str(e)
        print(f"  камера недоступна: {e}")
        return False


def arm():
    if ensure_started():
        _camera.arm()


def take():
    if not ensure_started():
        return None
    return _camera.take()


def status():
    if _camera is None:
        return _start_error or "не запущена"
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
    full_bg = np.full((w, ch), 80, np.uint8)

    def piece(n, with_object=False, bright=200):
        p = rng.integers(75, 86, size=(n, w, ch), dtype=np.int16)
        if with_object:
            p[:, X0 + 100:X0 + 400] = bright
        return p.astype(np.uint8)

    def object_rows(frame):
        """Строки кадра, в которых виден объект."""
        band = frame[:, X0:X1].astype(np.int16)
        diff = np.abs(band - full_bg[X0:X1].astype(np.int16))
        fg = (diff > TH).any(axis=2).sum(axis=1)
        rows = np.flatnonzero(fg >= KEEP_PIX)
        return (int(rows[0]), int(rows[-1])) if rows.size else None

    ok = True
    print(f"кадр должен быть всегда {FRAME_HEIGHT} строк, объект по центру\n")

    for length in (40, 200, 500, 700):
        det = ObjectDetector(full_bg[X0:X1], full_bg)
        found = []
        found += det.feed(piece(300))
        found += det.feed(piece(length, with_object=True))
        found += det.feed(piece(600))              # пауза и строки снизу

        if len(found) != 1:
            print(f"  объект {length} строк: найдено кадров {len(found)}, "
                  f"ожидался 1")
            ok = False
            continue

        frame = found[0]
        span = object_rows(frame)
        if span is None:
            print(f"  объект {length} строк: в кадре его нет")
            ok = False
            continue

        top, bottom = span
        centre = (top + bottom) / 2
        offset = centre - FRAME_HEIGHT / 2
        visible = bottom - top + 1
        fits = length <= FRAME_HEIGHT
        good = (frame.shape == (FRAME_HEIGHT, w, ch)
                and abs(offset) <= 2
                and (visible == length if fits else visible == FRAME_HEIGHT))
        ok = ok and good
        print(f"  объект {length:4d} строк -> кадр {frame.shape[0]}x"
              f"{frame.shape[1]}, объект в строках {top}..{bottom} "
              f"({visible} видно), смещение центра {offset:+.1f}  "
              f"{'ок' if good else 'ПЛОХО'}")

    # фон вокруг объекта должен быть настоящим, а не ровной заливкой
    det = ObjectDetector(full_bg[X0:X1], full_bg)
    det.feed(piece(300))
    det.feed(piece(100, with_object=True))
    frames = det.feed(piece(600))
    top_rows = frames[0][:150, X0:X1]
    uniform = np.all(top_rows == top_rows[0])
    print(f"\n  фон над объектом взят из потока (не ровная заливка): "
          f"{'да' if not uniform else 'НЕТ'}")
    ok = ok and not uniform

    # короткий всплеск не должен считаться объектом
    det = ObjectDetector(full_bg[X0:X1], full_bg)
    short = det.feed(piece(5, with_object=True)) + det.feed(piece(600))
    print(f"  короткий всплеск в 5 строк: кадров {len(short)} (ожидалось 0)")
    ok = ok and len(short) == 0

    # два объекта подряд не должны потеряться
    det = ObjectDetector(full_bg[X0:X1], full_bg)
    two = det.feed(piece(300))
    two += det.feed(piece(60, with_object=True))
    two += det.feed(piece(80))
    two += det.feed(piece(60, with_object=True))
    two += det.feed(piece(600))
    print(f"  два объекта подряд: кадров {len(two)} (ожидалось 2)")
    ok = ok and len(two) == 2

    print("\nдетектор:", "работает" if ok else "НЕ РАБОТАЕТ")
    return ok


def main():
    if "--selftest" in sys.argv:
        raise SystemExit(0 if selftest() else 1)

    if not ensure_started():
        raise SystemExit(1)
    print("жду предметы, Ctrl+C для выхода")
    arm()
    try:
        while True:
            obj = take()
            if obj is not None:
                name = time.strftime("object_%H-%M-%S.png")
                cv2.imwrite(name, obj)
                print(f"  сохранено {name}  ({obj.shape[0]} строк)  "
                      f"[{status()}]")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n" + status())
    finally:
        stop()


if __name__ == "__main__":
    main()