# -*- coding: utf-8 -*-
"""Jelly4: захват + детекция скользящим окном.

Идея: не собирать объект построчно, а смотреть на последние WIN_PIECES
порций как на обычную картинку. Сохраняем только те объекты, что целиком
помещаются в окно и не касаются его краёв. Обрезанный краем объект просто
пропускаем - окно сдвинется, и он попадёт в следующее уже целиком.

Остановка: Ctrl+C.
"""

import csv
import datetime
import os
import queue
import sys
import threading
import time
from collections import deque
from ctypes import (CDLL, c_int, c_uint, c_float, c_ushort, byref,
                    create_string_buffer)

import cv2
import numpy as np

if sys.platform == "win32":
    from ctypes import WinDLL
    SDK_DIR = r"C:\Program Files (x86)\CatchBEST\IndustryCamera\Applications\KSJShow\x64"
    LIB = "KSJApi64.dll"
else:
    SDK_DIR = "/opt/ksjapi/arm64"
    LIB = "libksjapi.so"

# ----------------------------------------------------------------- настройки
IDX = 0
LINES = 16
EXPOSURE_MS = 0.1
FIXED_RATE = 8000.0
GAIN = 200
X0, X1 = 560, 1300

BG_ROWS = 512          # строк пустого фона для калибровки
# Окно задаём В СТРОКАХ: оно должно быть заметно больше самого длинного
# объекта. Число порций в нём вычисляется само, поэтому LINES можно менять
# ради задержки, не трогая логику.
WIN_ROWS_MIN = 512
TH = 15                # порог отличия от фона
MIN_AREA = 400         # площадь компоненты, px
# Отступы от краёв окна. Снизу приходят СВЕЖИЕ строки, и объект там ещё
# доезжает: его слабый задний край порог не проходит, компонента кажется
# законченной, и объект сохраняется обрезанным. Поэтому снизу отступ
# большой - он гарантирует, что объект доехал целиком вместе с краем.
# Сверху строки САМЫЕ СТАРЫЕ, объект оттуда уже уезжает и давно сохранён,
# так что хватает нескольких пикселей.
EDGE_NEW = 80          # отступ от нижнего (свежего) края окна
EDGE_OLD = 3           # отступ от верхнего (старого) края
PAD = 20               # запас фона вокруг вырезки
OPEN_K, CLOSE_K = 3, 11

OUT_DIR = "objects"
LOG_FILE = "objects.csv"
PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]

BW = X1 - X0
ROWS_PER_SEC = FIXED_RATE * 2

# ------------------------------------------------------------------- камера
if sys.platform == "win32":
    os.add_dll_directory(SDK_DIR)
    ksj = WinDLL(os.path.join(SDK_DIR, LIB))
else:
    ksj = CDLL(os.path.join(SDK_DIR, LIB))

ksj.KSJ_ExposureTimeSet.argtypes = (c_int, c_float)
ksj.KSJ_SetFixedFrameRateEx.argtypes = (c_int, c_float)
ksj.KSJ_CaptureSetFieldOfViewEx.argtypes = (c_int,) * 7 + (c_ushort,)
ksj.KSJ_CaptureSetTimeOut.argtypes = (c_int, c_uint)

ksj.KSJ_Init()
if ksj.KSJ_DeviceGetCount() < 1:
    raise SystemExit("камера не найдена")
if hasattr(ksj, "KSJ_Open"):
    ksj.KSJ_Open(IDX)

ksj.KSJ_TriggerModeSet(IDX, 3)
ksj.KSJ_SetFixedFrameRateEx(IDX, FIXED_RATE)
ksj.KSJ_ExposureTimeSet(IDX, EXPOSURE_MS)
ksj.KSJ_CaptureSetFieldOfViewEx(IDX, 0, 0, 2048, 2, 0, 0, LINES)
piece_sec = LINES / FIXED_RATE
ksj.KSJ_CaptureSetTimeOut(IDX, int(piece_sec * 2000 + 5000))

segs = c_int()
if ksj.KSJ_AWAIBA_GetSegmentNum(IDX, byref(segs)) >= 0:
    for sgm in range(segs.value):
        ksj.KSJ_AWAIBA_SetGain(IDX, sgm, GAIN)
        ksj.KSJ_AWAIBA_AutoBlackLevel(IDX, sgm)

w, h, bits = c_int(), c_int(), c_int()
ksj.KSJ_CaptureGetSizeEx(IDX, byref(w), byref(h), byref(bits))
W, H, CH = w.value, h.value, bits.value // 8
buf = create_string_buffer(W * H * CH)
os.makedirs(OUT_DIR, exist_ok=True)

WIN_PIECES = max(2, -(-WIN_ROWS_MIN // H))      # округление вверх
WIN_ROWS = H * WIN_PIECES
print("порция %dx%d (%.1f мс), окно %d порций = %d строк (%.1f мс)"
      % (W, H, piece_sec * 1000, WIN_PIECES, WIN_ROWS,
         WIN_ROWS / ROWS_PER_SEC * 1000))
print("отступ от свежего края %d строк (+%.1f мс к задержке)"
      % (EDGE_NEW, EDGE_NEW / ROWS_PER_SEC * 1000))

# ------------------------------------------------------------------- потоки
frames = queue.Queue(maxsize=64)
write_q = queue.Queue(maxsize=256)
stats = {"ok": 0, "err": {}}
running = True


def grabber():
    while running:
        rc = ksj.KSJ_CaptureRgbData(IDX, buf)
        if rc < 0:
            stats["err"][rc] = stats["err"].get(rc, 0) + 1
            continue
        stats["ok"] += 1
        try:
            frames.put_nowait(
                np.frombuffer(buf, np.uint8).reshape(H, W, CH).copy())
        except queue.Full:
            stats["err"]["очередь"] = stats["err"].get("очередь", 0) + 1


def writer():
    while True:
        item = write_q.get()
        if item is None:
            break
        cv2.imwrite(item[0], item[1], PNG_PARAMS)


threading.Thread(target=grabber, daemon=True).start()
threading.Thread(target=writer, daemon=True).start()

# --------------------------------------------------------------- калибровка
print("держите фон пустым, собираю %d строк..." % BG_ROWS)
cal = []
while sum(p.shape[0] for p in cal) < BG_ROWS:
    cal.append(frames.get(timeout=piece_sec * 6 + 10)[:, X0:X1])
background = np.median(np.vstack(cal), axis=0).astype(np.int16)
print("фон снят:", np.round(background.mean(axis=0), 1), "\nможно пускать\n")

logf = open(LOG_FILE, "a", newline="", encoding="utf-8")
logw = csv.writer(logf)
if logf.tell() == 0:
    logw.writerow(["file", "время", "строка", "ширина", "высота", "высота, мс",
                   "задержка, мс", "площадь", "резкость"])

ker_open = np.ones((OPEN_K, OPEN_K), np.uint8)
ker_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))

window = deque(maxlen=WIN_PIECES)     # последние порции (цвет)
masks = deque(maxlen=WIN_PIECES)      # их готовые маски
has_fg = deque(maxlen=WIN_PIECES)     # есть ли в порции хоть что-то
idle = 0                              # сколько окон подряд пропущено
base_row = 0                          # номер первой строки окна
saved = 0
seen = deque(maxlen=200)              # абсолютные позиции уже сохранённых

try:
    while True:
        try:
            piece = frames.get(timeout=piece_sec * 6 + 10)
        except queue.Empty:
            print("порции не приходят:", stats)
            break
        t_arrived = time.perf_counter()

        if len(window) == WIN_PIECES:
            base_row += H                 # самая старая порция выпала
        belt = piece[:, X0:X1]
        window.append(belt)

        # Тяжёлую арифметику делаем ОДИН раз на порцию, а не на каждое окно.
        # Раньше одни и те же пиксели проходили вычитание фона WIN_PIECES раз.
        d = belt.astype(np.int16) - background
        d -= np.median(d[:, ::8], axis=1, keepdims=True).astype(np.int16)
        m = (np.abs(d) > TH).any(axis=2).astype(np.uint8) * 255
        masks.append(m)
        has_fg.append(bool(cv2.countNonZero(m)))

        if len(window) < WIN_PIECES:
            continue

        # Если во всём окне нет ни одного отклонившегося пикселя, объектов
        # там быть не может. Морфология и поиск компонент пропускаются -
        # на пустой ленте это почти всё время работы.
        if not any(has_fg):
            idle += 1
            continue

        rows = WIN_ROWS
        img = None                        # цветное окно собираем лениво

        # Морфология и поиск компонент - по окну: разрывы на стыках порций
        # должны сшиваться, иначе объект снова распадётся.
        mask = np.vstack(masks)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker_close)

        n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            x, y, bw, bh, area = st[i]
            if area < MIN_AREA:
                continue
            # Главное правило: объект должен целиком помещаться в окно
            # и быть достаточно далеко от свежего края, иначе он ещё едет.
            if y + bh > rows - EDGE_NEW:
                continue
            if y < EDGE_OLD:
                continue
            if x < EDGE_OLD or x + bw > BW - EDGE_OLD:
                continue

            abs_row = base_row + y + bh // 2
            col = x + bw // 2
            # Один объект виден в нескольких окнах подряд - сохраняем раз.
            if any(abs(abs_row - r) < bh and abs(col - c) < 20
                   for r, c in seen):
                continue
            seen.append((abs_row, col))

            y0, y1 = max(0, y - PAD), min(rows, y + bh + PAD)
            x0, x1 = max(0, x - PAD), min(BW, x + bw + PAD)
            if img is None:               # склеиваем только когда есть что резать
                img = np.vstack(window)
            crop = img[y0:y1, x0:x1].copy()

            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
            height_ms = bh / ROWS_PER_SEC * 1000
            lag_ms = (time.perf_counter() - t_arrived) * 1000
            stamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

            name = os.path.join(OUT_DIR, "obj_%04d.png" % saved)
            write_q.put((name, crop))
            print("%s  объект %-4d %4dx%-4d px  высота %5.1f мс  "
                  "задержка %5.1f мс  площадь %6d  резкость %7.1f  %s"
                  % (stamp, saved, bw, bh, height_ms, lag_ms, area, sharp,
                     os.path.basename(name)))
            logw.writerow([os.path.basename(name), stamp, abs_row, bw, bh,
                           "%.1f" % height_ms, "%.1f" % lag_ms, area,
                           "%.1f" % sharp])
            logf.flush()
            saved += 1

except KeyboardInterrupt:
    print("\nостановка")

running = False
write_q.put(None)
time.sleep(0.3)
logf.close()
print("сохранено: %d, порций: %d, из них пропущено пустых: %d, ошибки: %s"
      % (saved, stats["ok"], idle, stats["err"] or "нет"))
if hasattr(ksj, "KSJ_Close"):
    ksj.KSJ_Close(IDX)
ksj.KSJ_UnInit()