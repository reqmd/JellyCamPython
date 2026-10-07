# -*- coding: utf-8 -*-
"""Jelly4: захват построчно + автоматическая фиксация объектов.

Сохраняет каждый пойманный объект в двух видах:
  obj_NNN.png       - вырезка по объекту с запасом PAD (для оценки фокуса)
  obj_NNN_band.png  - вся полоса строк целиком, с фоном вокруг

Фон (неравномерный, но стабильный) снимается автоматически по первым
BG_ROWS строкам: нужно, чтобы в этот момент под камерой ничего не было.

Остановка: Ctrl+C.
"""

import os
import queue
import sys
import threading
import time
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

IDX = 0
LINES = 256                 # размер транспортной порции
EXPOSURE_MS = 0.1
FIXED_RATE = 8000.0

# Синий фон держится примерно с 560 по 1300 пиксель, дальше спад и края.
X0, X1 = 560, 1300

BG_ROWS = 512               # строк пустого фона для калибровки
TH = 15                     # порог отличия от фона (шум по каналам 1.1/4.5/1.0)
MIN_FG_PIXELS = 12          # пикселей в строке, чтобы начать объект
KEEP_PIX = 4                # пикселей, чтобы продолжать (тонкие места)
GAP_ROWS = 10               # пустых строк подряд = объект кончился
MIN_ROWS = 15
MAX_ROWS = 6000
MIN_AREA = 100
PAD = 30                    # запас вокруг объекта в вырезке

OUT_DIR = "objects"
SAVE_BAND = True            # сохранять ещё и всю полосу с фоном
SHOW = True                # живое окно (на 2500 строк/с лучше выключить)

BW = X1 - X0

# ------------------------------------------------------------ камера
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

GAIN = 200          # 0..255, подбирается по фону

segs = c_int()
if ksj.KSJ_AWAIBA_GetSegmentNum(IDX, byref(segs)) >= 0:
    for sgm in range(segs.value):
        lo, hi = c_int(), c_int()
        ksj.KSJ_AWAIBA_GetGainRange(IDX, sgm, byref(lo), byref(hi))
        ksj.KSJ_AWAIBA_SetGain(IDX, sgm, min(GAIN, hi.value))
        ksj.KSJ_AWAIBA_AutoBlackLevel(IDX, sgm)
    print("усиление %d на %d сегментах" % (GAIN, segs.value))

w, h, bits = c_int(), c_int(), c_int()
ksj.KSJ_CaptureGetSizeEx(IDX, byref(w), byref(h), byref(bits))
W, H, CH = w.value, h.value, bits.value // 8
buf = create_string_buffer(W * H * CH)
os.makedirs(OUT_DIR, exist_ok=True)
print("порция %dx%d, %.1f порций/с, поток %.0f строк/с"
      % (W, H, FIXED_RATE / LINES, FIXED_RATE * H / LINES))

frames = queue.Queue(maxsize=64)
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


threading.Thread(target=grabber, daemon=True).start()

# ------------------------------------------------------------- калибровка
print("держите фон пустым, собираю %d строк..." % BG_ROWS)
cal = []
while sum(p.shape[0] for p in cal) < BG_ROWS:
    cal.append(frames.get(timeout=piece_sec * 4 + 10)[:, X0:X1])
# Медиана по строкам даёт вектор фона на каждый столбец. Неравномерность
# подсветки учитывается автоматически, потому что она постоянна по времени.
background = np.median(np.vstack(cal), axis=0).astype(np.int16)
print("фон снят, средние R,G,B:", np.round(background.mean(axis=0), 1))
print("можно пускать объекты\n")

if SHOW:
    cv2.namedWindow("jelly4", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("jelly4", 900, 500)

# --------------------------------------------------------- состояние сборки
rows_buf = []            # строки текущего объекта
pre = []                 # запас пустых строк перед объектом
active = False
gap = 0
saved = 0
rows_total = 0
t_start = None


def finish():
    """Объект собран: вырезать, измерить резкость, сохранить."""
    global saved, rows_buf
    band = np.stack(rows_buf)
    rows_buf = []
    if band.shape[0] < MIN_ROWS:
        return

    d = band.astype(np.int16) - background
    d -= np.median(d, axis=1, keepdims=True).astype(np.int16)
    mask = (np.abs(d) > TH).any(axis=2).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area < MIN_AREA:
            continue
        y0, y1 = max(0, y - PAD), min(band.shape[0], y + bh + PAD)
        x0, x1 = max(0, x - PAD), min(BW, x + bw + PAD)
        crop = band[y0:y1, x0:x1]

        # Оценка резкости: дисперсия лапласиана. Чем больше, тем чётче.
        # Сравнивая число между проездами, можно крутить фокус вслепую.
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())

        name = os.path.join(OUT_DIR, "obj_%04d.png" % saved)
        cv2.imwrite(name, crop)
        if SAVE_BAND:
            cv2.imwrite(os.path.join(OUT_DIR, "obj_%04d_band.png" % saved),
                        band)
        print("объект %-4d %4dx%-4d px  площадь %6d  резкость %7.1f  %s"
              % (saved, bw, bh, area, sharp, os.path.basename(name)))
        saved += 1


try:
    while True:
        try:
            piece = frames.get(timeout=piece_sec * 4 + 10)
        except queue.Empty:
            print("порции не приходят:", stats)
            break

        belt = piece[:, X0:X1]

        # Маска считается разом на всю порцию - при 2500 строк/с цикл
        # по строкам на Python не успевает в принципе.
        d = belt.astype(np.int16) - background
        d -= np.median(d, axis=1, keepdims=True).astype(np.int16)
        counts = (np.abs(d) > TH).any(axis=2).sum(axis=1)

        for r in range(belt.shape[0]):
            rows_total += 1
            busy = counts[r] >= (KEEP_PIX if active else MIN_FG_PIXELS)
            if busy:
                if not active:
                    t_start = time.perf_counter()
                    rows_buf.extend(pre)
                    pre = []
                    active = True
                rows_buf.append(belt[r])
                gap = 0
                if len(rows_buf) > MAX_ROWS:
                    rows_buf = []
                    active = False
            else:
                if active:
                    gap += 1
                    rows_buf.append(belt[r])
                    if gap >= GAP_ROWS:
                        active = False
                        gap = 0
                        finish()
                else:
                    pre.append(belt[r])
                    if len(pre) > PAD:
                        pre.pop(0)

        if SHOW:
            cv2.imshow("jelly4", belt)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

except KeyboardInterrupt:
    print("\nостановка")

running = False
print("\nстрок обработано: %d, объектов сохранено: %d" % (rows_total, saved))
print("порций принято: %d, ошибки: %s" % (stats["ok"], stats["err"] or "нет"))

got = c_float()
ksj.KSJ_GetFixedFrameRateEx(IDX, byref(got))
print("частота реально установлена: %.1f" % got.value)

if SHOW:
    cv2.destroyAllWindows()
if hasattr(ksj, "KSJ_Close"):
    ksj.KSJ_Close(IDX)
ksj.KSJ_UnInit()