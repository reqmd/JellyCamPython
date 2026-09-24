# -*- coding: utf-8 -*-
"""Jelly4: непрерывный поток -> выделение объектов на ленте -> вырезки.

Клавиши:
  c  калибровка по пустой ленте (обязательно перед работой)
  d  показывать маску детекции вместо картинки
  q  выход
Вырезки сохраняются в папку crops/.
"""

import os
import threading
import queue
import time
from ctypes import (WinDLL, c_int, c_float, c_ushort, byref,
                    create_string_buffer)

import numpy as np
import cv2

# ------------------------------------------------------------------ параметры
SDK_DIR = r"C:\Program Files (x86)\CatchBEST\IndustryCamera\Applications\KSJShow\x64"
IDX = 0
LINES = 256
EXPOSURE_MS = 0.6
FIXED_RATE = 134.0

BELT_X0, BELT_X1 = 260, 1200   # границы ленты в пикселях, подберите по кадру
WORK_H = 2048                  # высота рабочего буфера в строках
THRESH = 28                    # порог отличия от фона ленты
MIN_AREA = 800                 # минимальная площадь объекта, пикселей
PAD = 20                       # запас вокруг объекта при вырезке
MARGIN = 40                    # объект считается прошедшим, отступив на столько

OUT_DIR = "crops"          # вырезка как есть (прямоугольник с фоном)
MASK_DIR = "crops_mask"    # объект на чёрном фоне
CALIB_FILE = "flatfield.npz"
CSV_FILE = "objects.csv"
MM_PER_PX = 0.0879         # калибровка: сколько мм в одном пикселе

# ------------------------------------------------------------ настройка камеры
os.add_dll_directory(SDK_DIR)
ksj = WinDLL(os.path.join(SDK_DIR, "KSJApi64.dll"))
ksj.KSJ_ExposureTimeSet.argtypes = (c_int, c_float)
ksj.KSJ_SetFixedFrameRateEx.argtypes = (c_int, c_float)
ksj.KSJ_CaptureSetFieldOfViewEx.argtypes = (c_int,) * 7 + (c_ushort,)

ksj.KSJ_Init()
ksj.KSJ_TriggerModeSet(IDX, 3)
ksj.KSJ_SetFixedFrameRateEx(IDX, FIXED_RATE)
ksj.KSJ_ExposureTimeSet(IDX, EXPOSURE_MS)
ksj.KSJ_CaptureSetFieldOfViewEx(IDX, 0, 0, 2048, 2, 0, 0, LINES)

w, h, bits = c_int(), c_int(), c_int()
ksj.KSJ_CaptureGetSizeEx(IDX, byref(w), byref(h), byref(bits))
W, H, CH = w.value, h.value, bits.value // 8
buf = create_string_buffer(W * H * CH)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

csv = open(CSV_FILE, "a", encoding="utf-8")
if csv.tell() == 0:
    csv.write("id,bbox_w,bbox_h,area_px,rect_w,rect_h,angle,"
              "w_mm,h_mm,area_mm2,fill\n")

# ----------------------------------------------------- поток, который снимает
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
        piece = np.frombuffer(buf, np.uint8).reshape(H, W, CH).copy()
        try:
            frames.put_nowait(piece)
        except queue.Full:
            stats["err"]["очередь"] = stats["err"].get("очередь", 0) + 1


threading.Thread(target=grabber, daemon=True).start()

# ------------------------------------------------------- калибровка (плоское поле)
# Идея: снять пустую ленту и запомнить, какую яркость даёт каждый столбец.
# Дальше делим кадр на этот профиль - уходят и вертикальные полосы,
# и неравномерность подсветки. Фон становится ровным, порог работает везде.
ref = None
if os.path.exists(CALIB_FILE):
    ref = np.load(CALIB_FILE)["ref"]
    print("калибровка загружена из", CALIB_FILE)


def flatten(gray):
    """Выровнять яркость по столбцам. Возвращает фон, приведённый к 128."""
    if ref is None:
        return gray.astype(np.float32)
    return gray.astype(np.float32) * (128.0 / np.maximum(ref, 1.0))


# ------------------------------------------------------------------ буферы
ribbon = np.zeros((WORK_H, W, CH), np.uint8)
rows_total = 0          # сколько строк ленты прошло всего (абсолютный счётчик)
emitted = []            # уже сохранённые объекты: (абс. строка центра, столбец)
saved = 0

cv2.namedWindow("belt", cv2.WINDOW_NORMAL)
cv2.resizeWindow("belt", 900, 600)
show_mask = False

print("наведите на пустую ленту и нажмите 'c' для калибровки")

while True:
    try:
        piece = frames.get(timeout=3.0)
    except queue.Empty:
        print("порции не приходят:", stats)
        break

    ribbon = np.vstack([ribbon[H:], piece])
    rows_total += H

    # ---------------------------------------------------------- детекция
    gray = cv2.cvtColor(ribbon, cv2.COLOR_BGR2GRAY)
    flat = flatten(gray)

    # Работаем только по ленте, кромки и фон снаружи не трогаем.
    roi = flat[:, BELT_X0:BELT_X1]

    # Объект - всё, что заметно отличается от ровного фона в обе стороны.
    diff = np.abs(roi - 128.0)
    mask = (diff > THRESH).astype(np.uint8) * 255

    # Убрать пятнышки от текстуры ленты и залить дырки внутри объектов.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)

    view = ribbon.copy()
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area < MIN_AREA:
            continue
        x += BELT_X0                       # обратно в координаты всей ленты

        # Объект готов, только когда полностью вошёл в буфер и отошёл от
        # нижнего края: иначе рискуем вырезать его наполовину.
        if y + bh > WORK_H - MARGIN:
            cv2.rectangle(view, (x, y), (x + bw, y + bh), (0, 165, 255), 3)
            continue

        # Абсолютная позиция центра: не зависит от прокрутки буфера.
        abs_row = rows_total - WORK_H + y + bh // 2
        col = x + bw // 2

        # Тот же объект приходит в нескольких порциях подряд - сохраняем раз.
        if any(abs(abs_row - r) < 30 and abs(col - c) < 40 for r, c in emitted):
            cv2.rectangle(view, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            continue
        emitted.append((abs_row, col))
        emitted[:] = emitted[-200:]

        y0, y1 = max(0, y - PAD), min(WORK_H, y + bh + PAD)
        x0 = max(BELT_X0, x - PAD)
        x1 = min(BELT_X1, x + bw + PAD)

        crop = ribbon[y0:y1, x0:x1].copy()

        # Маска именно этого объекта: lab хранит номера компонент,
        # берём только пиксели с нашим номером i.
        m = (lab[y0:y1, x0 - BELT_X0:x1 - BELT_X0] == i).astype(np.uint8)

        # Объект на чёрном фоне - то, что пойдёт в сеть.
        masked = crop.copy()
        masked[m == 0] = 0

        # Минимальный охватывающий прямоугольник: даёт реальные размеры
        # объекта независимо от того, как он повёрнут на ленте.
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        (_, _), (rw, rh), ang = cv2.minAreaRect(c)
        rw, rh = max(rw, rh), min(rw, rh)      # длинная сторона первой

        name = "obj_%05d.png" % saved
        cv2.imwrite(os.path.join(OUT_DIR, name), crop)
        cv2.imwrite(os.path.join(MASK_DIR, name), masked)
        csv.write("%s,%d,%d,%d,%.1f,%.1f,%.1f,%.2f,%.2f,%.2f,%.3f\n"
                  % (name, bw, bh, area, rw, rh, ang,
                     rw * MM_PER_PX, rh * MM_PER_PX,
                     area * MM_PER_PX ** 2,
                     area / max(rw * rh, 1)))
        csv.flush()
        saved += 1
        print("объект %d: %.1f x %.1f мм (%.0f x %.0f px), площадь %.1f мм2"
              % (saved, rw * MM_PER_PX, rh * MM_PER_PX, rw, rh,
                 area * MM_PER_PX ** 2))
        cv2.rectangle(view, (x, y), (x + bw, y + bh), (0, 0, 255), 3)

    # ------------------------------------------------------------ показ
    if show_mask:
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        vis = view
    cv2.imshow("belt", vis)

    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break
    elif key == ord('d'):
        show_mask = not show_mask
    elif key == ord('c'):
        # Среднее по строкам даёт профиль чувствительности столбцов.
        ref = cv2.cvtColor(ribbon, cv2.COLOR_BGR2GRAY).mean(axis=0)
        ref = cv2.GaussianBlur(ref.reshape(1, -1).astype(np.float32),
                               (1, 1), 0).ravel()
        np.savez(CALIB_FILE, ref=ref)
        print("калибровка снята и сохранена, фон приведён к 128")

running = False
csv.close()
print("сохранено объектов: %d, ошибок: %s" % (saved, stats["err"] or "нет"))
print("размеры записаны в", CSV_FILE)
cv2.destroyAllWindows()
ksj.KSJ_UnInit()