# -*- coding: utf-8 -*-
"""Захват с Jelly4 построчно: камера отдаёт порции, код получает строки.

LINES задаёт только размер транспортной порции, на логику он не влияет.
Поставьте LINES = 1, чтобы камера отдавала по одному считыванию (2 строки) -
это настоящий построчный режим, но 134 вызова в секунду нагружают USB
сильнее, чем один. Значение 256 эффективнее и даёт тот же поток строк.
"""

import os
import threading
import queue
from ctypes import (WinDLL, c_int, c_float, c_ushort, byref,
                    create_string_buffer)

import numpy as np
import cv2

SDK_DIR = r"C:\Program Files (x86)\CatchBEST\IndustryCamera\Applications\KSJShow\x64"
IDX = 0

LINES = 256            # размер транспортной порции, не логики
EXPOSURE_MS = 0.5
FIXED_RATE = 134.0
RIBBON_H = 1024        # только для показа

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
print("порция %dx%d, %.1f порций/с, поток %.0f строк/с"
      % (W, H, FIXED_RATE / LINES, FIXED_RATE * H / LINES))

# ---------------------------------------------- поток, который только снимает
frames = queue.Queue(maxsize=32)
stats = {"ok": 0, "err": {}}
running = True


def grabber():
    """Никакой обработки: забрал порцию - сразу просит следующую."""
    while running:
        rc = ksj.KSJ_CaptureRgbData(IDX, buf)
        if rc < 0:
            stats["err"][rc] = stats["err"].get(rc, 0) + 1
            continue
        stats["ok"] += 1
        # copy() обязателен: buf будет перезаписан следующей порцией
        piece = np.frombuffer(buf, np.uint8).reshape(H, W, CH).copy()
        try:
            frames.put_nowait(piece)
        except queue.Full:
            stats["err"]["очередь переполнена"] = \
                stats["err"].get("очередь переполнена", 0) + 1


t = threading.Thread(target=grabber, daemon=True)
t.start()


# ------------------------------------------------------- построчный интерфейс
def rows(timeout=2.0):
    """Генератор строк: (номер строки с начала съёмки, строка (W, 3)).

    Порции разбираются внутри, наружу идёт непрерывный поток строк.
    Строки - срезы порции, не копии; если нужно сохранить строку
    надолго, берите row.copy().
    """
    n = 0
    while running:
        try:
            piece = frames.get(timeout=timeout)
        except queue.Empty:
            print("порции не приходят:", stats)
            return
        for r in range(piece.shape[0]):
            yield n, piece[r]
            n += 1


# ------------------------------------------------------------------- пример
# Здесь строки просто копятся в ленту для показа. Замените тело цикла
# на свою построчную логику - детекцию, пороги, накопление объекта.
ribbon = np.zeros((RIBBON_H, W, CH), np.uint8)
fill = 0

cv2.namedWindow("jelly4", cv2.WINDOW_NORMAL)
cv2.resizeWindow("jelly4", 900, 500)

for n, row in rows():
    # --- ваша обработка одной строки ---
    ribbon[fill] = row
    fill += 1

    # Окно обновляем не на каждой строке, иначе весь процессор уйдёт
    # на отрисовку, а камера начнёт терять данные.
    if fill < RIBBON_H:
        continue

    cv2.imshow("jelly4", ribbon)
    ribbon = np.roll(ribbon, -RIBBON_H // 2, axis=0)
    fill = RIBBON_H // 2

    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break
    if key == ord('s'):
        cv2.imwrite("ribbon.png", ribbon)
        print("сохранено ribbon.png, строк всего: %d" % n)

running = False
t.join(timeout=2.0)
print("принято порций: %d" % stats["ok"])
print("ошибок:", stats["err"] if stats["err"] else "нет")

cv2.destroyAllWindows()
ksj.KSJ_UnInit()