# -*- coding: utf-8 -*-
"""Захват с Jelly4 без потери строк: съёмка в отдельном потоке."""

import os
import threading
import queue
from ctypes import (WinDLL, c_int, c_float, c_ushort, byref,
                    create_string_buffer)

import numpy as np
import cv2

SDK_DIR = r"C:\Program Files (x86)\CatchBEST\IndustryCamera\Applications\KSJShow\x64"
IDX = 0

LINES = 256
RIBBON_H = 1024
EXPOSURE_MS = 0.5
FIXED_RATE = 133.0

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
print("порция %dx%d, ожидается %.1f порций в секунду" % (W, H, FIXED_RATE / LINES))

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

# ------------------------------------------------------------------- просмотр
cv2.namedWindow("jelly4", cv2.WINDOW_NORMAL)
cv2.resizeWindow("jelly4", 900, 500)
ribbon = np.zeros((RIBBON_H, W, CH), np.uint8)

while True:
    try:
        piece = frames.get(timeout=2.0)
    except queue.Empty:
        print("порции не приходят:", stats)
        break

    ribbon = np.vstack([ribbon[H:], piece])
    cv2.imshow("jelly4", ribbon)

    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break
    if key == ord('s'):
        cv2.imwrite("ribbon.png", ribbon)
        print("сохранено ribbon.png")

running = False
t.join(timeout=2.0)
print("принято порций: %d" % stats["ok"])
print("ошибок:", stats["err"] if stats["err"] else "нет")

cv2.destroyAllWindows()
ksj.KSJ_UnInit()