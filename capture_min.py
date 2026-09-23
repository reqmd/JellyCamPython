# -*- coding: utf-8 -*-
"""Минимальный захват с CatchBEST Jelly4 (линейная камера 2048 px, цвет)."""

import os
from ctypes import WinDLL, c_int, c_float, c_ushort, byref, create_string_buffer

import numpy as np
import cv2

SDK_DIR = r"C:\Program Files\MVTec\bin\x64-win64"
IDX = 0        # индекс камеры
LINES = 1024   # сколько считываний сенсора склеить в один кадр
EXPOSURE_MS = 0.8    # выдержка одной строки, мс (диапазон 0.0021..0.8)
TRIGGER_MODE = 0    

#FIXED_RATE = 100.0

os.add_dll_directory(SDK_DIR)
ksj = WinDLL(os.path.join(SDK_DIR, "KSJApi64.dll"))
ksj.KSJ_ExposureTimeSet.argtypes = (c_int, c_float)
ksj.KSJ_CaptureSetFieldOfViewEx.argtypes = (c_int,) * 7 + (c_ushort,)

ksj.KSJ_Init()

# 1. Режим съёмки: 0 = внутренний, камера снимает непрерывно сама.
ksj.KSJ_TriggerModeSet(IDX, TRIGGER_MODE)

# 2. Выдержка одной строки, мс.
ksj.KSJ_ExposureTimeSet(IDX, EXPOSURE_MS)

# 3. Склейка строк в площадное изображение: весь сенсор (2048x2) x LINES раз.
ksj.KSJ_CaptureSetFieldOfViewEx(IDX, 0, 0, 2048, 2, 0, 0, LINES)

# 4. Спрашиваем итоговый размер кадра и выделяем под него буфер.
w, h, bits = c_int(), c_int(), c_int()
ksj.KSJ_CaptureGetSizeEx(IDX, byref(w), byref(h), byref(bits))
W, H, CH = w.value, h.value, bits.value // 8
buf = create_string_buffer(W * H * CH)
print("кадр %dx%d, %d канал(ов)" % (W, H, CH))

# 5. Цикл: забрать кадр, показать. Esc или q - выход.
while True:
    if ksj.KSJ_CaptureRgbData(IDX, buf) < 0:
        continue
    img = np.frombuffer(buf, np.uint8).reshape(H, W, CH)
    cv2.imshow("jelly4", img)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cv2.imwrite("frame.png", img)
cv2.destroyAllWindows()
ksj.KSJ_UnInit()