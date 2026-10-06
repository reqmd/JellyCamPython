# -*- coding: utf-8 -*-
"""Почему кадры чёрные: нет света, нулевое усиление или не работает цвет.

Запуск: sudo LD_LIBRARY_PATH=/opt/ksjapi/arm64 "$(which python)" diag_black.py
"""

import os
import sys
from ctypes import (CDLL, c_int, c_uint, c_bool, c_float, c_ushort, byref,
                    create_string_buffer)

import numpy as np

SDK_DIR = "/opt/ksjapi/arm64"
IDX = 0
LINES = 64            # мелкая порция: быстрее отклик при отладке
FIXED_RATE = 134.0

# ------------------------------------------------- что лежит рядом с .so
need_libs = ["libksjapi.so", "libksjbayer.so", "libksjcam_98.so"]
print("файлы SDK:")
for n in need_libs:
    p = os.path.join(SDK_DIR, n)
    print("  %-22s %s" % (n, "есть" if os.path.exists(p) else "НЕТ"))
print("  (без libksjbayer.so цветное изображение собирается в нули)\n")

ksj = CDLL(os.path.join(SDK_DIR, "libksjapi.so"))
ksj.KSJ_ExposureTimeSet.argtypes = (c_int, c_float)
ksj.KSJ_SetFixedFrameRateEx.argtypes = (c_int, c_float)
ksj.KSJ_CaptureSetFieldOfViewEx.argtypes = (c_int,) * 7 + (c_ushort,)
ksj.KSJ_CaptureSetTimeOut.argtypes = (c_int, c_uint)

ksj.KSJ_Init()
ksj.KSJ_DeviceGetCount()
rc = ksj.KSJ_Open(IDX)
print("KSJ_Open ->", rc)
if rc < 0:
    sys.exit("камера не открылась")

mono = c_int()
ksj.KSJ_QueryFunction(IDX, 0, byref(mono))
print("камера монохромная:", bool(mono.value))

# ------------------------------------------------------ усиление сегментов
# На свежеоткрытой камере усиление может быть нулевым - тогда кадр чёрный
# при любом освещении. На Windows там стояло 225 из 255.
segs = c_int()
if ksj.KSJ_AWAIBA_GetSegmentNum(IDX, byref(segs)) >= 0:
    print("сегментов сенсора:", segs.value)
    for s in range(segs.value):
        g, lo, hi = c_int(), c_int(), c_int()
        ksj.KSJ_AWAIBA_GetGainRange(IDX, s, byref(lo), byref(hi))
        ksj.KSJ_AWAIBA_GetGain(IDX, s, byref(g))
        print("  сегмент %d: усиление %d (диапазон %d..%d)"
              % (s, g.value, lo.value, hi.value))
        if g.value < hi.value // 2:
            ksj.KSJ_AWAIBA_SetGain(IDX, s, hi.value)
            print("    -> поднял до максимума %d" % hi.value)
        ksj.KSJ_AWAIBA_AutoBlackLevel(IDX, s)

# -------------------------------------------------------------- настройка
ksj.KSJ_TriggerModeSet(IDX, 0)          # free run: светом управляем вручную
ksj.KSJ_ExposureTimeSet(IDX, 0.8)       # максимум из известного диапазона
ksj.KSJ_CaptureSetFieldOfViewEx(IDX, 0, 0, 2048, 2, 0, 0, LINES)
ksj.KSJ_CaptureSetTimeOut(IDX, 10000)

w, h, bits = c_int(), c_int(), c_int()
ksj.KSJ_CaptureGetSizeEx(IDX, byref(w), byref(h), byref(bits))
W, H, CH = w.value, h.value, bits.value // 8
print("\nкадр %dx%d, %d бит на пиксель" % (W, H, bits.value))

exp = c_float()
if hasattr(ksj, "KSJ_ExposureTimeGet"):
    ksj.KSJ_ExposureTimeGet(IDX, byref(exp))
    print("выдержка реально установлена: %.4f мс" % exp.value)


def stats(a, label):
    print("  %-10s мин %3d  среднее %6.2f  макс %3d  ненулевых %.1f%%"
          % (label, a.min(), a.mean(), a.max(), 100.0 * (a > 0).mean()))


# ------------------------------- сравниваем сырые данные и цветные
print("\nпо 3 кадра каждого вида:")
raw_buf = create_string_buffer(W * H)          # 8 бит на пиксель
rgb_buf = create_string_buffer(W * H * CH)

print(" KSJ_CaptureRawData (сырые данные сенсора, без обработки цвета):")
for _ in range(3):
    rc = ksj.KSJ_CaptureRawData(IDX, raw_buf)
    if rc < 0:
        print("  ошибка", rc)
        continue
    stats(np.frombuffer(raw_buf, np.uint8), "raw")

print(" KSJ_CaptureRgbData (после восстановления цвета):")
for _ in range(3):
    rc = ksj.KSJ_CaptureRgbData(IDX, rgb_buf)
    if rc < 0:
        print("  ошибка", rc)
        continue
    stats(np.frombuffer(rgb_buf, np.uint8), "rgb")

print("""
Как читать:
  raw НЕ нулевой, rgb нулевой -> не работает восстановление цвета
                                 (нет libksjbayer.so или не нашлась)
  оба нулевые, усиление на максимуме -> до сенсора не доходит свет:
                                 крышка на объективе, нет подсветки
  оба ненулевые -> всё в порядке, дело было в усилении или выдержке""")

ksj.KSJ_Close(IDX)
ksj.KSJ_UnInit()
