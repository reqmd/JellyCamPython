# -*- coding: utf-8 -*-
"""
CatchBEST Jelly4 MU3L2K7C(AGYYO) - линейная (line scan) камера, 2048 px, цветная.
Захват изображения через KSJApi под Windows.

Требования: pip install numpy opencv-python
Разрядность Python должна совпадать с разрядностью KSJApi.dll (обычно x64).
"""

import os
import sys
import time
from ctypes import (CDLL, WinDLL, c_int, c_float, c_ushort, c_bool,
                    byref, create_string_buffer)

import numpy as np
import cv2

# ---------------------------------------------------------------- настройки
# Папка, где лежит KSJApi.dll (обычно C:\Program Files\CatchBEST\...\x64
# или папка Bin установленного SDK). Поставьте свой путь.
SDK_DIR = r"C:\Program Files\MVTec\bin\x64-win64"
IDX = 0              # индекс камеры
LINES = 1024         # сколько строк склеивать в один "кадр"
EXPOSURE_MS = 0.8    # выдержка одной строки, мс (диапазон 0.0021..0.8)
TRIGGER_MODE = 0     # 0=внутренний (free run), 1=внешний, 2=софтовый,
                     # 3=фиксированная частота, 5=энкодер
FIXED_RATE = 100.0   # используется только при TRIGGER_MODE = 3

# ------------------------------------------------------------------- загрузка
# Заголовки SDK противоречат сами себе: KSJApiRetCode.h объявляет
# RET_SUCCESS = 0, но в комментариях 25 функций написано RET_SUCCESS(1).
# Windows-сборка KSJApi64.dll возвращает 1, Linux-сборка - 0.
# Надёжный признак ошибки - отрицательное значение.
OK = (0, 1)
ERRORS = {
    -1: "неверный параметр", -2: "не выделилась память", -3: "функция не поддерживается",
    -4: "устройство не найдено", -5: "устройство не инициализировано",
    -6: "конфликт операций", -7: "нет прав", -8: "ошибка", -9: "ошибка",
    -12: "битый кадр (BADFRAME) - кадр нужно пропустить",
    -13: "невалидный кадр (INVALIDFRAME) - повторить захват",
    -14: "нулевой кадр (ZEROFRAME)", -16: "таймаут чтения",
    -17: "устройство закрыто", -21: "стриминг не запущен",
}


def check(name, code, fatal=True):
    if code < 0:
        msg = "%s -> %d (%s)" % (name, code, ERRORS.get(code, "см. KSJApiRetCode.h"))
        if fatal:
            raise RuntimeError(msg)
        print("  warn:", msg)
    return code


def load_api():
    if sys.platform == "win32":
        if hasattr(os, "add_dll_directory") and os.path.isdir(SDK_DIR):
            os.add_dll_directory(SDK_DIR)
        path = os.path.join(SDK_DIR, "KSJApi64.dll")
        return WinDLL(path if os.path.exists(path) else "KSJApi64.dll")
    return CDLL("libksjapi.so")


ksj = load_api()

# Явно объявляем типы там, где есть float/short - иначе ctypes передаст мусор.
ksj.KSJ_ExposureTimeSet.argtypes = (c_int, c_float)
ksj.KSJ_SetFixedFrameRateEx.argtypes = (c_int, c_float)
ksj.KSJ_CaptureMultiLineSet.argtypes = (c_int, c_ushort)

# ------------------------------------------------------------------ init
check("KSJ_Init", ksj.KSJ_Init())

v = [c_int() for _ in range(4)]
ksj.KSJ_GetVersion(*[byref(x) for x in v])
print("версия KSJApi: %d.%d.%d.%d" % tuple(x.value for x in v))

count = ksj.KSJ_DeviceGetCount()
print("камер найдено:", count)
if count < 1:
    sys.exit("камера не найдена: проверьте драйвер и кабель USB3")

dev_type, serial, fw, fpga = c_ushort(), c_int(), c_ushort(), c_ushort()
ksj.KSJ_DeviceGetInformationEx(IDX, byref(dev_type), byref(serial),
                               byref(fw), byref(fpga))
print("тип=%d serial=%d fw=%d fpga=%d" % (dev_type.value, serial.value,
                                          fw.value, fpga.value))

# Цветная камера или ч/б. KSJ_PROPERTY_MONO_DEVICE = 0 в enum KSJ_FUNCTION.
# (в демках производителя эта функция вызвана с неверным числом аргументов)
is_mono = c_int()
ksj.KSJ_QueryFunction(IDX, 0, byref(is_mono))
print("монохромная:", bool(is_mono.value))

# Линейный ли сенсор (Awaiba) - у Jelly4 должно быть True
is_awaiba = c_bool()
if ksj.KSJ_AWAIBA_IsUsed(IDX, byref(is_awaiba)) in OK:
    print("сенсор Awaiba (линейка):", is_awaiba.value)

# ------------------------------------------------- режим работы и параметры
check("KSJ_TriggerModeSet", ksj.KSJ_TriggerModeSet(IDX, TRIGGER_MODE))
if TRIGGER_MODE == 3:
    check("KSJ_SetFixedFrameRateEx",
          ksj.KSJ_SetFixedFrameRateEx(IDX, FIXED_RATE), fatal=False)

lo, hi = c_float(), c_float()
ksj.KSJ_ExposureTimeRangeGet(IDX, byref(lo), byref(hi))
print("выдержка допустима: %.4f .. %.1f мс" % (lo.value, hi.value))
check("KSJ_ExposureTimeSet", ksj.KSJ_ExposureTimeSet(IDX, EXPOSURE_MS))

# ГЛАВНОЕ для линейной камеры: сколько строк склеивается в один кадр.
# KSJ_CaptureMultiLineSet на этой прошивке не поддерживается (-3).
# Правильный путь - "многокадровая сшивка": задаём FOV высотой в 1 строку
# и просим API склеить wMultiFrameNum таких строк в одно площадное изображение.
# В документации прямо сказано, что этот механизм и предназначен для линеек.
ksj.KSJ_CaptureSetFieldOfViewEx.argtypes = (c_int, c_int, c_int, c_int, c_int,
                                            c_int, c_int, c_ushort)

sup = c_int()
ksj.KSJ_QueryFunction(IDX, 35, byref(sup))   # KSJ_SUPPORT_MULTI_FRAMES
print("многокадровая сшивка поддерживается:", bool(sup.value))

# Максимальный FOV сенсора
c0, r0, cs, rs = c_int(), c_int(), c_int(), c_int()
am_c, am_r = c_int(), c_int()
ksj.KSJ_CaptureGetDefaultFieldOfView(IDX, byref(c0), byref(r0), byref(cs),
                                     byref(rs), byref(am_c), byref(am_r))
print("максимальный FOV: start=(%d,%d) size=%dx%d"
      % (c0.value, r0.value, cs.value, rs.value))

SKIPNONE = 0
rc = ksj.KSJ_CaptureSetFieldOfViewEx(IDX, 0, 0, cs.value, rs.value,
                                     SKIPNONE, SKIPNONE, LINES)
check("KSJ_CaptureSetFieldOfViewEx", rc)

# API молча подставляет ближайшие допустимые значения, поэтому читаем обратно
n = c_ushort()
ksj.KSJ_CaptureGetFieldOfViewEx(IDX, byref(c0), byref(r0), byref(cs), byref(rs),
                                byref(am_c), byref(am_r), byref(n))
print("реально принято: единичный кадр %dx%d, склейка %d шт."
      % (cs.value, rs.value, n.value))

# Усиление по сегментам сенсора (у линейки оно задаётся посегментно).
segs = c_int()
if ksj.KSJ_AWAIBA_GetSegmentNum(IDX, byref(segs)) in OK:
    for s in range(segs.value):
        g, gmin, gmax = c_int(), c_int(), c_int()
        ksj.KSJ_AWAIBA_GetGainRange(IDX, s, byref(gmin), byref(gmax))
        ksj.KSJ_AWAIBA_GetGain(IDX, s, byref(g))
        print("  сегмент %d: усиление %d (диапазон %d..%d)"
              % (s, g.value, gmin.value, gmax.value))
        # ksj.KSJ_AWAIBA_SetGain(IDX, s, gmax.value // 2)
        ksj.KSJ_AWAIBA_AutoBlackLevel(IDX, s)

# ------------------------------------------------------------- размер буфера
w, h, bits = c_int(), c_int(), c_int()
check("KSJ_CaptureGetSizeEx", ksj.KSJ_CaptureGetSizeEx(IDX, byref(w), byref(h),
                                                       byref(bits)))
W, H, CH = w.value, h.value, bits.value // 8
print("кадр: %d x %d, %d бит (%d канал(ов))" % (W, H, bits.value, CH))

buf = create_string_buffer(W * H * CH)
grab = ksj.KSJ_CaptureRawData if CH == 1 else ksj.KSJ_CaptureRgbData

# --------------------------------------------------------------- захват
print("\nидёт захват, Esc или q - выход, s - сохранить кадр")
cv2.namedWindow("jelly4", cv2.WINDOW_NORMAL)
cv2.resizeWindow("jelly4", 1024, 600)

frames, t0, saved = 0, time.time(), 0
while True:
    rc = grab(IDX, buf)
    if rc < 0:
        print("capture:", ERRORS.get(rc, rc))
        if rc in (-12, -13, -14):      # битый/невалидный/пустой кадр
            continue
        break

    img = np.frombuffer(buf, np.uint8).reshape(H, W, CH)

    cv2.imshow("jelly4", img)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break
    if key == ord('s'):
        name = "frame_%03d.png" % saved
        cv2.imwrite(name, img)
        print("сохранено:", name)
        saved += 1

    frames += 1
    dt = time.time() - t0
    if dt >= 1.0:
        print("кадров/с = %.1f  (строк/с = %.0f)" % (frames / dt, frames * H / dt))
        frames, t0 = 0, time.time()

cv2.destroyAllWindows()
ksj.KSJ_UnInit()