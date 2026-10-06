# -*- coding: utf-8 -*-
"""Проверка камеры Jelly4. Windows и Linux ARM, версии API 6 и 8.

Отсутствующие в сборке функции не вызываются, а отмечаются в отчёте -
набор символов у версий 6.x и 8.x разный.

Linux:   LD_LIBRARY_PATH=/opt/ksjapi/arm64 python3 probe.py
Windows: python probe.py
"""

import os
import sys
import time
from ctypes import (CDLL, c_int, c_uint, c_bool, c_float, c_ushort, byref,
                    create_string_buffer)

import numpy as np

if sys.platform == "win32":
    from ctypes import WinDLL
    SDK_DIR = r"C:\Program Files (x86)\CatchBEST\IndustryCamera\Applications\KSJShow\x64"
    LIB_NAME = "KSJApi64.dll"
else:
    SDK_DIR = "/opt/ksjapi/arm64"
    LIB_NAME = "libksjapi.so"

IDX = 0
LINES = 256
EXPOSURE_MS = 0.5
FIXED_RATE = 134.0
N_PIECES = 10

# ------------------------------------------------------- загрузка библиотеки
lib_path = os.path.join(SDK_DIR, LIB_NAME)
print("библиотека:", lib_path)
if not os.path.exists(lib_path):
    sys.exit("файла нет - проверьте SDK_DIR")

if sys.platform == "win32":
    os.add_dll_directory(SDK_DIR)
    ksj = WinDLL(lib_path)
else:
    if SDK_DIR not in os.environ.get("LD_LIBRARY_PATH", ""):
        print("ВНИМАНИЕ: %s нет в LD_LIBRARY_PATH, плагины не найдутся"
              % SDK_DIR)
    print("плагин под тип 98:",
          "есть" if os.path.exists(os.path.join(SDK_DIR, "libksjcam_98.so"))
          else "НЕТ!")
    ksj = CDLL(lib_path)


def has(name):
    """Есть ли такая функция в этой сборке."""
    try:
        getattr(ksj, name)
        return True
    except AttributeError:
        return False


def call(name, *args):
    """Вызвать, если функция есть. Иначе вернуть None и сказать об этом."""
    if not has(name):
        print("  %s - нет в этой версии API" % name)
        return None
    return getattr(ksj, name)(*args)


# --------------------------------------------- порядок вызовов принципиален
# 1) Init  2) DeviceGetCount  3) Open  4) всё остальное
# В версии 8 устройство нужно открывать явно; до KSJ_DeviceGetCount
# открывать нечего, и KSJ_Open вернёт -8 (RET_FAIL).
print("\nKSJ_Init ->", ksj.KSJ_Init())

if has("KSJ_GetVersion"):
    v = [c_int() for _ in range(4)]
    ksj.KSJ_GetVersion(*[byref(x) for x in v])
    ver = tuple(x.value for x in v)
    print("версия KSJApi: %d.%d.%d.%d" % ver)

count = ksj.KSJ_DeviceGetCount()
print("камер найдено:", count)
if count < 1:
    sys.exit("камера не видна: проверьте lsusb и права udev")

# В версии 8 камера, похоже, открывается сама внутри KSJ_Init, и тогда
# KSJ_Open возвращает -8 как "уже открыто". Это не повод останавливаться:
# решает не код возврата, а то, пойдёт ли дальше захват.
rc = call("KSJ_Open", IDX)
print("KSJ_Open ->", rc)
if rc is not None and rc < 0:
    print("  открыть не удалось, но продолжаем: возможно, камера уже открыта")
    print("  (если дальше всё заработает - KSJ_Open просто не нужен)")

dev_type, serial, fw, fpga = c_ushort(), c_int(), c_ushort(), c_ushort()
ksj.KSJ_DeviceGetInformationEx(IDX, byref(dev_type), byref(serial),
                               byref(fw), byref(fpga))
print("тип=%d serial=%d fw=%d fpga=%d" % (dev_type.value, serial.value,
                                          fw.value, fpga.value))
if dev_type.value == 0:
    print("  тип нулевой - в версии 8 у KSJ_DeviceGetInformationEx может быть"
          " другая сигнатура, на захват это не влияет")

# ------------------------------------------- какие нужные функции на месте
need = ["KSJ_TriggerModeSet", "KSJ_SetFixedFrameRateEx", "KSJ_ExposureTimeSet",
        "KSJ_ExposureTimeRangeGet", "KSJ_CaptureSetFieldOfViewEx",
        "KSJ_CaptureGetSizeEx", "KSJ_CaptureSetTimeOut", "KSJ_CaptureRgbData",
        "KSJ_AWAIBA_IsUsed", "KSJ_AWAIBA_SetGain", "KSJ_WhiteBalanceSet",
        "KSJ_CaptureSetFlatFieldCorrection", "KSJ_Close"]
missing = [n for n in need if not has(n)]
print("\nнужные функции: есть %d из %d" % (len(need) - len(missing), len(need)))
if missing:
    print("отсутствуют:", ", ".join(missing))

# --------------------------------------------------------------- настройка
if has("KSJ_ExposureTimeSet"):
    ksj.KSJ_ExposureTimeSet.argtypes = (c_int, c_float)
if has("KSJ_SetFixedFrameRateEx"):
    ksj.KSJ_SetFixedFrameRateEx.argtypes = (c_int, c_float)
if has("KSJ_CaptureSetFieldOfViewEx"):
    ksj.KSJ_CaptureSetFieldOfViewEx.argtypes = (c_int,) * 7 + (c_ushort,)
if has("KSJ_CaptureSetTimeOut"):
    ksj.KSJ_CaptureSetTimeOut.argtypes = (c_int, c_uint)

print("\nнастройка:")
print("  TriggerModeSet(3) ->", call("KSJ_TriggerModeSet", IDX, 3))
print("  SetFixedFrameRateEx(%.0f) ->" % FIXED_RATE,
      call("KSJ_SetFixedFrameRateEx", IDX, FIXED_RATE))
print("  ExposureTimeSet(%.2f) ->" % EXPOSURE_MS,
      call("KSJ_ExposureTimeSet", IDX, EXPOSURE_MS))
print("  CaptureSetFieldOfViewEx(..., %d) ->" % LINES,
      call("KSJ_CaptureSetFieldOfViewEx", IDX, 0, 0, 2048, 2, 0, 0, LINES))

period = LINES / FIXED_RATE
print("  CaptureSetTimeOut(%d мс) ->" % int(period * 2000 + 5000),
      call("KSJ_CaptureSetTimeOut", IDX, int(period * 2000 + 5000)))

w, h, bits = c_int(), c_int(), c_int()
ksj.KSJ_CaptureGetSizeEx(IDX, byref(w), byref(h), byref(bits))
W, H, CH = w.value, h.value, bits.value // 8
if W * H * CH == 0:
    sys.exit("нулевой размер кадра - настройка не применилась")
buf = create_string_buffer(W * H * CH)
print("\nпорция %dx%d, %.1f МБ, период %.2f с"
      % (W, H, W * H * CH / 1e6, period))

# ------------------------------------------------------------------ замер
print("\nснимаю %d порций:" % N_PIECES)
prev, gaps, errs, k = None, [], {}, 0
while k < N_PIECES:
    rc = ksj.KSJ_CaptureRgbData(IDX, buf)
    now = time.time()
    if rc < 0:
        errs[rc] = errs.get(rc, 0) + 1
        print("  ошибка %d" % rc)
        if sum(errs.values()) > 20:
            print("  слишком много ошибок, прекращаю")
            break
        continue
    k += 1
    img = np.frombuffer(buf, np.uint8).reshape(H, W, CH)
    if prev is not None:
        dt = now - prev
        gaps.append(dt)
        print("  %2d: интервал %.3f с (лишнее %+.3f = %+.0f строк), яркость %.1f"
              % (k, dt, dt - period, (dt - period) * FIXED_RATE * 2, img.mean()))
    prev = now

if gaps:
    extra = float(np.mean(gaps)) - period
    print("\nв среднем теряется %.0f строк на границе порции"
          % max(0.0, extra * FIXED_RATE * 2))
print("ошибки:", errs or "нет")
print("""
коды: -3 функции нет, -4 устройства нет, -7 нет прав (udev/sudo),
      -8 общая ошибка, -12/-13 битый кадр, -16 таймаут, -17 закрыто""")

call("KSJ_Close", IDX)
ksj.KSJ_UnInit()
