# -*- coding: utf-8 -*-
"""Живой график яркости строки с камеры Jelly4 по трём каналам."""

import os
import sys
from ctypes import (CDLL, c_int, c_float, c_ushort, byref,
                    create_string_buffer)

import matplotlib.pyplot as plt
import numpy as np

if sys.platform == "win32":
    from ctypes import WinDLL
    SDK_DIR = r"C:\Program Files (x86)\CatchBEST\IndustryCamera\Applications\KSJShow\x64"
    LIB = "KSJApi64.dll"
else:
    SDK_DIR = "/opt/ksjapi/arm64"
    LIB = "libksjapi.so"

IDX = 0
WIDTH = 2048
LINES = 8               # маленькая порция: захват возвращается быстро
EXPOSURE_MS = 0.1
FIXED_RATE = 8000.0

# ------------------------------------------------------------------ камера
if sys.platform == "win32":
    os.add_dll_directory(SDK_DIR)
    ksj = WinDLL(os.path.join(SDK_DIR, LIB))
else:
    ksj = CDLL(os.path.join(SDK_DIR, LIB))

ksj.KSJ_ExposureTimeSet.argtypes = (c_int, c_float)
ksj.KSJ_SetFixedFrameRateEx.argtypes = (c_int, c_float)
ksj.KSJ_CaptureSetFieldOfViewEx.argtypes = (c_int,) * 7 + (c_ushort,)

ksj.KSJ_Init()
ksj.KSJ_DeviceGetCount()
if hasattr(ksj, "KSJ_Open"):
    ksj.KSJ_Open(IDX)

ksj.KSJ_TriggerModeSet(IDX, 3)
ksj.KSJ_SetFixedFrameRateEx(IDX, FIXED_RATE)
ksj.KSJ_ExposureTimeSet(IDX, EXPOSURE_MS)
ksj.KSJ_CaptureSetFieldOfViewEx(IDX, 0, 0, WIDTH, 2, 0, 0, LINES)

w, h, bits = c_int(), c_int(), c_int()
ksj.KSJ_CaptureGetSizeEx(IDX, byref(w), byref(h), byref(bits))
W, H, CH = w.value, h.value, bits.value // 8
buf = create_string_buffer(W * H * CH)

x = np.arange(W)

plt.ion()
fig, ax = plt.subplots()
line_r, = ax.plot(x, np.zeros(W), 'r-', label='R')
line_g, = ax.plot(x, np.zeros(W), 'g-', label='G')
line_b, = ax.plot(x, np.zeros(W), 'b-', label='B')
ax.set_ylim(0, 255)
ax.set_xlim(0, W)
ax.set_xlabel('Линия пикселей')
ax.set_ylabel('Яркость')
ax.legend()


def update_hist(rgb):
    """rgb - массив (пиксели, 3) одной строки."""
    line_r.set_ydata(rgb[:, 0])
    line_g.set_ydata(rgb[:, 1])
    line_b.set_ydata(rgb[:, 2])
    fig.canvas.draw()
    fig.canvas.flush_events()


def read_line():
    """Одна строка с камеры, (пиксели, 3) в порядке R, G, B."""
    if ksj.KSJ_CaptureRgbData(IDX, buf) < 0:
        return None
    piece = np.frombuffer(buf, np.uint8).reshape(H, W, CH)
    return piece[0][:, ::-1]        # камера отдаёт BGR, график ждёт RGB


while plt.fignum_exists(fig.number):
    rgb = read_line()
    if rgb is None:
        continue
    update_hist(rgb)

ksj.KSJ_UnInit()