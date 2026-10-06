# -*- coding: utf-8 -*-
"""Перенос flow.py на камеру Jelly4.

Логика та же, что была: фон - одна строка, абсолютные пороги по каналам,
row_has_object считает отклонившиеся пиксели, конечный автомат собирает
объект, find_obj режет его на компоненты и сохраняет.

Изменился только источник строк: вместо UDP-запросов к сетевой камере
строки приходят из Jelly4 порциями и разбираются генератором.
"""

import numpy as np
import cv2
import datetime
import os
import sys
import threading
import queue
from ctypes import (CDLL, c_int, c_uint, c_float, c_ushort, byref,
                    create_string_buffer)

if sys.platform == "win32":
    from ctypes import WinDLL

# ------------------------------------------------------------------ камера
if sys.platform == "win32":
    SDK_DIR = r"C:\Program Files (x86)\CatchBEST\IndustryCamera\Applications\KSJShow\x64"
    DLL_NAME = "KSJApi64.dll"
else:
    SDK_DIR = "/opt/ksjapi/arm64"      # папка со ВСЕМ набором .so
    DLL_NAME = "libksjapi.so"
IDX = 0
LINES = 1024
EXPOSURE_MS = 0.25
FIXED_RATE = 134.0

WIDTH = 2048
# Края листа лежат на 330 и 1695 и постоянно шевелятся: внутри кадра
# картинка уезжает влево на ~5 px, на границе порции скачком возвращается,
# плюс сам лист уползает вправо. Берём окно СТРОГО ВНУТРИ листа, чтобы
# подвижные края в него не попадали.
X0, X1 = 400, 1620

GAIN = 140                          # усиление обоих сегментов сенсора
TH = np.array([15, 15, 15])         # пороги по каналам
# Проволока скрепки ~9 px, строка через неё даёт 20-40 отличающихся пикселей.
# Прежние 50 отсекали скрепки почти полностью.
MIN_FG_PIXELS = 12                  # отклонившихся пикселей в строке = "объект есть"
GAP_ROWS = 30                       # столько подряд фоновых строк = объект закончился
# Скрепка 30 мм при 0.088 мм/px это ~340 строк, под углом больше.
MAX_ROWS = 2000                     # предохранитель от бесконечного объекта
SEAM_W = 0.8                        # шире такой доли поля и низкий = шов, не объект
SEAM_H = 40
# Мера структурности: 97-й процентиль модуля горизонтального градиента.
# Среднее не годится - у тонкой скрепки рамка в основном пустая и среднее
# размывается. Процентилю достаточно, чтобы 3% пикселей были краями.
# Замеры: скрепки 25-35, однотонные куски фона и ленты 2-4.
MIN_GRAD = 8.0
ALPHA = 0.01

BG_ROWS = 200                       # строк пустой ленты для расчёта фона

if sys.platform == "win32":
    os.add_dll_directory(SDK_DIR)
    ksj = WinDLL(os.path.join(SDK_DIR, DLL_NAME))
else:
    # libksjapi.so подгружает соседние libksjcam_NN.so через LD_LIBRARY_PATH,
    # а она читается при СТАРТЕ процесса - задавать её здесь уже поздно.
    # Запускайте через run.sh или пропишите папку в /etc/ld.so.conf.d/.
    if SDK_DIR not in os.environ.get("LD_LIBRARY_PATH", ""):
        print("ВНИМАНИЕ: %s нет в LD_LIBRARY_PATH, камера может не открыться"
              % SDK_DIR)
    ksj = CDLL(os.path.join(SDK_DIR, DLL_NAME))
ksj.KSJ_ExposureTimeSet.argtypes = (c_int, c_float)
ksj.KSJ_SetFixedFrameRateEx.argtypes = (c_int, c_float)
ksj.KSJ_CaptureSetFieldOfViewEx.argtypes = (c_int,) * 7 + (c_ushort,)
ksj.KSJ_CaptureSetTimeOut.argtypes = (c_int, c_uint)

ksj.KSJ_Init()
ksj.KSJ_TriggerModeSet(IDX, 3)
ksj.KSJ_SetFixedFrameRateEx(IDX, FIXED_RATE)
ksj.KSJ_ExposureTimeSet(IDX, EXPOSURE_MS)
ksj.KSJ_CaptureSetFieldOfViewEx(IDX, 0, 0, 2048, 2, 0, 0, LINES)

# Таймаут чтения кадра. По умолчанию он короткий, а одна порция набирается
# LINES / FIXED_RATE секунд: при 1024 и 134 это 7.6 с. Без этой настройки
# захват возвращает -16 (RET_TIMEOUT) и в очередь ничего не попадает.
piece_sec = LINES / FIXED_RATE
timeout_ms = int(piece_sec * 2000 + 5000)
ksj.KSJ_CaptureSetTimeOut(IDX, timeout_ms)
got = c_uint()
ksj.KSJ_CaptureGetTimeOut(IDX, byref(got))
print("порция набирается %.1f с, таймаут захвата %d мс" % (piece_sec, got.value))

# баланс белого: ставить ТОЛЬКО после KSJ_Init, иначе не применится
# ksj.KSJ_WhiteBalanceSet.argtypes = (c_int, c_int)
# ksj.KSJ_WhiteBalanceMatrixSet.argtypes = (c_int, c_float * 3)
# ksj.KSJ_WhiteBalanceSet(IDX, 8)        # аппаратный ручной
# ksj.KSJ_WhiteBalanceMatrixSet(IDX, (c_float * 3)(1.60, 1.29, 1.57))

# Открытие устройства. На Linux (API 8.x) обязательно и строго после
# KSJ_DeviceGetCount, иначе всё вернёт -19 (камера не создана).
count = ksj.KSJ_DeviceGetCount()
if count < 1:
    raise SystemExit("камера не найдена")
rc = ksj.KSJ_Open(IDX)
if rc < 0:
    raise SystemExit("KSJ_Open вернул %d: проверьте права (udev) "
                     "и не занята ли камера" % rc)

# Усиление НЕ сохраняется между запусками и у сегментов может разъезжаться
# (видели 29 и 0 - отсюда чёрные кадры и полосатость). Ставим явно и
# одинаково обоим, иначе два ряда сенсора дадут чередующиеся по яркости
# строки.
segs = c_int()
if ksj.KSJ_AWAIBA_GetSegmentNum(IDX, byref(segs)) >= 0:
    for sgm in range(segs.value):
        ksj.KSJ_AWAIBA_SetGain(IDX, sgm, GAIN)
        ksj.KSJ_AWAIBA_AutoBlackLevel(IDX, sgm)
    print("усиление %d выставлено на %d сегментах" % (GAIN, segs.value))

w, h, bits = c_int(), c_int(), c_int()
ksj.KSJ_CaptureGetSizeEx(IDX, byref(w), byref(h), byref(bits))
W, H, CH = w.value, h.value, bits.value // 8
buf = create_string_buffer(W * H * CH)
print("порция %dx%d, %d канал(ов)" % (W, H, CH))

frames = queue.Queue(maxsize=64)
stats = {"ok": 0, "err": {}}
running = True


def grabber():
    while running:
        rc = ksj.KSJ_CaptureRgbData(IDX, buf)
        if rc < 0:
            stats["err"][rc] = stats["err"].get(rc, 0) + 1
            # Печатаем ошибку, иначе проблема выглядит как зависание.
            print("ошибка захвата %d (всего таких: %d)"
                  % (rc, stats["err"][rc]))
            continue
        stats["ok"] += 1
        piece = np.frombuffer(buf, np.uint8).reshape(H, W, CH).copy()
        try:
            frames.put_nowait(piece)
        except queue.Full:
            stats["err"]["очередь"] = stats["err"].get("очередь", 0) + 1
            print("очередь переполнена, порция потеряна")


threading.Thread(target=grabber, daemon=True).start()


def read_line():
    """Аналог прежнего read_line: отдаёт одну строку (X1-X0, 3).

    При получении новой порции пересчитывает фон по её собственным строкам.
    """
    global background
    while True:
        if read_line.rows is None or read_line.pos >= read_line.rows.shape[0]:
            read_line.rows = frames.get(timeout=piece_sec * 3 + 10)
            read_line.pos = 0
            band = read_line.rows[:, X0:X1]
            background = np.median(band, axis=0).astype(np.int16)
            # Пересвет убивает детекцию: всё, что ярче объекта, сливается.
            sat = (band >= 250).mean()
            if sat > 0.01:
                print("ПЕРЕСВЕТ: %.1f%% пикселей в насыщении, уменьшите "
                      "GAIN или EXPOSURE_MS" % (100 * sat))
        row = read_line.rows[read_line.pos, X0:X1]
        read_line.pos += 1
        return row


read_line.rows = None
read_line.pos = 0

# ------------------------------------------------------------------ сессия
date = datetime.datetime.now()
session_name = date.strftime("%d-%m-%Y_%H-%M-%S")
os.makedirs(f"data/sessions/{session_name}", exist_ok=True)

# ------------------------------------------------------------------- фон
# background.npy больше не нужен. Фон считается заново для КАЖДОЙ порции
# как медиана её собственных строк. Это снимает ступеньку яркости между
# порциями и заодно дрейф освещения за время работы.
background = None
print("можно класть объекты")


def row_has_object(row):
    """True, если строка заметно отличается от фона."""
    diff = row.astype(np.int16) - background
    # Убираем общий сдвиг уровня этой строки: при скачке яркости целой
    # строки медиана уезжает вместе с ней, и строка остаётся "пустой".
    diff -= np.median(diff, axis=0).astype(np.int16)
    fg = (np.abs(diff) > TH).any(axis=1)
    n = int(fg.sum())
    if n > peak["fg"]:
        peak["fg"] = n
    return n >= MIN_FG_PIXELS


peak = {"fg": 0}


REJECTS = {}


def rej(why):
    REJECTS[why] = REJECTS.get(why, 0) + 1


def find_obj(image, session_name, threshold=(25, 25, 25), obj_counter=[0]):
    """image - собранный объект (строки, ширина, 3). Выделяет и сохраняет."""
    r_th, g_th, b_th = threshold
    d = image.astype(np.int16) - background
    d -= np.median(d, axis=1, keepdims=True).astype(np.int16)
    diff = np.abs(d)

    mask = ((diff[:, :, 0] > r_th) |
            (diff[:, :, 1] > g_th) |
            (diff[:, :, 2] > b_th)).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    n, labels, stats_, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    MIN_AREA = 600
    found = 0
    W_field = image.shape[1]
    print("  собран блок %d строк, компонент %d" % (image.shape[0], n - 1))
    for i in range(1, n):
        x, y, w, h, area = stats_[i]
        if area < MIN_AREA:
            print("    отброшен: площадь %d < %d (%dx%d)" % (area, MIN_AREA, w, h))
            rej("площадь")
            continue
        if w > SEAM_W * W_field and h < SEAM_H:
            print("    отброшен как шов: %dx%d" % (w, h))
            rej("шов")
            continue

        # Однотонное пятно - это не объект, а кусок фона или ленты,
        # попавший под порог на переходе бумага-лента.
        patch = image[y:y + h, x:x + w].astype(np.float32).mean(axis=2)
        grad = float(np.percentile(np.abs(np.diff(patch, axis=1)), 97))
        if grad < MIN_GRAD:
            print("    отброшен как однотонный: градиент %.2f < %.1f (%dx%d,"
                  " яркость %.0f)" % (grad, MIN_GRAD, w, h, patch.mean()))
            rej("однотонный")
            continue
        crop = image[y:y + h, x:x + w]
        fname = f"data/sessions/{session_name}/object_{obj_counter[0]}.png"
        # Jelly4 отдаёт BGR, cv2.imwrite ждёт BGR - разворот не нужен
        cv2.imwrite(fname, crop)
        obj_counter[0] += 1
        found += 1
    if found:
        print("сохранено объектов:", found)


# ---------------- основной цикл: конечный автомат ----------------
collecting = False
obj_rows = []
gap = 0

try:
    while True:
        try:
            row = read_line()
        except queue.Empty:
            print('ВНИМАНИЕ! ДОЛГИЙ ОТКЛИК ОТ КАМЕРЫ!', stats)
            continue

        if row_has_object(row):
            obj_rows.append(row)
            collecting = True
            gap = 0
            if len(obj_rows) > MAX_ROWS:      # слишком длинный - сбрасываем
                print("объект длиннее %d строк, сброшен" % MAX_ROWS)
                obj_rows = []
                collecting = False
        else:
            # background = ((1 - ALPHA) * background + ALPHA * row).astype(np.int16)
            if collecting:
                gap += 1
                obj_rows.append(row)          # фоновые строки в "хвост"
                if gap >= GAP_ROWS:           # объект точно закончился
                    image = np.stack(obj_rows)
                    find_obj(image, session_name)
                    obj_rows = []
                    collecting = False
                    gap = 0
except KeyboardInterrupt:
    print("\nостановка")

running = False
print("\nотбраковано по причинам:", REJECTS or "ничего")
print("порций принято:", stats["ok"], "ошибки:", stats["err"] or "нет")
print("максимум отклонившихся пикселей в строке за сеанс:", peak["fg"],
      "(порог MIN_FG_PIXELS = %d)" % MIN_FG_PIXELS)
ksj.KSJ_UnInit()
