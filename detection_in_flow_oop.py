# -*- coding: utf-8 -*-
"""Jelly4: потоковая детекция объектов на ленте.

Полотно не собирается. Каждая строка проверяется на отклонение от фона:
появилось - копим объект, пропало на GAP_ROWS строк - объект закончился.
Границы задаёт сам объект, а не размер буфера.

Запуск:  python jelly4_stream.py          обычный
         python jelly4_stream.py --recal   заново снять калибровку
Остановка: Ctrl+C.

Устройство файла:
  1. Настройки
  2. Camera    - обёртка над KSJAPI
  3. Grabber   - поток, который только забирает порции
  4. BeltModel - фон ленты, шум, признаки, маска
  5. Assembler - сборка объекта из потока строк
  6. Saver     - фильтры, вырезка, запись
  7. main
"""

import os
import sys
import time
import threading
import queue
import csv as csvmod
from collections import deque
from ctypes import (WinDLL, c_int, c_float, c_ushort, byref,
                    create_string_buffer)

import numpy as np
import cv2

# ============================================================ 1. Настройки

SDK_DIR = r"C:\Program Files (x86)\CatchBEST\IndustryCamera\Applications\KSJShow\x64"
DLL_NAME = "KSJApi64.dll"
CAM_INDEX = 0

# -- съёмка
LINES = 256            # считываний сенсора в одной порции
EXPOSURE_MS = 0.6      # выдержка строки (диапазон 0.0021..0.8)
FIXED_RATE = 134.0     # считываний в секунду, подобрано по круглой монете
BELT_X0, BELT_X1 = 300, 1160     # границы ленты в пикселях
MM_PER_PX = 0.0879     # калибровка масштаба

# -- поиск объекта
SMOOTH = (9, 9)        # сглаживание (по столбцам, по строкам)
K_SIGMA = 6.0          # порог отклонения в сигмах шума
RUN_MIN = 4            # минимальная непрерывная серия пикселей в строке
START_PIX = 10         # столько пикселей, чтобы начать объект
KEEP_PIX = 3           # столько, чтобы продолжать (тонкие места)
GAP_ROWS = 25          # пустых строк подряд = объект кончился
BG_ALPHA = 0.02        # скорость подстройки фона по пустым строкам

# -- отбор и вырезка
MIN_ROWS, MAX_ROWS = 25, 4000
MIN_AREA = 1200        # px
MIN_SIDE = 15          # px, короткая сторона
MAX_ASPECT = 25.0
OPEN_K, CLOSE_K = 5, 21   # сначала открытие (убрать крапинки), потом замыкание
PAD = 20

# -- вывод
OUT_DIR, MASK_DIR, REJ_DIR = "crops", "crops_mask", "rejects"
CSV_FILE, CALIB_FILE = "objects.csv", "calib.npz"
SAVE_REJECTS, MAX_REJECTS = True, 60

BELT_W = BELT_X1 - BELT_X0
FEAT_NAMES = ["B", "G", "R", "R-B", "R-G"]


# =============================================================== 2. Camera

class Camera:
    """Открытие камеры, настройка режима, получение порций строк."""

    def __init__(self, sdk_dir=SDK_DIR, index=CAM_INDEX):
        os.add_dll_directory(sdk_dir)
        self.api = WinDLL(os.path.join(sdk_dir, DLL_NAME))
        self.idx = index
        self.api.KSJ_ExposureTimeSet.argtypes = (c_int, c_float)
        self.api.KSJ_SetFixedFrameRateEx.argtypes = (c_int, c_float)
        self.api.KSJ_CaptureSetFieldOfViewEx.argtypes = (c_int,) * 7 + (c_ushort,)
        self.api.KSJ_Init()

    def configure(self):
        a, i = self.api, self.idx
        a.KSJ_TriggerModeSet(i, 3)                    # фиксированная частота
        a.KSJ_SetFixedFrameRateEx(i, FIXED_RATE)
        a.KSJ_ExposureTimeSet(i, EXPOSURE_MS)
        # Весь сенсор (2048 x 2) склеивается LINES раз в одно изображение.
        a.KSJ_CaptureSetFieldOfViewEx(i, 0, 0, 2048, 2, 0, 0, LINES)

        w, h, bits = c_int(), c_int(), c_int()
        a.KSJ_CaptureGetSizeEx(i, byref(w), byref(h), byref(bits))
        self.W, self.H, self.CH = w.value, h.value, bits.value // 8
        self.buf = create_string_buffer(self.W * self.H * self.CH)
        return self.W, self.H, self.CH

    def grab(self):
        """Вернуть порцию (H, W, 3) или код ошибки (отрицательное число)."""
        rc = self.api.KSJ_CaptureRgbData(self.idx, self.buf)
        if rc < 0:
            return rc
        # copy обязателен: буфер будет перезаписан следующей порцией
        return np.frombuffer(self.buf, np.uint8).reshape(
            self.H, self.W, self.CH).copy()

    def close(self):
        self.api.KSJ_UnInit()


# ============================================================== 3. Grabber

class Grabber(threading.Thread):
    """Ничего, кроме захвата. Пока основной поток думает, камера снимает."""

    def __init__(self, cam, maxsize=64):
        super().__init__(daemon=True)
        self.cam = cam
        self.q = queue.Queue(maxsize=maxsize)
        self.ok = 0
        self.err = {}
        self.running = True

    def run(self):
        while self.running:
            p = self.cam.grab()
            if isinstance(p, int):
                self.err[p] = self.err.get(p, 0) + 1
                continue
            self.ok += 1
            try:
                self.q.put_nowait(p)
            except queue.Full:
                self.err["очередь"] = self.err.get("очередь", 0) + 1

    def get(self, timeout=5.0):
        return self.q.get(timeout=timeout)

    def stop(self):
        self.running = False


# ============================================================ 4. BeltModel

class BeltModel:
    """Что такое 'пустая лента': уровень, шум и признаки для сравнения."""

    def __init__(self):
        self.gain = np.ones((BELT_W, 3), np.float32)
        self.bg = None
        self.sigma = None

    @staticmethod
    def features(a):
        """Пять признаков: три канала плюс два цветоразностных.

        R-B и R-G ловят объекты, отличающиеся ЦВЕТОМ при той же яркости.
        Плетение ленты меняет все каналы одинаково, поэтому в разности
        каналов текстура взаимно вычитается и шум там в разы меньше.
        Кофейное зерно даёт в R-B около десяти сигм против одной в R.
        """
        b, g, r = a[..., 0], a[..., 1], a[..., 2]
        return np.stack([b, g, r, r - b, r - g], axis=-1)

    def calibrate(self, belt_rows):
        """belt_rows: (n, BELT_W, 3) пустой ленты. Хватает одной порции."""
        a = belt_rows.astype(np.float32)
        ref = np.median(a, axis=0)
        self.gain = (128.0 / np.maximum(ref, 1.0)).astype(np.float32)

        # Фон и шум меряем ПО ТЕМ ЖЕ сглаженным признакам, по каким ищем.
        # Иначе шум завышается и реальный порог выходит строже задуманного.
        f = self.features(cv2.blur(a * self.gain, SMOOTH))
        self.bg = np.median(f, axis=0)
        self.sigma = 1.4826 * np.median(np.abs(f - self.bg), axis=0) + 0.5

        sat = (ref > 245).mean()
        if sat > 0.01:
            print("  ВНИМАНИЕ: %.0f%% фона в насыщении - уменьшите выдержку"
                  " или усиление" % (100 * sat))

    def report(self):
        print("калибровка:")
        for k, nm in enumerate(FEAT_NAMES):
            print("  %-4s фон %7.1f  шум %5.2f  порог %6.1f"
                  % (nm, self.bg[:, k].mean(), self.sigma[:, k].mean(),
                     K_SIGMA * self.sigma[:, k].mean()))

    def save(self, path=CALIB_FILE):
        np.savez(path, gain=self.gain, bg=self.bg, sigma=self.sigma,
                 belt=np.array([BELT_X0, BELT_X1]))

    def load(self, path=CALIB_FILE):
        d = np.load(path)
        if tuple(d["belt"]) != (BELT_X0, BELT_X1):
            return False                      # границы ленты изменились
        self.gain, self.bg, self.sigma = d["gain"], d["bg"], d["sigma"]
        return True

    def analyse(self, belt):
        """Порция -> (маска отклонений, занятость каждой строки, признаки)."""
        f = self.features(cv2.blur(belt.astype(np.float32) * self.gain, SMOOTH))
        dev = np.abs(f - self.bg) / self.sigma
        mask = (dev.max(axis=-1) > K_SIGMA).astype(np.uint8)

        # Строка занята только по длинным непрерывным сериям: эрозия
        # горизонтальным ядром убирает одиночные крапинки текстуры.
        runs = cv2.erode(mask, np.ones((1, RUN_MIN), np.uint8))
        return mask, runs.sum(axis=1), f

    def update(self, feat_row):
        """Подстройка по ПУСТОЙ строке: объект на фон влиять не может."""
        self.bg = (1 - BG_ALPHA) * self.bg + BG_ALPHA * feat_row
        self.sigma = ((1 - BG_ALPHA) * self.sigma
                      + BG_ALPHA * (1.25 * np.abs(feat_row - self.bg) + 0.5))


# ============================================================ 5. Assembler

class Assembler:
    """Состояние сборки: превращает поток строк в законченные объекты."""

    def __init__(self):
        self.rows, self.masks = [], []
        self.pre = deque(maxlen=PAD)       # запас пустых строк перед объектом
        self.active = False
        self.gap = 0
        self.overflow = 0

    def feed(self, row, mrow, busy_count):
        """Подать одну строку. Вернуть собранный объект или None."""
        threshold = KEEP_PIX if self.active else START_PIX
        if busy_count >= threshold:
            if not self.active:
                self.rows.extend(self.pre)
                self.masks.extend([np.zeros(BELT_W, np.uint8)] * len(self.pre))
                self.pre.clear()
                self.active = True
            self.rows.append(row)
            self.masks.append(mrow)
            self.gap = 0
            if len(self.rows) > MAX_ROWS:
                self.reset()
                self.overflow += 1
            return None

        if not self.active:
            self.pre.append(row)
            return None

        self.gap += 1
        self.rows.append(row)
        self.masks.append(mrow)
        if self.gap < GAP_ROWS:
            return None

        band = (np.stack(self.rows), np.stack(self.masks).astype(np.uint8))
        self.reset()
        return band

    def reset(self):
        self.rows, self.masks = [], []
        self.active = False
        self.gap = 0


# ================================================================ 6. Saver

class Saver:
    """Чистка маски, отбор по геометрии, вырезка и запись на диск."""

    def __init__(self):
        for d in (OUT_DIR, MASK_DIR, REJ_DIR):
            os.makedirs(d, exist_ok=True)
        self.f = open(CSV_FILE, "a", newline="", encoding="utf-8")
        self.csv = csvmod.writer(self.f)
        if self.f.tell() == 0:
            self.csv.writerow(["id", "rows", "cols", "area_px", "rect_w",
                               "rect_h", "angle", "w_mm", "h_mm", "area_mm2",
                               "fill"])
        self.saved = 0
        self.rejected = 0
        self.reasons = {}
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (OPEN_K, OPEN_K))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (CLOSE_K, CLOSE_K))

    def _reject(self, why, crop=None):
        self.rejected += 1
        self.reasons[why] = self.reasons.get(why, 0) + 1
        if SAVE_REJECTS and crop is not None and self.rejected <= MAX_REJECTS:
            cv2.imwrite(os.path.join(REJ_DIR, "rej_%04d_%s.png"
                                     % (self.rejected, why)), crop)

    def process(self, img, msk):
        if img.shape[0] < MIN_ROWS:
            self._reject("korotkiy")
            return

        # ПОРЯДОК ВАЖЕН: открытие убирает крапинки, потом замыкание сшивает
        # разрывы. В обратном порядке замыкание склеивает шум в кляксы.
        msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN, self.k_open)
        msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, self.k_close)

        n, lab, st, _ = cv2.connectedComponentsWithStats(msk, 8)
        if n <= 1:
            self._reject("pusto")
            return
        for i in range(1, n):
            self._one(img, lab, st[i], i)

    def _one(self, img, lab, stat, i):
        x, y, bw, bh, area = stat
        dbg = img[max(0, y - 5):y + bh + 5, max(0, x - 5):x + bw + 5]

        if area < MIN_AREA:
            return self._reject("malo_ploshadi", dbg)
        if x < 3 or x + bw > BELT_W - 3:
            return self._reject("kray_lenty", dbg)

        comp = (lab[y:y + bh, x:x + bw] == i).astype(np.uint8)
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return
        c = max(cs, key=cv2.contourArea)
        (_, _), (w0, h0), ang = cv2.minAreaRect(c)
        long_, short_ = max(w0, h0), min(w0, h0)

        if short_ < MIN_SIDE:
            return self._reject("tonkiy", dbg)
        if long_ / max(short_, 1.0) > MAX_ASPECT:
            return self._reject("vytyanutiy", dbg)

        # Заливка внутренних дырок по внешнему контуру.
        filled = np.zeros_like(comp)
        cv2.drawContours(filled, [c], -1, 1, thickness=cv2.FILLED)

        y0, y1 = max(0, y - PAD), min(img.shape[0], y + bh + PAD)
        x0, x1 = max(0, x - PAD), min(BELT_W, x + bw + PAD)
        crop = img[y0:y1, x0:x1].copy()
        mm = np.zeros((y1 - y0, x1 - x0), np.uint8)
        mm[y - y0:y - y0 + bh, x - x0:x - x0 + bw] = filled
        masked = crop.copy()
        masked[mm == 0] = 0

        name = "obj_%05d.png" % self.saved
        cv2.imwrite(os.path.join(OUT_DIR, name), crop)
        cv2.imwrite(os.path.join(MASK_DIR, name), masked)
        self.csv.writerow([name, bh, bw, area, "%.1f" % long_, "%.1f" % short_,
                           "%.1f" % ang, "%.2f" % (long_ * MM_PER_PX),
                           "%.2f" % (short_ * MM_PER_PX),
                           "%.2f" % (area * MM_PER_PX ** 2),
                           "%.3f" % (area / max(long_ * short_, 1.0))])
        self.f.flush()
        self.saved += 1
        print("  объект %-3d %6.1f x %-5.1f мм  %4d строк  заполн %.2f  %s"
              % (self.saved, long_ * MM_PER_PX, short_ * MM_PER_PX, bh,
                 area / max(long_ * short_, 1.0), name))

    def close(self):
        self.f.close()


# ================================================================= 7. main

def main():
    recal = "--recal" in sys.argv

    cam = Camera()
    W, H, CH = cam.configure()
    print("порция %dx%d, ожидается %.2f порций/с" % (W, H, FIXED_RATE / LINES))

    grab = Grabber(cam)
    grab.start()
    t0 = time.time()

    belt_model = BeltModel()
    ready = False
    if not recal and os.path.exists(CALIB_FILE) and belt_model.load():
        belt_model.report()
        ready = True
        print("калибровка взята из файла, можно класть объекты\n")
    else:
        print("держите ленту пустой, калибровка займёт одну порцию...")

    asm = Assembler()
    saver = Saver()
    rows_total = busy_total = pieces = 0

    try:
        while True:
            try:
                piece = grab.get()
            except queue.Empty:
                print("порции не приходят:", grab.err)
                break
            pieces += 1
            belt = piece[:, BELT_X0:BELT_X1]

            if not ready:                     # одной порции достаточно
                belt_model.calibrate(belt)
                belt_model.save()
                belt_model.report()
                ready = True
                print("можно класть объекты\n")
                continue

            mask, counts, feat = belt_model.analyse(belt)

            for r in range(H):
                rows_total += 1
                band = asm.feed(belt[r], mask[r], counts[r])
                if counts[r] >= (KEEP_PIX if asm.active else START_PIX):
                    busy_total += 1
                else:
                    belt_model.update(feat[r])
                if band is not None:
                    saver.process(*band)

            if pieces % 10 == 0:
                el = time.time() - t0
                exp = el * FIXED_RATE / LINES
                print("[%5.0f с] порций %d из ~%.0f (%.0f%%) | занято строк "
                      "%.2f%% | сохранено %d | брак %d"
                      % (el, grab.ok, exp, 100.0 * grab.ok / max(exp, 1),
                         100.0 * busy_total / max(rows_total, 1),
                         saver.saved, saver.rejected))
    except KeyboardInterrupt:
        print("\nостановка по Ctrl+C")

    grab.stop()
    saver.close()
    el = time.time() - t0
    exp = el * FIXED_RATE / LINES
    print("\nработа %.1f с | порций %d из ожидаемых %.0f (%.0f%%)"
          % (el, grab.ok, exp, 100.0 * grab.ok / max(exp, 1)))
    print("строк %d, занятых %.2f%%"
          % (rows_total, 100.0 * busy_total / max(rows_total, 1)))
    print("сохранено %d, отбраковано %d %s"
          % (saver.saved, saver.rejected, saver.reasons or ""))
    print("ошибки захвата:", grab.err or "нет")
    cam.close()


if __name__ == "__main__":
    main()