# -*- coding: utf-8 -*-
"""Jelly4: захват -> детекция -> классификация на NPU Rockchip.

Схема как в прежнем flow_with_nn: два потока и очередь между ними.
  grabber      - только забирает порции у камеры
  capture_loop - разбирает порции на строки, собирает объекты, кладёт в очередь
  inference_loop - достаёт из очереди, режет на компоненты, классифицирует

Запуск: LD_LIBRARY_PATH=/opt/ksjapi/arm64 python3 flow_with_nn.py
"""

import datetime
import os
import queue
import sys
import threading
import time
from ctypes import (CDLL, c_int, c_uint, c_float, c_ushort, byref,
                    create_string_buffer)

import cv2
import numpy as np
import yaml

if sys.platform == "win32":
    from ctypes import WinDLL

# ================================================================== камера
if sys.platform == "win32":
    SDK_DIR = r"C:\Program Files (x86)\CatchBEST\IndustryCamera\Applications\KSJShow\x64"
    DLL_NAME = "KSJApi64.dll"
else:
    SDK_DIR = "/opt/ksjapi/arm64"
    DLL_NAME = "libksjapi.so"
IDX = 0
LINES = 1024
EXPOSURE_MS = 0.25
FIXED_RATE = 134.0

WIDTH = 2048
X0, X1 = 400, 1620
GAIN = 140

# ============================================================== детекция
TH = np.array([15, 15, 15])
MIN_FG_PIXELS = 12
GAP_ROWS = 30
MAX_ROWS = 2000
SEAM_W, SEAM_H = 0.8, 40
MIN_GRAD = 8.0
MIN_AREA = 600

# ========================================================== классификация
CLASSES = {0: 'Скрепка', 1: 'Зерно'}
INPUT_SIZE = 224          # как при обучении (mobilenet_v3_small, 224x224)
USE_RKNN = True           # False - считать на torch, для отладки на ПК
NEED_SAVE = True          # сохранять вырезки в сессию

# ------------------------------------------------------------------ модели
RKNN_PATH = TORCH_PATH = None
if os.path.exists('models.yaml'):
    with open('models.yaml', 'r') as file:
        data = yaml.safe_load(file)
        RKNN_PATH = data.get('rknn')
        TORCH_PATH = data.get('torch')

# ------------------------------------------------------- инициализация камеры
if sys.platform == "win32":
    os.add_dll_directory(SDK_DIR)
    ksj = WinDLL(os.path.join(SDK_DIR, DLL_NAME))
else:
    if SDK_DIR not in os.environ.get("LD_LIBRARY_PATH", ""):
        print("ВНИМАНИЕ: %s нет в LD_LIBRARY_PATH" % SDK_DIR)
    ksj = CDLL(os.path.join(SDK_DIR, DLL_NAME))

ksj.KSJ_ExposureTimeSet.argtypes = (c_int, c_float)
ksj.KSJ_SetFixedFrameRateEx.argtypes = (c_int, c_float)
ksj.KSJ_CaptureSetFieldOfViewEx.argtypes = (c_int,) * 7 + (c_ushort,)
ksj.KSJ_CaptureSetTimeOut.argtypes = (c_int, c_uint)

ksj.KSJ_Init()
count = ksj.KSJ_DeviceGetCount()
if count < 1:
    raise SystemExit("камера не найдена")
rc = ksj.KSJ_Open(IDX)
if rc < 0:
    raise SystemExit("KSJ_Open вернул %d: проверьте права (udev)" % rc)

ksj.KSJ_TriggerModeSet(IDX, 3)
ksj.KSJ_SetFixedFrameRateEx(IDX, FIXED_RATE)
ksj.KSJ_ExposureTimeSet(IDX, EXPOSURE_MS)
ksj.KSJ_CaptureSetFieldOfViewEx(IDX, 0, 0, 2048, 2, 0, 0, LINES)

piece_sec = LINES / FIXED_RATE
ksj.KSJ_CaptureSetTimeOut(IDX, int(piece_sec * 2000 + 5000))

segs = c_int()
if ksj.KSJ_AWAIBA_GetSegmentNum(IDX, byref(segs)) >= 0:
    for sgm in range(segs.value):
        ksj.KSJ_AWAIBA_SetGain(IDX, sgm, GAIN)
        ksj.KSJ_AWAIBA_AutoBlackLevel(IDX, sgm)

w, h, bits = c_int(), c_int(), c_int()
ksj.KSJ_CaptureGetSizeEx(IDX, byref(w), byref(h), byref(bits))
W, H, CH = w.value, h.value, bits.value // 8
buf = create_string_buffer(W * H * CH)
print("камера готова: порция %dx%d, набирается %.1f с" % (W, H, piece_sec))

# ------------------------------------------------------------------ очереди
frames = queue.Queue(maxsize=64)      # порции от камеры
obj_queue = queue.Queue(maxsize=20)   # собранные объекты на классификацию
stats = {"ok": 0, "err": {}}
running = True
background = None


def grabber():
    """Только захват: забрал порцию - сразу просит следующую."""
    while running:
        rc = ksj.KSJ_CaptureRgbData(IDX, buf)
        if rc < 0:
            stats["err"][rc] = stats["err"].get(rc, 0) + 1
            print("ошибка захвата %d" % rc)
            continue
        stats["ok"] += 1
        piece = np.frombuffer(buf, np.uint8).reshape(H, W, CH).copy()
        try:
            frames.put_nowait(piece)
        except queue.Full:
            stats["err"]["очередь"] = stats["err"].get("очередь", 0) + 1


def read_line():
    """Одна строка (X1-X0, 3). Фон пересчитывается на каждой новой порции."""
    global background
    if read_line.rows is None or read_line.pos >= read_line.rows.shape[0]:
        read_line.rows = frames.get(timeout=piece_sec * 3 + 10)
        read_line.pos = 0
        band = read_line.rows[:, X0:X1]
        background = np.median(band, axis=0).astype(np.int16)
        sat = (band >= 250).mean()
        if sat > 0.01 and time.time() - read_line.last_warn > 60:
            read_line.last_warn = time.time()
            print("ПЕРЕСВЕТ: %.1f%% в насыщении" % (100 * sat))
    row = read_line.rows[read_line.pos, X0:X1]
    read_line.pos += 1
    return row


read_line.rows = None
read_line.pos = 0
read_line.last_warn = 0.0

date = datetime.datetime.now()
session_name = date.strftime("%d-%m-%Y_%H-%M-%S")
os.makedirs(f"data/sessions/{session_name}", exist_ok=True)


# ================================================================ детекция
def row_has_object(row):
    """True, если строка заметно отличается от фона."""
    diff = row.astype(np.int16) - background
    diff -= np.median(diff, axis=0).astype(np.int16)
    fg = (np.abs(diff) > TH).any(axis=1)
    return int(fg.sum()) >= MIN_FG_PIXELS


def find_obj(image, session_name=None, need_save=True, threshold=(25, 25, 25),
             obj_counter=[0]):
    """image - собранный блок (строки, ширина, 3). Возвращает список вырезок."""
    r_th, g_th, b_th = threshold
    d = image.astype(np.int16) - background
    d -= np.median(d, axis=1, keepdims=True).astype(np.int16)
    diff = np.abs(d)

    mask = ((diff[:, :, 0] > r_th) |
            (diff[:, :, 1] > g_th) |
            (diff[:, :, 2] > b_th)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    n, labels, stats_, centroids = cv2.connectedComponentsWithStats(mask, 8)
    W_field = image.shape[1]
    objects = []

    for i in range(1, n):
        x, y, w_, h_, area = stats_[i]
        if area < MIN_AREA:
            continue
        if w_ > SEAM_W * W_field and h_ < SEAM_H:
            continue
        patch = image[y:y + h_, x:x + w_].astype(np.float32).mean(axis=2)
        if float(np.percentile(np.abs(np.diff(patch, axis=1)), 97)) < MIN_GRAD:
            continue

        crop = image[y:y + h_, x:x + w_]          # BGR, как отдаёт камера
        fname = None
        if need_save:
            fname = f"data/sessions/{session_name}/object_{obj_counter[0]}.png"
            cv2.imwrite(fname, crop)
        objects.append((crop, fname))
        obj_counter[0] += 1

    return objects


# ============================================================ классификация
def prepare(crop):
    """Вырезка -> вход сети: RGB, квадрат INPUT_SIZE, NHWC uint8.

    Камера отдаёт BGR, а обучение шло на RGB (в train.py стоит
    cvtColor BGR2RGB после imread). Без конвертации классы перепутаются.
    """
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return np.expand_dims(img, axis=0)            # (1, H, W, 3) uint8


class Classifier:
    """Обёртка: RKNN на NPU либо torch на CPU.

    RKNNLite инициализируется ОДИН раз, а не на каждый объект:
    load_rknn + init_runtime занимают сотни миллисекунд.
    Нормализация (mean/std) задаётся при конвертации в rknn.config,
    поэтому сюда приходит сырой uint8.
    """

    def __init__(self, use_rknn=True):
        self.use_rknn = use_rknn
        if use_rknn:
            from rknnlite.api import RKNNLite
            self.rknn = RKNNLite()
            if self.rknn.load_rknn(RKNN_PATH) != 0:
                raise SystemExit("не загрузилась модель " + str(RKNN_PATH))
            if self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
                raise SystemExit("не инициализировался NPU")
            # Прогрев: первый вызов всегда дольше остальных.
            warm = np.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
            for _ in range(3):
                self.rknn.inference(inputs=[warm])
            print("NPU готов:", RKNN_PATH)
        else:
            import torch
            from torchvision import models, transforms
            import torch.nn as nn
            m = models.mobilenet_v3_small()
            m.classifier[3] = nn.Linear(m.classifier[3].in_features,
                                        len(CLASSES))
            m.load_state_dict(torch.load(TORCH_PATH, map_location='cpu'))
            m.eval()
            self.torch = torch
            self.model = m
            self.tf = transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])
            print("torch готов:", TORCH_PATH)

    def predict(self, crop):
        batch = prepare(crop)
        if self.use_rknn:
            out = self.rknn.inference(inputs=[batch])
            return int(np.argmax(out[0]))
        t = self.torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
        with self.torch.no_grad():
            return int(self.model(self.tf(t)).argmax(1).item())

    def close(self):
        if self.use_rknn:
            self.rknn.release()


# ================================================================== потоки
def capture_loop():
    """Разбирает порции на строки и собирает объекты в очередь."""
    try:
        collecting = False
        obj_rows = []
        gap = 0
        t_start = None
        while running:
            try:
                row = read_line()
            except queue.Empty:
                print('ВНИМАНИЕ! ДОЛГИЙ ОТКЛИК ОТ КАМЕРЫ!', stats)
                continue

            if row_has_object(row):
                if not collecting:
                    t_start = time.perf_counter()
                obj_rows.append(row)
                collecting = True
                gap = 0
                if len(obj_rows) > MAX_ROWS:
                    print("объект длиннее %d строк, сброшен" % MAX_ROWS)
                    obj_rows = []
                    collecting = False
            else:
                if collecting:
                    gap += 1
                    obj_rows.append(row)
                    if gap >= GAP_ROWS:
                        image = np.stack(obj_rows)
                        try:
                            # t_start - момент, когда объект только появился
                            # под камерой. От него и считаем полное время.
                            obj_queue.put_nowait((image, t_start))
                        except queue.Full:
                            print("очередь полна, объект пропущен")
                        obj_rows = []
                        collecting = False
                        gap = 0
    except Exception:
        import traceback
        print('Основной цикл захвата упал')
        traceback.print_exc()


def inference_loop():
    """Режет блок на компоненты и классифицирует каждую."""
    try:
        clf = Classifier(use_rknn=USE_RKNN)
        while running:
            image, t_start = obj_queue.get()
            t_detect = time.perf_counter()

            objects = find_obj(image, session_name=session_name,
                               need_save=NEED_SAVE)
            t_found = time.perf_counter()

            for crop, fname in objects:
                cls = CLASSES[clf.predict(crop)]
                t_done = time.perf_counter()
                # Полное время: от появления объекта под камерой до ответа сети.
                print("%-8s %4dx%-4d  полное %6.0f мс  "
                      "(сбор %5.0f, выделение %4.0f, сеть %4.0f)  %s"
                      % (cls, crop.shape[1], crop.shape[0],
                         (t_done - t_start) * 1000,
                         (t_detect - t_start) * 1000,
                         (t_found - t_detect) * 1000,
                         (t_done - t_found) * 1000,
                         os.path.basename(fname) if fname else ''))

            obj_queue.task_done()
    except Exception:
        import traceback
        print('Inference loop упал')
        traceback.print_exc()


threading.Thread(target=grabber, daemon=True).start()
t1 = threading.Thread(target=capture_loop, daemon=True)
t2 = threading.Thread(target=inference_loop, daemon=True)
t1.start()
t2.start()

try:
    while t1.is_alive() and t2.is_alive():
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nостановка")

running = False
print("порций принято:", stats["ok"], "ошибки:", stats["err"] or "нет")
if hasattr(ksj, "KSJ_Close"):
    ksj.KSJ_Close(IDX)
ksj.KSJ_UnInit()
