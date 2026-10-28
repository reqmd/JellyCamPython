"""Замер времени .rknn НА ПЛАТЕ.

Требует rknn-toolkit-lite2 (пакет rknnlite), а не rknn-toolkit2.
Полный toolkit на плате без target=... считает модель в программной
симуляции на CPU и показывает время, не связанное с NPU.

Установка на плате:
    pip install rknn_toolkit_lite2-*-cp*-linux_aarch64.whl
"""

import os
import time

import cv2
import numpy as np
import yaml
from rknnlite.api import RKNNLite

# ---------------------------------------------------------------- настройки
CONFIG_PATH = 'models.yaml'
TEST_IMAGE = None            # None -> случайный шум нужного размера
WARMUP = 20
REPEATS = 200
CORE_MASK = RKNNLite.NPU_CORE_0        # NPU_CORE_0_1_2 — три ядра разом
FILL = (0, 0, 0)
# ---------------------------------------------------------------------------


def resize_pad(image: np.ndarray, size: int, fill=FILL) -> np.ndarray:
    """Тот же препроцессинг, что в train.py. Вход/выход BGR."""
    h, w = image.shape[:2]
    scale = size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    y0, x0 = (size - new_h) // 2, (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def stats(name: str, samples_us) -> None:
    arr = np.asarray(samples_us, dtype=np.float64)
    print(f'{name:<16} среднее {arr.mean():9.1f} мкс   медиана {np.median(arr):9.1f}   '
          f'мин {arr.min():9.1f}   макс {arr.max():9.1f}')


def main():
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    size = cfg['input_size'][2]
    rknn_path = cfg['rknn']

    if not os.path.exists(rknn_path):
        raise FileNotFoundError(f'нет модели: {rknn_path}')

    # Вход NPU: uint8 RGB, БЕЗ деления на 255 — нормализацию по
    # mean_values/std_values выполняет сам NPU по конфигу сборки.
    if TEST_IMAGE:
        bgr = cv2.imread(TEST_IMAGE, flags=cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f'не читается: {TEST_IMAGE}')
        img = cv2.cvtColor(resize_pad(bgr, size), cv2.COLOR_BGR2RGB)
    else:
        img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)

    img = np.expand_dims(img, 0)           # [1, H, W, 3], NHWC

    rknn = RKNNLite()
    try:
        print(f'--> загрузка {rknn_path}')
        if rknn.load_rknn(rknn_path) != 0:
            raise RuntimeError('load_rknn failed')

        print('--> init_runtime (NPU)')
        if rknn.init_runtime(core_mask=CORE_MASK) != 0:
            raise RuntimeError('init_runtime failed')

        print(f'--> прогрев ({WARMUP})')
        for _ in range(WARMUP):
            rknn.inference(inputs=[img])

        print(f'--> замер ({REPEATS})')
        per_call = []
        for _ in range(REPEATS):
            t0 = time.perf_counter_ns()
            out = rknn.inference(inputs=[img])
            per_call.append((time.perf_counter_ns() - t0) / 1e3)

        print()
        stats('inference()', per_call)
        mean_us = float(np.mean(per_call))
        print(f'{"пропускная":<16} {1e6 / mean_us:9.0f} FPS (одно ядро NPU)')
        print(f'\nвыход: {np.asarray(out[0]).ravel()}')
    finally:
        rknn.release()


if __name__ == '__main__':
    main()