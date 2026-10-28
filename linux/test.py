"""Проверка точности .rknn НА ПЛАТЕ (инференс на NPU).

Аналог test_model.py, но вместо PyTorch на CPU — реальный NPU через
rknn-toolkit-lite2. Берёт картинки из тестовой папки, сравнивает
предсказания с именем папки-класса и меряет время.

Требует rknn-toolkit-lite2 (пакет rknnlite).
"""

import os
import random
import time

import cv2
import numpy as np
import yaml
from rknnlite.api import RKNNLite

# ---------------------------------------------------------------- настройки
TEST_DIR = 'linux/data/data_for_nn/test'   # папка с подпапками-классами
N_SAMPLES = 12                             # 0 или None -> вся папка
SEED = 0
CONFIG_PATH = 'models.yaml'
GRID_PATH = 'preview_npu.png'              # None — не сохранять
CORE_MASK = RKNNLite.NPU_CORE_0            # NPU_CORE_0_1_2 — три ядра
WARMUP = 10
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


def preprocess(path: str, size: int):
    """Файл -> (вход NPU [1,H,W,3] uint8 RGB, картинка BGR для превью).

    В NPU идёт uint8 БЕЗ деления на 255: нормализацию выполняет сам NPU
    по mean_values/std_values, заданным при сборке модели.
    """
    bgr = cv2.imread(path, flags=cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f'не читается: {path}')

    padded = resize_pad(bgr, size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb, 0), padded


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def collect_samples(test_dir: str, n):
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    samples = []
    for class_name in sorted(os.listdir(test_dir)):
        class_root = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_root):
            continue
        for name in sorted(os.listdir(class_root)):
            if os.path.splitext(name)[1].lower() in exts:
                samples.append((os.path.join(class_root, name), class_name))

    if not samples:
        raise RuntimeError(f'не найдено изображений в {test_dir}')

    random.seed(SEED)
    random.shuffle(samples)
    return samples[:n] if n else samples


def make_grid(previews, labels, cell=96, cols=6):
    rows = (len(previews) + cols - 1) // cols
    grid = np.full((rows * cell, cols * cell, 3), 30, dtype=np.uint8)
    for i, (img, (text, ok)) in enumerate(zip(previews, labels)):
        r, c = divmod(i, cols)
        tile = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_NEAREST)
        color = (80, 220, 80) if ok else (60, 60, 230)
        cv2.rectangle(tile, (0, 0), (cell - 1, cell - 1), color, 2)
        cv2.putText(tile, text, (4, cell - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, color, 1, cv2.LINE_AA)
        grid[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = tile
    return grid


def stats(name: str, samples_us) -> None:
    arr = np.asarray(samples_us, dtype=np.float64)
    print(f'{name:<16} среднее {arr.mean():9.1f} мкс   медиана {np.median(arr):9.1f}   '
          f'мин {arr.min():9.1f}   макс {arr.max():9.1f}')


def main():
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    size = cfg['input_size'][2]
    classes = cfg['classes']
    rknn_path = cfg['rknn']

    if not os.path.exists(rknn_path):
        raise FileNotFoundError(f'нет модели: {rknn_path}')

    samples = collect_samples(TEST_DIR, N_SAMPLES)

    rknn = RKNNLite()
    try:
        if rknn.load_rknn(rknn_path) != 0:
            raise RuntimeError('load_rknn failed')
        if rknn.init_runtime(core_mask=CORE_MASK) != 0:
            raise RuntimeError('init_runtime failed')

        print(f'модель: {rknn_path}')
        print(f'классы: {classes}   вход: {size}x{size}')
        print(f'картинок: {len(samples)}\n')

        warm_input, _ = preprocess(samples[0][0], size)
        for _ in range(WARMUP):
            rknn.inference(inputs=[warm_input])

        previews, labels, correct = [], [], 0
        pre_us, inf_us = [], []

        for path, true_class in samples:
            t0 = time.perf_counter_ns()
            npu_input, preview = preprocess(path, size)
            t1 = time.perf_counter_ns()
            out = rknn.inference(inputs=[npu_input])
            t2 = time.perf_counter_ns()

            pre_us.append((t1 - t0) / 1e3)
            inf_us.append((t2 - t1) / 1e3)

            logits = np.asarray(out[0]).ravel()
            idx = int(np.argmax(logits))          # argmax по логитам,
            probs = softmax(logits)               # softmax только для вывода
            pred_class, conf = classes[idx], float(probs[idx])

            ok = pred_class == true_class
            correct += ok
            print(f'{"OK " if ok else "ERR"}  {os.path.basename(path):<24} '
                  f'истина={true_class:<10} предсказано={pred_class:<10} '
                  f'p={conf:.3f}   препроц {pre_us[-1]:8.1f} мкс   '
                  f'инференс {inf_us[-1]:8.1f} мкс')

            previews.append(preview)
            labels.append((f'{pred_class} {conf:.2f}', ok))

        print(f'\nверно: {correct}/{len(samples)}  ({correct / len(samples):.1%})')

        print('\n--- время (NPU) ---')
        stats('препроцессинг', pre_us)
        stats('инференс', inf_us)
        total = float(np.mean(pre_us) + np.mean(inf_us))
        print(f'{"итого":<16} среднее {total:9.1f} мкс   ({1e6 / total:.0f} FPS)')

        if GRID_PATH:
            cv2.imwrite(GRID_PATH, make_grid(previews, labels))
            print(f'\nпревью: {GRID_PATH}')
    finally:
        rknn.release()


if __name__ == '__main__':
    main()