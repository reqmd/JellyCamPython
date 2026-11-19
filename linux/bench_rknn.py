"""Замер времени .rknn НА ПЛАТЕ.

Требует rknn-toolkit-lite2 (пакет rknnlite), а не rknn-toolkit2.
Полный toolkit на плате без target=... считает модель в программной
симуляции на CPU и показывает время, не связанное с NPU.

Установка на плате:
    pip install rknn_toolkit_lite2-*-cp*-linux_aarch64.whl
"""

import argparse
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


def resize_pad(image: np.ndarray, width: int, height: int | None = None,
               fill=FILL) -> np.ndarray:
    """Сохранение пропорций, паддинг до width x height (BGR)."""
    height = int(height if height is not None else width)
    width = int(width)
    if width <= 0 or height <= 0:
        raise ValueError('ширина и высота входа должны быть положительными')
    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError('пустое изображение')
    scale = min(width / w, height / h)
    new_w = min(width, max(1, round(w * scale)))
    new_h = min(height, max(1, round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), fill, dtype=np.uint8)
    y0, x0 = (height - new_h) // 2, (width - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def stats(name: str, samples_us) -> None:
    arr = np.asarray(samples_us, dtype=np.float64)
    print(f'{name:<16} среднее {arr.mean():9.1f} мкс   медиана {np.median(arr):9.1f}   '
          f'мин {arr.min():9.1f}   макс {arr.max():9.1f}')


def main():
    parser = argparse.ArgumentParser(description='RKNN benchmark на RK3588')
    parser.add_argument('--config', default=CONFIG_PATH)
    parser.add_argument('--repeats', type=int, default=REPEATS)
    parser.add_argument('--warmup', type=int, default=WARMUP)
    parser.add_argument('--image', default=TEST_IMAGE,
                        help='путь к тестовому изображению; без него — шум')
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error('--repeats >= 1, --warmup >= 0')
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    _, _, height, width = cfg['input_size']
    rknn_path = cfg['rknn']

    if not os.path.exists(rknn_path):
        raise FileNotFoundError(f'нет модели: {rknn_path}')

    # Вход NPU: uint8 RGB, БЕЗ деления на 255 — нормализацию по
    # mean_values/std_values выполняет сам NPU по конфигу сборки.
    if args.image:
        bgr = cv2.imread(args.image, flags=cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f'не читается: {args.image}')
        img = cv2.cvtColor(resize_pad(bgr, width, height), cv2.COLOR_BGR2RGB)
    else:
        img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    img = np.expand_dims(img, 0)           # [1, H, W, 3], NHWC

    rknn = RKNNLite()
    try:
        print(f'--> загрузка {rknn_path}; вход {width}x{height}')
        if rknn.load_rknn(rknn_path) != 0:
            raise RuntimeError('load_rknn failed')

        print('--> init_runtime (NPU)')
        if rknn.init_runtime(core_mask=CORE_MASK) != 0:
            raise RuntimeError('init_runtime failed')

        print(f'--> прогрев ({args.warmup})')
        for _ in range(args.warmup):
            rknn.inference(inputs=[img])

        import gc

        def rss_mb():
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
            return 0.0
      
      
        print("\n=== RKNN GC GEN0 TEST ===")
        
        old_thresholds = gc.get_threshold()
        old_enabled = gc.isenabled()
        
        gc.collect(2)
        gc.set_threshold(700, 1_000_000, 1_000_000)
        gc.enable()
        
        wall_times = []
        gc_events = []
        gc_starts = {}
        
        
        def gc_callback(phase, info):
            gen = info["generation"]
            now = time.perf_counter_ns()
        
            if phase == "start":
                gc_starts[gen] = now
        
            elif phase == "stop":
                started = gc_starts.pop(gen, None)
        
                if started is not None:
                    gc_events.append((
                        gen,
                        (now - started) / 1e6,
                        info["collected"]
                    ))
        
        
        try:
            baseline = rss_mb()
        
            print("GC thresholds:", gc.get_threshold())
            print(f"Начальная память: {baseline:.2f} МБ")
        
            gc.callbacks.append(gc_callback)
        
            for i in range(1, 20001):
                t0 = time.perf_counter_ns()
        
                out = rknn.inference(inputs=[img])
        
                elapsed = (time.perf_counter_ns() - t0) / 1e6
                wall_times.append(elapsed)
        
                if i % 1000 == 0:
                    current = rss_mb()
                    growth = current - baseline
        
                    print(
                        f"Вызовов: {i:6d} | "
                        f"RSS: {current:8.2f} МБ | "
                        f"Рост: {growth:+8.2f} МБ"
                    )
        
                    if growth > 128:
                        print("Остановка: рост памяти > 128 МБ")
                        break
        
            gc.callbacks.remove(gc_callback)
        
            wall = np.asarray(wall_times)
        
            print("\n=== RKNN LATENCY ===")
        
            for p in (50, 95, 99, 99.9, 100):
                print(
                    f"P{p}: "
                    f"{np.percentile(wall, p):.3f} мс"
                )
        
            for limit in (0.5, 1, 2, 5, 10):
                count = np.count_nonzero(wall > limit)
                print(f"> {limit} мс: {count}")
        
            print("\n=== GC STATISTICS ===")
        
            for gen in (0, 1, 2):
                events = [
                    e for e in gc_events
                    if e[0] == gen
                ]
        
                if not events:
                    print(f"Gen {gen}: сборок нет")
                    continue
        
                durations = np.asarray([e[1] for e in events])
        
                print(
                    f"Gen {gen}: "
                    f"сборок={len(events)} | "
                    f"среднее={durations.mean():.3f} мс | "
                    f"P99={np.percentile(durations, 99):.3f} мс | "
                    f"макс={durations.max():.3f} мс | "
                    f"собрано={sum(e[2] for e in events)}"
                )
        
            print("\n=== MEMORY SUMMARY ===")
            print(f"Начало: {baseline:.2f} МБ")
            print(f"Конец:  {rss_mb():.2f} МБ")
        
        finally:
            if gc_callback in gc.callbacks:
                gc.callbacks.remove(gc_callback)
        
            gc.set_threshold(*old_thresholds)
        
            if old_enabled:
                gc.enable()
            else:
                gc.disable()
        
        print("=== GC GEN0 TEST COMPLETE ===\n")
        print(f'--> замер ({args.repeats})')
        per_call = []
        cpu_call = []
        
        for _ in range(args.repeats):
            cpu0 = time.thread_time_ns()
            t0 = time.perf_counter_ns()
        
            out = rknn.inference(inputs=[img])
        
            t1 = time.perf_counter_ns()
            cpu1 = time.thread_time_ns()
        
            per_call.append((t1 - t0) / 1000)
            cpu_call.append((cpu1 - cpu0) / 1000)
        
        wall = np.asarray(per_call)
        cpu = np.asarray(cpu_call)
        
        print("\n=== ЗАДЕРЖКИ RKNN ===")
        
        for percentile in (50, 90, 95, 99, 99.9):
            print(
                f"P{percentile}: "
                f"{np.percentile(wall, percentile):.1f} мкс"
            )
        
        print("\n=== КОЛИЧЕСТВО ДОЛГИХ ВЫЗОВОВ ===")
        
        for threshold in (500, 1000, 2000, 5000, 10000):
            count = np.count_nonzero(wall > threshold)
            print(f"> {threshold} мкс: {count}")
        
        print("\n=== 10 САМЫХ ДОЛГИХ ВЫЗОВОВ ===")
        
        indices = np.argsort(wall)[-10:][::-1]
        
        for i in indices:
            print(
                f"Вызов {i + 1}: "
                f"wall={wall[i]:.1f} мкс "
                f"cpu={cpu[i]:.1f} мкс "
                f"разница={wall[i] - cpu[i]:.1f} мкс"
            )

        print()
        stats('inference()', per_call)
        mean_us = float(np.mean(per_call))
        print(f'{"пропускная":<16} {1e6 / mean_us:9.0f} FPS (одно ядро NPU)')
        print(f'\nвыход: {np.asarray(out[0]).ravel()}')
    finally:
        rknn.release()


if __name__ == '__main__':
    main()