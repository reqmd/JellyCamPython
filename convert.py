import os
import random

import cv2
import numpy as np
import yaml
from rknn.api import RKNN

CONFIG_PATH = 'models.yaml'
DATASET_TXT = 'dataset.txt'
CALIB_DIR = 'calib'


def resize_pad(image: np.ndarray, size: int, fill=(0, 0, 0)) -> np.ndarray:
    """То же, что ResizePad в train.py, но на numpy/BGR.

    Эту же функцию нужно применять в инференсе на устройстве — иначе
    препроцессинг разойдётся с обучением.
    """
    h, w = image.shape[:2]
    scale = size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    y0, x0 = (size - new_h) // 2, (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def make_calib_list(image_root: str,
                    config_path: str = CONFIG_PATH,
                    out_txt: str = DATASET_TXT,
                    calib_dir: str = CALIB_DIR,
                    limit: int = 400):
    """Готовит калибровочный набор для INT8.

    Картинки прогоняются через resize_pad и сохраняются уже в размере входа
    модели. Если отдать toolkit'у оригиналы, он приведёт их к 32x32 обычным
    ресайзом и исказит пропорции — статистика для квантизации соберётся
    не по тем данным, которые модель видит в работе.
    """
    with open(config_path, 'r') as f:
        size = yaml.safe_load(f)['input_size'][2]   # H из [1, 3, H, W]

    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    paths = []
    for root, _, files in os.walk(image_root):
        for name in files:
            if os.path.splitext(name)[1].lower() in exts:
                paths.append(os.path.join(root, name))

    if not paths:
        raise RuntimeError(f'не найдено изображений в {image_root}')

    random.seed(42)
    random.shuffle(paths)
    paths = paths[:limit]

    os.makedirs(calib_dir, exist_ok=True)
    out_paths = []
    for i, src in enumerate(paths):
        image = cv2.imread(src, flags=cv2.IMREAD_COLOR)
        if image is None:
            print(f'пропущено (не читается): {src}')
            continue
        dst = os.path.abspath(os.path.join(calib_dir, f'{i:04d}.png'))
        # imwrite ожидает BGR и пишет корректный по цвету файл; toolkit
        # прочитает его как RGB — ровно то, на чём обучалась модель.
        cv2.imwrite(dst, resize_pad(image, size))
        out_paths.append(dst)

    with open(out_txt, 'w') as f:
        f.write('\n'.join(out_paths) + '\n')
    print(f'{out_txt}: {len(out_paths)} изображений {size}x{size} в {calib_dir}/')
    return out_txt


def convert_to_rknn(config_path: str = CONFIG_PATH,
                    quantize: bool = True,
                    dataset: str = DATASET_TXT,
                    target: str = 'rk3588',
                    perf: bool = False) -> None:
    """Собирает .rknn из ONNX.

    perf=True измеряет время по слоям, но требует ПОДКЛЮЧЁННОЙ платы
    (init_runtime идёт на железо). На машине сборки оставляйте False.
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    onnx_path = cfg['onnx']
    rknn_path = cfg['rknn']
    mean_values = cfg['mean_values']      # соответствуют ToTensor() без Normalize
    std_values = cfg['std_values']

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f'нет ONNX: {onnx_path}')
    if quantize and not os.path.exists(dataset):
        raise FileNotFoundError(
            f'нет {dataset} — сначала вызовите make_calib_list(<train dir>)'
        )

    os.makedirs(os.path.dirname(os.path.abspath(rknn_path)), exist_ok=True)

    rknn = RKNN(verbose=True)
    try:
        rknn.config(
            mean_values=mean_values,
            std_values=std_values,
            target_platform=target,
            # В train.py картинки конвертируются BGR->RGB, модель обучена на RGB.
            # На устройстве после cv2.imread тоже нужен cvtColor в RGB.
            quant_img_RGB2BGR=False,
        )

        print('--> загрузка ONNX')
        # inputs/input_size_list не передаём: форма входа в ONNX уже статическая
        # ([1,3,H,W] из train.py), а явное указание заставляет toolkit думать,
        # что модель надо обрезать, и он предупреждает об этом.
        if rknn.load_onnx(model=onnx_path) != 0:
            raise RuntimeError('load_onnx failed')

        print(f'--> построение RKNN (quantization={quantize})')
        if rknn.build(do_quantization=quantize,
                      dataset=dataset if quantize else None) != 0:
            raise RuntimeError('build failed')

        print('--> экспорт')
        if rknn.export_rknn(rknn_path) != 0:
            raise RuntimeError('export_rknn failed')

        if quantize:
            print('--> анализ точности квантизации')
            # косинусная близость выходов слоёв fp32 vs int8;
            # для последнего слоя ждём > 0.99. Работает на симуляторе,
            # плата не нужна.
            with open(dataset, 'r') as f:
                sample = f.readline().strip()
            try:
                rknn.accuracy_analysis(inputs=[sample], target=None)
            except Exception as e:
                print(f'accuracy_analysis пропущен: {e}')

        if perf:
            # Только при подключённой плате: eval_perf требует init_runtime.
            print('--> профилирование слоёв на устройстве')
            if rknn.init_runtime(target=target, perf_debug=True) != 0:
                raise RuntimeError('init_runtime failed — плата не подключена?')
            rknn.eval_perf()

        print('готово:', rknn_path)
    finally:
        rknn.release()


if __name__ == '__main__':
    make_calib_list('linux/data/data_for_nn/train')   # раскомментировать один раз
    convert_to_rknn(quantize=True, perf=False)