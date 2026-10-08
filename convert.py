from rknn.api import RKNN
import numpy as np
import cv2
import yaml
import os

def convert_in_rknn():
    with open('models.yaml', 'r') as file:
        data = yaml.safe_load(file)
        ONNX_PATH = data['onnx']
        RKNN_PATH = data['rknn']
    rknn = RKNN(verbose = True)
    rknn.config(mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform='rk3588')

    # 2. загрузка ONNX
    print('--> загрузка ONNX')
    ret = rknn.load_onnx(model=ONNX_PATH)
    if ret != 0:
        print('ошибка load_onnx'); exit(ret)

    # 3. построение модели
    #    do_quantization=True -> INT8 (быстрее на NPU, но нужен датасет для калибровки)
    #    do_quantization=False -> FP16 (проще, чуть медленнее, без калибровки)
    print('--> построение RKNN')
    ret = rknn.build(
        do_quantization=False,          # начни с False; INT8 подключишь позже
        # dataset='dataset.txt',        # для INT8: txt со списком путей к калибровочным картинкам
    )
    if ret != 0:
        print('ошибка build'); exit(ret)

    # 4. экспорт
    print('--> экспорт RKNN')
    ret = rknn.export_rknn(RKNN_PATH)
    if ret != 0:
        print('ошибка export_rknn'); exit(ret)

    print('готово:', RKNN_PATH)
    rknn.release()


if not os.path.exists('rknn'):
    os.mkdir('rknn')
convert_in_rknn()