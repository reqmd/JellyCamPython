#!/usr/bin/env python3
"""
Выделение объектов с синего фона.

Настройки — в блоке НАСТРОЙКИ ниже. Запуск: python extract_objects.py
Результат: objects/coffee/coffee_001.png, coffee_002.png, ...
(имя папки = часть имени файла до первого "_")
"""

import os
import cv2
import numpy as np

# ---------------------- НАСТРОЙКИ ----------------------
INPUT_DIR = r"C:/Users/Куликов/Desktop/coffee and soya"  # папка с изображениями
OUT_ROOT = "objects"        # куда складывать вырезанные объекты
EXTENSIONS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

MIN_AREA = 50               # минимальная площадь объекта, px
PADDING = 2                 # отступ вокруг объекта, px
TRANSPARENT = True          # True — PNG с прозрачным фоном, False — чёрный фон
SAVE_MASK = False           # сохранять ли маску для отладки

HUE_MIN, HUE_MAX = 90, 135  # диапазон синего (H в OpenCV: 0..179)
SAT_MIN = 60                # ниже — считается серым, не фоном
VAL_MIN = 40                # темнее — считается тёмным краем кадра (тоже фон)
# -------------------------------------------------------


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread не умеет пути с кириллицей на Windows — читаем через numpy."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, img):
    """Аналогично для записи."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def folder_name_from_path(path):
    """coffee_data.bmp -> coffee ; nuts.png -> nuts"""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("_")[0],  stem.split("_")[2]


def blue_background_mask(bgr):
    """Возвращает маску ОБЪЕКТОВ (255 = объект, 0 = фон)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([HUE_MIN, SAT_MIN, VAL_MIN], dtype=np.uint8)
    upper = np.array([HUE_MAX, 255, 255], dtype=np.uint8)
    background = cv2.inRange(hsv, lower, upper)

    # тёмные полосы по краям кадра тоже считаем фоном
    dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, VAL_MIN]))
    background = cv2.bitwise_or(background, dark)

    objects = cv2.bitwise_not(background)

    # убираем шум и затыкаем дырки внутри объектов
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    objects = cv2.morphologyEx(objects, cv2.MORPH_OPEN, k, iterations=1)
    objects = cv2.morphologyEx(objects, cv2.MORPH_CLOSE, k, iterations=2)
    return objects


def extract(path):
    bgr = imread_unicode(path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"[!] не удалось прочитать {path}")
        return 0

    mask = blue_background_mask(bgr)

    base, data = folder_name_from_path(path)
    out_dir = os.path.join(OUT_ROOT, base)
    os.makedirs(out_dir, exist_ok=True)

    if SAVE_MASK:
        imwrite_unicode(os.path.join(out_dir, "_mask.png"), mask)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    h, w = mask.shape
    saved = 0
    for i in range(1, n):  # 0 — фон
        x, y, bw, bh, area = stats[i]
        if area < MIN_AREA:
            continue

        x0, y0 = max(0, x - PADDING), max(0, y - PADDING)
        x1, y1 = min(w, x + bw + PADDING), min(h, y + bh + PADDING)

        crop = bgr[y0:y1, x0:x1]
        crop_mask = (labels[y0:y1, x0:x1] == i).astype(np.uint8) * 255

        if TRANSPARENT:
            b, g, r = cv2.split(crop)
            out = cv2.merge([b, g, r, crop_mask])
        else:
            out = cv2.bitwise_and(crop, crop, mask=crop_mask)

        saved += 1
        imwrite_unicode(os.path.join(out_dir, f"{base}_{data}_{saved:03d}.png"), out)

    print(f"{os.path.basename(path)}: найдено объектов — {saved} -> {out_dir}/")
    return saved


def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"Папка не найдена: {INPUT_DIR}")
        return

    files = sorted(
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(EXTENSIONS)
    )

    if not files:
        print(f"В папке {INPUT_DIR} нет подходящих изображений")
        return

    total = sum(extract(p) for p in files)
    print(f"Всего сохранено: {total}")


if __name__ == "__main__":
    main()