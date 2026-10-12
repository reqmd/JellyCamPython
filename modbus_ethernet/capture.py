"""
Виртуальная съёмка.

Приложение пишет в регистр 1 значение 2 ("снимай"), затем кладёт в
регистр 401999 метку 65278 и опрашивает его, пока там не появится
высота снятого изображения.

Пока вместо камеры подставляется картинка из файла. Позже загрузка
файла в shoot() заменится кадром с линейной USB-камеры.
"""

import os
import cv2

from valve_config import HERE, REGISTERS

COMMAND_REG = 1          # командный регистр
COMMAND_CAPTURE = 2      # значение, запускающее съёмку
STATUS_REG = 401999      # приложение пишет сюда метку и опрашивает
BUSY_VALUE = 65278       # метка "съёмка идёт"

IMAGE_FILE = os.path.join(HERE, "cat.jpg")
WIDTH = 2048             # ширина строки в пикселях
CHANNELS = 3             # R, G, B
MAX_HEIGHT = 4096        # потолок высоты снимка

# Плоский список: R, G, B каждого пикселя, строки идут подряд.
_image = None
_height = 0


def shoot():
    """Загружает изображение. Позже здесь будет кадр с камеры."""
    global _image, _height

    img = cv2.imread(IMAGE_FILE, cv2.IMREAD_COLOR)
    if img is None:
        print(f"  не читается {IMAGE_FILE}, съёмка не выполнена")
        return

    if img.shape[1] != WIDTH:
        new_h = max(1, round(img.shape[0] * WIDTH / img.shape[1]))
        img = cv2.resize(img, (WIDTH, new_h))
    img = img[:MAX_HEIGHT]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    _image = img.reshape(-1).tolist()
    _height = img.shape[0]
    print(f"  снято: {WIDTH} x {_height}")


def on_write(start, values):
    """Отслеживает команду съёмки и метку в регистре состояния."""
    if start <= COMMAND_REG < start + len(values):
        if values[COMMAND_REG - start] == COMMAND_CAPTURE:
            shoot()

    # Метка приходит после команды и затирает собой прежнее значение,
    # поэтому высоту выставляем именно здесь.
    if start <= STATUS_REG < start + len(values):
        if values[STATUS_REG - start] == BUSY_VALUE and _height:
            REGISTERS[STATUS_REG] = _height
            print(f"  регистр {STATUS_REG} = {_height} (высота изображения)")


def is_ready() -> bool:
    return _image is not None


def image_value(offset: int):
    """
    Значение снимка по смещению от начала блока данных.
    None, если снимка нет или смещение за его пределами.
    """
    if _image is not None and 0 <= offset < len(_image):
        return _image[offset]
    return None