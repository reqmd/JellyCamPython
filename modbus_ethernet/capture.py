"""
Виртуальная съёмка.

Приложение пишет в регистр 1 значение 2 ("снимай"), затем кладёт в
регистр 401999 метку 65278 и опрашивает его, пока там не появится
высота снятого изображения. После этого читает:

    20366, 6144 регистра    - стабильный фон, по три значения на пиксель
                              (как у графика), сейчас просто синий
    402000, H * 3072        - само изображение

Упаковка изображения: в одном регистре два подряд идущих байта потока
R,G,B,R,G,B,... Старший байт - первый, младший - следующий:

    регистр 0 = R0 * 256 + G0
    регистр 1 = B0 * 256 + R1
    регистр 2 = G1 * 256 + B1

Отсюда и число регистров: 2048 * H * 3 / 2. Для H = 512 это 1572864.
Значения байтов - uint8, 0..255.

Пока вместо камеры подставляется картинка из файла. Позже загрузка
файла в shoot() заменится кадром с линейной USB-камеры.
"""

import os

import cv2
import numpy as np
import time

from valve_config import HERE, REGISTERS

COMMAND_REG = 1          # командный регистр
COMMAND_CAPTURE = 2      # значение, запускающее съёмку
STATUS_REG = 401999      # приложение пишет сюда метку и опрашивает
BUSY_VALUE = 65278       # метка "съёмка идёт"
PHOTO_BASE = 402000      # с этого регистра идёт упакованное изображение

IMAGE_FILE = os.path.join(HERE, "cat.jpg")
WIDTH = 2048             # ширина строки в пикселях
CHANNELS = 3             # R, G, B
HEIGHT = 512
MAX_HEIGHT = 1024        # потолок высоты снимка

MIN_BYTE = 1
DELAY_SEC = 3.0           # сколько длится съёмка
_busy_since = None        # момент получения метки, пока идёт съёмка
BACKGROUND = (25, 25, 255)  # стабильный фон кадра: синий

# Упакованное изображение: numpy uint16, по два байта в регистре.
_packed = None
_height = 0


def shoot():
    """Загружает изображение. Позже здесь будет кадр с камеры."""
    global _packed, _height

    img = cv2.imread(IMAGE_FILE, cv2.IMREAD_COLOR)
    if img is None:
        print(f"  не читается {IMAGE_FILE}, съёмка не выполнена")
        return

    img = cv2.resize(img, (WIDTH, HEIGHT))
    #img = img[:MAX_HEIGHT]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = np.maximum(img, MIN_BYTE)

    flat = img.reshape(-1)                       # uint8: R,G,B,R,G,B,...
    if flat.size % 2:                            # на всякий случай
        flat = np.append(flat, np.uint8(0))
    _packed = (flat[0::2].astype(np.uint16) << 8) | flat[1::2]
    _height = img.shape[0]

    print(f"  снято: {WIDTH} x {_height}, "
          f"{_packed.size} регистров изображения")


def on_write(start, values):
    """Съёмка запускается меткой в регистре состояния."""
    global _busy_since

    if start <= COMMAND_REG < start + len(values):
        if values[COMMAND_REG - start] == COMMAND_CAPTURE:
            print(f"  команда съёмки (регистр {COMMAND_REG}), "
                  f"ждём метку {BUSY_VALUE} в регистре {STATUS_REG}")

    if start <= STATUS_REG < start + len(values):
        if values[STATUS_REG - start] == BUSY_VALUE:
            _busy_since = time.monotonic()
            print(f"  съёмка начата, готовность через {DELAY_SEC} с")


def status_value() -> int:
    """Метка, пока идёт съёмка; после задержки - высота изображения."""
    global _busy_since

    if _busy_since is not None:
        if time.monotonic() - _busy_since < DELAY_SEC:
            return BUSY_VALUE
        _busy_since = None                       # чтобы не снимать повторно
        shoot()
        if _height:
            REGISTERS[STATUS_REG] = _height
            print(f"  регистр {STATUS_REG} = {_height} (высота изображения)")

    return REGISTERS.get(STATUS_REG, 0)

def is_ready() -> bool:
    return _packed is not None


def photo_slice(start: int, count: int):
    """
    Быстрый путь: кадр целиком лежит внутри блока изображения, отдаём срез.
    None - если кадр выходит за границы блока хотя бы частично; такие
    кадры собираются поштучно через photo_value, чтобы запрос, начатый
    чуть раньше или кончающийся чуть позже, не терял данные.
    """
    if _packed is None:
        return None
    offset = start - PHOTO_BASE
    if offset < 0 or offset + count > _packed.size:
        return None
    return _packed[offset:offset + count].tolist()


def photo_value(addr: int):
    """Один регистр изображения. None, если адрес вне блока."""
    if _packed is None:
        return None
    offset = addr - PHOTO_BASE
    if 0 <= offset < _packed.size:
        return int(_packed[offset])
    return None


def background_value(offset: int):
    """Стабильный фон кадра. None, пока снимка нет - тогда живой график."""
    if _packed is None:
        return None
    return BACKGROUND[offset % CHANNELS]