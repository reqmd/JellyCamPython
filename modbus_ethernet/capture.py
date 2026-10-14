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

Откуда берётся кадр, задаёт MODE:

    1 - картинка с диска (capture.png), съёмка имитируется задержкой
    2 - линейная камера: ждём, пока мимо проедет предмет, и снимаем его
"""

import os
import time

import cv2
import numpy as np

from valve_config import HERE, REGISTERS

# --------------------------------------------------------------------------
# Режим съёмки
# --------------------------------------------------------------------------

MODE_FILE = 1            # картинка с диска
MODE_LINE_CAMERA = 2     # кадр с линейной камеры

MODE = MODE_LINE_CAMERA


# Если камера не поднялась, подставлять картинку с диска, чтобы приложение
# не зависло в ожидании высоты. False - честно ждать камеру.
FALLBACK_TO_FILE = True

COMMAND_REG = 1          # командный регистр
COMMAND_CAPTURE = 2      # значение, запускающее съёмку
STATUS_REG = 401999      # приложение пишет сюда метку и опрашивает
BUSY_VALUE = 65278       # метка "съёмка идёт"
PHOTO_BASE = 402000      # с этого регистра идёт упакованное изображение

IMAGE_FILE = os.path.join(HERE, "capture.png")
WIDTH = 2048             # ширина строки в пикселях
CHANNELS = 3             # R, G, B
MAX_HEIGHT = 4096        # потолок высоты снимка

BACKGROUND = (0, 0, 255)  # стабильный фон кадра: синий
DELAY_SEC = 1.0           # сколько длится съёмка
MIN_BYTE = 1              # нижняя граница яркости, см. комментарий в shoot()

# Упакованное изображение: numpy uint16, по два байта в регистре.
_packed = None
_height = 0
_busy_since = None        # момент получения метки, пока идёт съёмка


def _frame_from_file():
    """Картинка с диска, BGR."""
    img = cv2.imread(IMAGE_FILE, cv2.IMREAD_COLOR)
    if img is None:
        print(f"  не читается {IMAGE_FILE}")
    return img


_fallback_warned = False


def _frame_from_camera():
    """Снятый объект с камеры, BGR. None, если предмет ещё не проехал."""
    global _fallback_warned
    import line_camera

    if not line_camera.ensure_started():
        if not FALLBACK_TO_FILE:
            return None
        if not _fallback_warned:
            print("  камера недоступна, работаю по картинке с диска")
            _fallback_warned = True
        return _frame_from_file()

    return line_camera.take()


def get_frame():
    """Кадр по текущему режиму. None - снимать пока нечего."""
    if MODE == MODE_LINE_CAMERA:
        return _frame_from_camera()
    return _frame_from_file()


def shoot(img=None):
    """Упаковывает кадр в регистры. True, если получилось."""
    global _packed, _height

    if img is None:
        img = get_frame()
    if img is None:
        return False

    if img.shape[1] != WIDTH:
        new_h = max(1, round(img.shape[0] * WIDTH / img.shape[1]))
        img = cv2.resize(img, (WIDTH, new_h))
    img = img[:MAX_HEIGHT]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Ни один байт не должен быть нулём, иначе регистр окажется равен нулю.
    # Приложение считает нулевой регистр непринятыми данными и до
    # бесконечности перезапрашивает такой кусок. У настоящей камеры шум
    # матрицы не даёт чистого нуля: её самый тёмный байт равен 1.
    img = np.maximum(img, MIN_BYTE)

    flat = img.reshape(-1)                       # uint8: R,G,B,R,G,B,...
    if flat.size % 2:                            # на всякий случай
        flat = np.append(flat, np.uint8(MIN_BYTE))
    _packed = (flat[0::2].astype(np.uint16) << 8) | flat[1::2]
    _height = img.shape[0]

    print(f"  снято: {WIDTH} x {_height}, "
          f"{_packed.size} регистров изображения")
    return True


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
            if MODE == MODE_LINE_CAMERA:
                import line_camera
                line_camera.arm()       # годится только предмет, снятый после
                print("  съёмка начата, жду предмет")
            else:
                print(f"  съёмка начата, готовность через {DELAY_SEC} с")


def status_value() -> int:
    """
    Что отдавать при чтении регистра состояния.

    Пока кадра нет - метка. Как только он появился, изображение
    упаковывается и в регистр кладётся его высота. Готовность проверяется
    по чтениям, без отдельного потока и без блокировки: приложение и так
    опрашивает этот регистр, а сервер продолжает отвечать на всё остальное.
    """
    global _busy_since

    if _busy_since is not None:
        if MODE == MODE_LINE_CAMERA:
            img = get_frame()                    # None, пока предмет не проехал
            if img is None:
                return BUSY_VALUE
        else:
            if time.monotonic() - _busy_since < DELAY_SEC:
                return BUSY_VALUE
            img = None                           # возьмётся в shoot()

        _busy_since = None                       # чтобы не снимать повторно
        if shoot(img):
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