"""
Сервер DSP-протокола (модифицированный Modbus поверх UDP).

Эмулирует оптический сортировщик: отвечает на запросы приложения
о типе камеры, поле зрения и расположении клапанов.

Запуск:
    python3 dsp_server.py 192.168.2.101 5001
"""

import json
import math
import os
import random
import socket
import struct
import time

from valve_config import compute_fov, clamp_u16, save_valves, load_valves, apply_valves, store_valve_write

UNIT = 1          # адрес устройства
DSP_STATE = 1     # байт состояния DSP, в ответах всегда 1
FUNC_READ = 0x03
FUNC_WRITE = 0x10

IP_ADDRESS = '192.168.2.101'
PORT = 5001

# --------------------------------------------------------------------------
# Параметры устройства
# --------------------------------------------------------------------------

CAMERA_TYPE = 10        # регистр 41006
FOV_LEFT_REG = 20106    # поле зрения, левая граница
FOV_RIGHT_REG = 20107   # поле зрения, правая граница
VALVE_COUNT_REG = 20108 # число клапанов
VALVE_BASE = 20109      # с этого регистра идут позиции клапанов, по одному на клапан
VALVE_MAX = 128         # потолок числа клапанов
VALVE_W_COUNT_REG = 20237
VALVE_W_BASE = 20238
FOV_LEFT = 1            # начальное поле зрения, пока приложение не прислало клапаны
FOV_RIGHT = 2047
VALVE_COUNT = 64        # начальное число клапанов
FOV_MARGIN = 10         # запас, который формула добавляет за крайними клапанами

# Куда сохраняется конфигурация клапанов, присланная приложением.
# Файл читается при запуске, поэтому настройки переживают перезапуск сервера.
VALVE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "valves.json")

# Сколько регистров класть в один ответный пакет.
# 728 - максимум, который влезает в одну датаграмму:
#   728 * 2 + 14 = 1470 байт при пределе 1472 (MTU 1500 минус заголовки IP и UDP).
# Запрос на 6144 регистра (3 канала по 2048 пикселей) даёт 8 кадров
# по 728 регистров и последний на 320: 728 * 8 + 320 = 6144.
# У всех кадров, кроме последнего, байт [1] равен 0, у последнего - 1.
MAX_REGS_PER_FRAME = 728

# --------------------------------------------------------------------------
# Карта регистров. Всё, чего нет в словаре, равно нулю.
# --------------------------------------------------------------------------

REGISTERS = {
    99: 1,
    41006: CAMERA_TYPE,
    FOV_LEFT_REG: FOV_LEFT,
    FOV_RIGHT_REG: FOV_RIGHT,
    VALVE_COUNT_REG: VALVE_COUNT,
    VALVE_W_COUNT_REG: VALVE_COUNT, 
}

# Клапаны расставлены равномерно по полю зрения, каждый в центре своей полосы.
# Слоты с VALVE_BASE + VALVE_COUNT по VALVE_BASE + 127 остаются нулевыми.
_step = (FOV_RIGHT - FOV_LEFT) / VALVE_COUNT
for _i in range(VALVE_COUNT):
    _pos = round(FOV_LEFT + (_i + 0.5) * _step)
    REGISTERS[VALVE_BASE + _i] = _pos
    REGISTERS[VALVE_W_BASE + _i] = _pos


# --------------------------------------------------------------------------
# Данные камеры: 2048 пикселей, по три канала на пиксель, вперемежку.
# Регистры идут так:
#   база + 0 = R пикселя 0, база + 1 = G пикселя 0, база + 2 = B пикселя 0,
#   база + 3 = R пикселя 1, база + 4 = G пикселя 1, ... и так далее
# Всего 2048 * 3 = 6144 регистра. Яркость каждого канала - uint8, 0 .. 255.
#
# Блоков два, график у них одинаковый:
#   106   .. 6249   сырые данные, с полным шумом
#   20366 .. 26509  усреднённые данные, шум подавлен
#
# Значения генерируются заново на каждый запрос, поэтому график живой.
# --------------------------------------------------------------------------

RAW_BASE = 106              # сырые данные
AVG_BASE = 20366            # усреднённые данные
PIXEL_COUNT = 2048          # число пикселей в строке
CHANNELS = 3                # R, G, B

PIXEL_MAX = 16384             # потолок яркости, uint8
CHANNEL_LEVEL = (170 * 64, 125 * 64, 80 * 64)   # средняя яркость каналов R, G, B
NOISE = 50                  # размах шума вокруг среднего, +/- отсчётов
WAVE_AMPLITUDE = 100         # размах плавной волны вдоль строки
WAVE_PERIOD = 128           # длина волны в пикселях
DRIFT_SPEED = 150           # скорость сдвига волны, пикселей в секунду

# По скольким кадрам усредняется второй график. Шум падает как корень
# из числа кадров: при окне 16 размах уменьшается вчетверо.
AVERAGE_WINDOW = 16
AVG_NOISE = max(0, round(NOISE / math.sqrt(AVERAGE_WINDOW)))

# Форма волны считается один раз при запуске, дальше только сдвигается.
_WAVE = [round(WAVE_AMPLITUDE * math.sin(2 * math.pi * _i / WAVE_PERIOD))
         for _i in range(PIXEL_COUNT)]

BLOCK_SIZE = PIXEL_COUNT * CHANNELS
RAW_END = RAW_BASE + BLOCK_SIZE
AVG_END = AVG_BASE + BLOCK_SIZE


def pixel_value(offset: int, shift: int, noise: int) -> int:
    """Яркость одного канала одного пикселя."""
    pixel, channel = divmod(offset, CHANNELS)
    value = CHANNEL_LEVEL[channel] + _WAVE[(pixel + shift) % PIXEL_COUNT]
    if noise:
        value += random.randint(-noise, noise)
    return 0 if value < 0 else (PIXEL_MAX if value > PIXEL_MAX else value)


def register_value(addr: int, shift: int) -> int:
    """Значение регистра: данные камеры генерируются, остальное из карты."""
    if RAW_BASE <= addr < RAW_END:
        return pixel_value(addr - RAW_BASE, shift, NOISE)
    if AVG_BASE <= addr < AVG_END:
        return pixel_value(addr - AVG_BASE, shift, AVG_NOISE)
    return REGISTERS.get(addr, 0)


# --------------------------------------------------------------------------
# Контрольная сумма
# --------------------------------------------------------------------------

def crc16(data: bytes) -> int:
    """CRC-16/MODBUS, в кадре идёт старшим байтом вперёд."""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def with_crc(frame: bytes) -> bytes:
    return frame + struct.pack(">H", crc16(frame))


def crc_ok(frame: bytes) -> bool:
    return frame[-2:] == struct.pack(">H", crc16(frame[:-2]))


# --------------------------------------------------------------------------
# Сборка ответов
# --------------------------------------------------------------------------

def build_read_frame(start: int, count: int, last: bool) -> bytes:
    """
    [0] адрес устройства   [1] 1 = последний фрейм   [2] 3   [3] состояние DSP
    [4..7] адрес начального регистра   [8..11] число регистров
    [12..] значения регистров   [n-2..] CRC
    Длина = count * 2 + 14
    """
    shift = int(time.monotonic() * DRIFT_SPEED)
    values = [register_value(start + i, shift) for i in range(count)]
    frame = bytes([UNIT, 1 if last else 0, FUNC_READ, DSP_STATE])
    frame += struct.pack(">II", start, count)
    frame += struct.pack(f">{count}H", *values)
    return with_crc(frame)


def build_read_response(start: int, count: int) -> list:
    """Один кадр или несколько, если включено деление."""
    if not MAX_REGS_PER_FRAME or count <= MAX_REGS_PER_FRAME:
        return [build_read_frame(start, count, True)]

    frames = []
    sent = 0
    while sent < count:
        chunk = min(MAX_REGS_PER_FRAME, count - sent)
        last = sent + chunk >= count
        frames.append(build_read_frame(start + sent, chunk, last))
        sent += chunk
    return frames


def build_write_response(start: int, count: int) -> bytes:
    """Подтверждение записи: та же шапка, без значений. 14 байт."""
    frame = bytes([UNIT, 1, FUNC_WRITE, DSP_STATE])
    frame += struct.pack(">II", start, count)
    return with_crc(frame)


# --------------------------------------------------------------------------
# Обработка запроса
# --------------------------------------------------------------------------

def handle(data: bytes) -> list:
    """Возвращает список кадров ответа (может быть пустым)."""
    if len(data) < 13 or not crc_ok(data):
        return []
    if data[0] != UNIT:
        return []

    func = data[2]
    start, count = struct.unpack(">II", data[3:11])

    if func == FUNC_READ:
        if count == 0 or count > 30000:
            return []
        frames = build_read_response(start, count)
        print(f"чтение: регистр {start}, количество {count}"
              f"{f', ответ из {len(frames)} фрагментов' if len(frames) > 1 else ''}")
        return frames

    if func == FUNC_WRITE:
        if count == 0 or len(data) != count * 2 + 13:
            return []
        values = list(struct.unpack(f">{count}H", data[11:11 + count * 2]))

        shown = values if count <= 8 else values[:8] + ["..."]
        print(f"запись: регистр {start}, количество {count}, значения {shown}")

        # Записанное попадает в карту регистров, чтобы его можно было прочитать.
        for i, v in enumerate(values):
            REGISTERS[start + i] = v

        # Если запись затронула регистр с числом клапанов - разбираем блок,
        # пересчитываем поле зрения и сохраняем конфигурацию на диск.
        for count_reg, base in ((VALVE_W_COUNT_REG, VALVE_W_BASE), (VALVE_COUNT_REG, VALVE_BASE)):
            if start <= count_reg < start + count:
                store_valve_write(count_reg, base)
                break

        return [build_write_response(start, count)]

    return []


def main():
    load_valves()          # если есть сохранённая конфигурация - поднимаем её

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
    sock.bind((IP_ADDRESS, PORT))
    print(f"слушаю {IP_ADDRESS}:{PORT}, адрес устройства {UNIT}")
    print(f"тип камеры {CAMERA_TYPE}, "
          f"поле зрения {REGISTERS[FOV_LEFT_REG]}..{REGISTERS[FOV_RIGHT_REG]}, "
          f"клапанов {REGISTERS[VALVE_COUNT_REG]}")

    while True:
        data, addr = sock.recvfrom(65535)
        for frame in handle(data):
            sock.sendto(frame, addr)


if __name__ == "__main__":
    main()