"""
Сервер DSP-протокола (модифицированный Modbus поверх UDP).

Эмулирует оптический сортировщик: отвечает на запросы приложения
о типе камеры, поле зрения и расположении клапанов.

Запуск:
    python3 dsp_server.py 192.168.2.101 5001
"""

import socket
import struct
import sys

UNIT = 1          # адрес устройства
DSP_STATE = 1     # байт состояния DSP, в ответах всегда 1
FUNC_READ = 0x03
FUNC_WRITE = 0x10

# --------------------------------------------------------------------------
# Параметры устройства
# --------------------------------------------------------------------------

CAMERA_TYPE = 10        # регистр 41006
FOV_LEFT = 1            # регистр 20106, левая граница поля зрения
FOV_RIGHT = 2047        # регистр 20107, правая граница
VALVE_COUNT = 64        # регистр 20108, число клапанов (потолок 128)
VALVE_BASE = 20109      # с этого регистра идут позиции клапанов, по одному на клапан

# Сколько регистров класть в один ответный пакет.
# 0 - весь ответ одним пакетом. 1024 - запрос на 2048 регистров
# делится на две датаграммы: у первой байт [1] = 0, у второй = 1.
MAX_REGS_PER_FRAME = 1024

# --------------------------------------------------------------------------
# Карта регистров. Всё, чего нет в словаре, равно нулю.
# --------------------------------------------------------------------------

REGISTERS = {
    99: 1,                  # связь с устройством установлена
    41006: CAMERA_TYPE,     # тип камеры
    20106: FOV_LEFT,        # поле зрения, левая граница
    20107: FOV_RIGHT,       # поле зрения, правая граница
    20108: VALVE_COUNT,     # число клапанов
}

# Клапаны расставлены равномерно по полю зрения, каждый в центре своей полосы.
# Слоты с VALVE_BASE + VALVE_COUNT по VALVE_BASE + 127 остаются нулевыми.
_step = (FOV_RIGHT - FOV_LEFT) / VALVE_COUNT
for _i in range(VALVE_COUNT):
    REGISTERS[VALVE_BASE + _i] = round(FOV_LEFT + (_i + 0.5) * _step)


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
    values = [REGISTERS.get(start + i, 0) for i in range(count)]
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
        values = struct.unpack(f">{count}H", data[11:11 + count * 2])
        print(f"запись: регистр {start} = {list(values)}")
        return [build_write_response(start, count)]

    return []


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5001

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
    sock.bind((host, port))
    print(f"слушаю {host}:{port}, адрес устройства {UNIT}")
    print(f"тип камеры {CAMERA_TYPE}, поле зрения {FOV_LEFT}..{FOV_RIGHT}, "
          f"клапанов {VALVE_COUNT}")

    while True:
        data, addr = sock.recvfrom(65535)
        for frame in handle(data):
            sock.sendto(frame, addr)


if __name__ == "__main__":
    main()