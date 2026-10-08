"""
Сервер DSP-протокола (модифицированный Modbus поверх UDP).

Принимает запрос на чтение, отвечает значениями регистров.
Все регистры равны 0, кроме 99-го — он равен 1.

Запуск:
    python3 dsp_server.py 192.168.2.101 5001
"""

import socket
import struct
import sys

UNIT = 1          # адрес устройства
DSP_STATE = 0     # байт состояния DSP
FUNC_READ = 0x03
FUNC_WRITE = 0x10

# Значения регистров. Всё, чего нет в словаре, равно нулю.
REGISTERS = {
    99: 1,
}


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


def build_read_response(start: int, count: int) -> bytes:
    """
    [0] адрес устройства   [1] 1 = последний фрейм   [2] 3   [3] состояние DSP
    [4..7] адрес начального регистра   [8..11] число регистров
    [12..] значения регистров   [n-2..] CRC
    Длина = count * 2 + 14
    """
    values = [REGISTERS.get(start + i, 0) for i in range(count)]
    frame = bytes([UNIT, 1, FUNC_READ, DSP_STATE])
    frame += struct.pack(">II", start, count)
    frame += struct.pack(f">{count}H", *values)
    return with_crc(frame)


def build_write_response(start: int, count: int) -> bytes:
    """Подтверждение записи: та же шапка, без значений. 14 байт."""
    frame = bytes([UNIT, 1, FUNC_WRITE, DSP_STATE])
    frame += struct.pack(">II", start, count)
    return with_crc(frame)


def handle(data: bytes):
    """Возвращает кадр ответа или None, если запрос надо проигнорировать."""
    if len(data) < 13 or not crc_ok(data):
        return None
    if data[0] != UNIT:
        return None

    func = data[2]
    start, count = struct.unpack(">II", data[3:11])

    if func == FUNC_READ:
        if count == 0 or count > 700:        # 700 регистров ещё влезают в MTU
            return None
        print(f"чтение: регистр {start}, количество {count}")
        return build_read_response(start, count)

    if func == FUNC_WRITE:
        values = struct.unpack(f">{count}H", data[11:11 + count * 2])
        print(f"запись: регистр {start} = {list(values)}")
        return build_write_response(start, count)

    return None


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5001

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    print(f"слушаю {host}:{port}, адрес устройства {UNIT}")

    while True:
        data, addr = sock.recvfrom(65535)
        response = handle(data)
        if response:
            sock.sendto(response, addr)


if __name__ == "__main__":
    main()