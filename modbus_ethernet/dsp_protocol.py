"""
Общая часть модифицированного протокола Modbus поверх UDP.

Формат запроса на чтение (13 байт):
    [0]      адрес устройства
    [1]      резерв (0)
    [2]      код функции (3)
    [3..6]   адрес начального регистра, uint32 big-endian
    [7..10]  число регистров,           uint32 big-endian
    [11..12] CRC16

Формат ответа (count * 2 + 14 байт):
    [0]      адрес устройства
    [1]      0 - фрейм не последний, 1 - последний
    [2]      код функции (3)
    [3]      состояние DSP
    [4..7]   адрес начального регистра фрагмента, uint32 big-endian
    [8..11]  число регистров во фрагменте,        uint32 big-endian
    [12..]   значения регистров, uint16 big-endian каждый
    [n-2..]  CRC16
"""

import struct

FUNC_READ = 0x03

REQ_LEN = 13        # длина запроса целиком
RESP_HDR = 12       # заголовок ответа [0..11]
CRC_LEN = 2

MAX_REGS = 100_000  # защита от мусора в поле длины

# UDP payload при MTU 1500 = 1472 байта; минус 14 служебных => 729 регистров.
# Берём с запасом, чтобы не ловить IP-фрагментацию.
MAX_REGS_PER_FRAME = 500


class ProtocolError(Exception):
    pass


class CrcError(ProtocolError):
    pass


def crc16_modbus(data: bytes) -> int:
    """CRC-16/MODBUS: полином 0xA001, инициализация 0xFFFF."""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def pack_crc(value: int, big_endian: bool = True) -> bytes:
    return struct.pack(">H" if big_endian else "<H", value)


def append_crc(frame: bytes, big_endian: bool = True) -> bytes:
    return frame + pack_crc(crc16_modbus(frame), big_endian)


def verify_crc(frame: bytes, big_endian: bool = True) -> bool:
    if len(frame) < CRC_LEN + 1:
        return False
    body, tail = frame[:-CRC_LEN], frame[-CRC_LEN:]
    return tail == pack_crc(crc16_modbus(body), big_endian)


# --------------------------------------------------------------------------
# Запрос
# --------------------------------------------------------------------------

def build_request(unit: int, start: int, count: int,
                  crc_big_endian: bool = True) -> bytes:
    frame = bytes([unit & 0xFF, 0x00, FUNC_READ]) + struct.pack(">II", start, count)
    return append_crc(frame, crc_big_endian)


def parse_request(dgram: bytes, crc_big_endian: bool = True,
                  check_crc: bool = True) -> dict:
    if len(dgram) != REQ_LEN:
        raise ProtocolError(
            f"длина запроса {len(dgram)} вместо {REQ_LEN}: {dgram.hex(' ')}")

    if check_crc and not verify_crc(dgram, crc_big_endian):
        raise CrcError(f"CRC запроса не сошлась: {dgram.hex(' ')}")

    unit, reserved, func = dgram[0], dgram[1], dgram[2]
    start, count = struct.unpack(">II", dgram[3:11])

    if func != FUNC_READ:
        raise ProtocolError(f"код функции 0x{func:02X} не поддерживается")

    return {"unit": unit, "reserved": reserved, "func": func,
            "start": start, "count": count}


# --------------------------------------------------------------------------
# Ответ
# --------------------------------------------------------------------------

def build_response_frame(unit: int, last: bool, dsp: int,
                         start: int, regs, crc_big_endian: bool = True) -> bytes:
    """Один фрагмент ответа."""
    count = len(regs)
    frame = bytes([unit & 0xFF, 1 if last else 0, FUNC_READ, dsp & 0xFF])
    frame += struct.pack(">II", start, count)
    frame += struct.pack(f">{count}H", *regs)
    return append_crc(frame, crc_big_endian)


def build_response_frames(unit: int, dsp: int, start: int, regs,
                          max_per_frame: int = MAX_REGS_PER_FRAME,
                          crc_big_endian: bool = True):
    """Режет ответ на фрагменты, последнему ставит флаг last=1."""
    frames = []
    total = len(regs)
    offset = 0
    while True:
        chunk = regs[offset:offset + max_per_frame]
        last = offset + len(chunk) >= total
        frames.append(build_response_frame(
            unit, last, dsp, start + offset, chunk, crc_big_endian))
        offset += len(chunk)
        if last:
            return frames


def parse_response_frame(dgram: bytes, crc_big_endian: bool = True,
                         check_crc: bool = True) -> dict:
    if len(dgram) < RESP_HDR + CRC_LEN:
        raise ProtocolError(f"слишком короткая датаграмма: {len(dgram)} байт")

    unit, last, func, dsp = dgram[0], dgram[1], dgram[2], dgram[3]
    start, count = struct.unpack(">II", dgram[4:RESP_HDR])

    if func != FUNC_READ:
        raise ProtocolError(f"код функции 0x{func:02X}: {dgram.hex(' ')}")
    if count > MAX_REGS:
        raise ProtocolError(f"абсурдное число регистров {count}")

    expected = count * 2 + 14
    if len(dgram) != expected:
        raise ProtocolError(
            f"длина {len(dgram)} != ожидаемой {expected} при count={count}")

    if check_crc and not verify_crc(dgram, crc_big_endian):
        raise CrcError(f"CRC ответа не сошлась: {dgram.hex(' ')}")

    regs = list(struct.unpack(f">{count}H", dgram[RESP_HDR:-CRC_LEN]))
    return {"unit": unit, "last": bool(last), "dsp": dsp,
            "start": start, "count": count, "regs": regs}
