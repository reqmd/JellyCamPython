"""
Сервер DSP-протокола (модифицированный Modbus поверх UDP).

Эмулирует оптический сортировщик: отвечает на запросы приложения
о типе камеры, поле зрения и расположении клапанов.

Запуск:
    python3 dsp_server.py 192.168.2.101 5001
"""

import math
import os
import random
import errno
import socket
import traceback
import struct
import sys
import time

import numpy as np

import capture
import valves
import sorter
from valve_config import (
    CONFIG,
    REGISTERS,                    # общий словарь регистров, второго быть не должно
    VALVE_COUNT_REG, VALVE_BASE,
    VALVE_W_COUNT_REG, VALVE_W_BASE,
    FOV_LEFT_REG, FOV_RIGHT_REG,
    CAMERA_TYPE,
    load_valves, store_valve_write,
)

UNIT = 1          # адрес устройства
DSP_STATE = 1     # байт состояния DSP, в ответах всегда 1
FUNC_READ = 0x03
FUNC_WRITE = 0x10

IP_ADDRESS = '192.168.2.101'
PORT = 5001

# --------------------------------------------------------------------------
# Параметры устройства
#
# Адреса регистров, карта REGISTERS и работа с клапанами живут в
# valve_config.py и импортируются выше. Дублировать их здесь нельзя:
# появится второй словарь, и записанное приложением не дойдёт до расчёта.
# --------------------------------------------------------------------------

# Сколько регистров класть в один ответный пакет.
# 728 - максимум, который влезает в одну датаграмму:
#   728 * 2 + 14 = 1470 байт при пределе 1472 (MTU 1500 минус заголовки IP и UDP).
# Запрос на 6144 регистра (3 канала по 2048 пикселей) даёт 8 кадров
# по 728 регистров и последний на 320: 728 * 8 + 320 = 6144.
# У всех кадров, кроме последнего, байт [1] равен 0, у последнего - 1.
MAX_REGS_PER_FRAME = 728

# Потолок на длину одного запроса чтения: защита от мусора в поле длины.
# Поднят до размера полного снимка (2048 * H * 3 / 2 регистров), чтобы
# приложение могло забрать изображение целиком одним запросом.
MAX_READ_REGS = max(30000, capture.WIDTH * capture.MAX_HEIGHT
                    * capture.CHANNELS // 2)

# Пауза между датаграммами одного ответа, в секундах.
# Снимок - это больше двух тысяч датаграмм подряд, и приёмный буфер
# приложения может не успевать их забирать: в Wireshark пакеты при этом
# видны, а до программы доходит не всё, и картинка получается чёрной.
# 0 - слать без пауз. Если видны потери, начните с 0.0002.
FRAME_DELAY = 0.0

# Запросы, на которые сервер ответил дольше, печатаются с временем - так
# в логе видно, где сервер тормозит. Ответ из многих фрагментов (снимок)
# печатается с временем всегда.
SLOW_REPLY_MS = 50
TIMED_FRAMES = 100

# Приложение шлёт каждый запрос дважды подряд. Для снимка это значит два
# потока по 2161 фрагменту: второй идёт, пока приложение уже разбирает
# первый и перезапрашивает потерянные куски, и забивает канал. Повтор того
# же самого большого чтения в течение этого времени после конца ответа
# не отправляем. 0 - отвечать на каждый.
DUP_BIG_READ_SEC = 1.0
_last_big = (None, 0.0)          # (байты запроса, когда ушёл ответ)

# Подробные чтения/записи отключены по умолчанию. true в config.json
# включает полный протокольный лог, включая частые чтения графика/статуса.
LOG_REGISTERS = bool((CONFIG.get("logging") or {}).get("log_registers", False))

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
    # Регистр состояния съёмки: метка, пока идёт съёмка, потом высота.
    if addr == capture.STATUS_REG:
        return capture.status_value()

    shot = capture.photo_value(addr)
    if shot is not None:
        return shot

    if RAW_BASE <= addr < RAW_END:
        return pixel_value(addr - RAW_BASE, shift, NOISE)
    if AVG_BASE <= addr < AVG_END:
        # После съёмки этот блок отдаёт стабильный фон кадра,
        # до съёмки - усреднённый живой график.
        back = capture.background_value(addr - AVG_BASE)
        return back if back is not None else pixel_value(addr - AVG_BASE,
                                                         shift, AVG_NOISE)
    return REGISTERS.get(addr, 0)


# --------------------------------------------------------------------------
# Контрольная сумма
# --------------------------------------------------------------------------

def _make_crc_table():
    """Таблица для CRC-16/MODBUS: полином 0xA001, по байту за шаг."""
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
        table.append(crc)
    return table


_CRC_TABLE = _make_crc_table()


_CRC_NP = np.array(_CRC_TABLE, dtype=np.uint16)

# Длинные кадры считаются через numpy, короткие - циклом: на десятке байт
# цикл быстрее, чем подготовка массивов.
CRC_NUMPY_FROM = 64

# Для каждой длины кадра: (вклад начального 0xFFFF, таблица вкладов
# байтов по позициям, номера позиций). Длин всего несколько - полный
# фрагмент 728 регистров, хвост снимка, график.
_crc_pos_cache = {}


def _crc_tables(n):
    """
    CRC линейна: CRC кадра - это XOR вкладов каждого байта на своём месте
    и вклада начального значения. Вклад байта b на позиции i - это CRC
    байта b, за которым идут n - 1 - i нулевых байтов. Считаем такие
    таблицы один раз на длину и потом получаем CRC одним XOR по массиву,
    без цикла Python по полутора тысячам байтов.
    """
    t = _crc_pos_cache.get(n)
    if t is not None:
        return t
    pos = np.empty((n, 256), dtype=np.uint16)
    cur = _CRC_NP.copy()                  # байт на последнем месте
    pos[n - 1] = cur
    for i in range(n - 2, -1, -1):        # ещё один нулевой байт после
        cur = (cur >> 8) ^ _CRC_NP[cur & 0xFF]
        pos[i] = cur
    init = 0xFFFF
    for _ in range(n):
        init = (init >> 8) ^ _CRC_TABLE[init & 0xFF]
    t = (init, pos, np.arange(n))
    if len(_crc_pos_cache) > 32:          # мусорные длины не копим
        _crc_pos_cache.clear()
    _crc_pos_cache[n] = t
    return t


def crc16(data: bytes) -> int:
    """
    CRC-16/MODBUS, в кадре идёт старшим байтом вперёд.

    На передачу снимка приходится больше трёх мегабайт. Побайтовый цикл
    Python тратил на них около секунды процессора, и всё это время
    сервер отвечал медленно, а приложение начинало перезапрашивать куски
    снимка поштучно. Длинные кадры теперь считаются через numpy.
    """
    if len(data) >= CRC_NUMPY_FROM:
        init, pos, idx = _crc_tables(len(data))
        b = np.frombuffer(data, dtype=np.uint8)
        return int(init ^ np.bitwise_xor.reduce(pos[idx, b]))
    crc = 0xFFFF
    for b in data:
        crc = (crc >> 8) ^ _CRC_TABLE[(crc ^ b) & 0xFF]
    return crc


def with_crc(frame: bytes) -> bytes:
    return frame + struct.pack(">H", crc16(frame))


def crc_ok(frame: bytes) -> bool:
    return frame[-2:] == struct.pack(">H", crc16(frame[:-2]))


# --------------------------------------------------------------------------
# Сборка ответов
# --------------------------------------------------------------------------

def graph_slice(start: int, count: int):
    """
    Кусок графика, если его даёт камера.

    Блок 106 показывает текущую строку сенсора, блок 20366 - усреднённую.
    Пока камеры нет, возвращается None, и график рисуется генератором,
    как раньше.
    """
    live, avg = capture.graph_lines()
    for base, data in ((RAW_BASE, live), (AVG_BASE, avg)):
        if data is None:
            continue
        offset = start - base
        if 0 <= offset and offset + count <= data.size:
            return data[offset:offset + count]
    return None


def build_read_frame(start: int, count: int, last: bool) -> bytes:
    """
    [0] адрес устройства   [1] 1 = последний фрейм   [2] 3   [3] состояние DSP
    [4..7] адрес начального регистра   [8..11] число регистров
    [12..] значения регистров   [n-2..] CRC
    Длина = count * 2 + 14
    """
    # Блок изображения отдаём срезом готового массива: поштучная сборка
    # полутора миллионов регистров была бы слишком медленной.
    values = capture.photo_slice(start, count)
    if values is None:
        values = graph_slice(start, count)      # строка с камеры, если есть
    if values is None:
        shift = int(time.monotonic() * DRIFT_SPEED)
        values = [register_value(start + i, shift) for i in range(count)]
    frame = bytes([UNIT, 1 if last else 0, FUNC_READ, DSP_STATE])
    frame += struct.pack(">II", start, count)
    if isinstance(values, np.ndarray):
        frame += values.astype(">u2").tobytes()
    else:
        frame += struct.pack(f">{count}H", *values)
    return with_crc(frame)


def iter_read_frames(start: int, count: int):
    """
    Кадры ответа по одному, генератором.

    Это важно для снимка: если собрать все 2161 кадр заранее, первый байт
    уходит почти через полсекунды. Приложение столько не ждёт - оно
    начинает запрашивать куски поштучно, а наши запоздалые ответы на эти
    запросы приходят уже после всего потока и сбивают ему сборку.
    Генератор отдаёт первый кадр сразу, как только он готов.
    """
    if not MAX_REGS_PER_FRAME or count <= MAX_REGS_PER_FRAME:
        yield build_read_frame(start, count, True)
        return

    sent = 0
    while sent < count:
        chunk = min(MAX_REGS_PER_FRAME, count - sent)
        last = sent + chunk >= count
        yield build_read_frame(start + sent, chunk, last)
        sent += chunk


def build_write_response(start: int, count: int) -> bytes:
    """Подтверждение записи: та же шапка, без значений. 14 байт."""
    frame = bytes([UNIT, 1, FUNC_WRITE, DSP_STATE])
    frame += struct.pack(">II", start, count)
    return with_crc(frame)


# --------------------------------------------------------------------------
# Обработка запроса
# --------------------------------------------------------------------------

def handle(data: bytes):
    """
    Возвращает кадры ответа: список для записи, генератор для чтения.
    Разбор запроса и все побочные действия выполняются сразу, а кадры
    чтения собираются по мере отправки.
    """
    if len(data) < 13 or not crc_ok(data):
        return []
    if data[0] != UNIT:
        return []

    func = data[2]
    start, count = struct.unpack(">II", data[3:11])

    if func == FUNC_READ:
        if count == 0 or count > MAX_READ_REGS:
            print(f"чтение: регистр {start}, количество {count} -- "
                  f"вне допустимого (1..{MAX_READ_REGS}), игнорирую")
            return []
        per = MAX_REGS_PER_FRAME or count
        parts = -(-count // per)          # округление вверх
        if LOG_REGISTERS:
            print(f"чтение: регистр {start}, количество {count}"
                  f"{f', ответ из {parts} фрагментов' if parts > 1 else ''}")
        return iter_read_frames(start, count)

    if func == FUNC_WRITE:
        if count == 0 or len(data) != count * 2 + 13:
            return []
        values = list(struct.unpack(f">{count}H", data[11:11 + count * 2]))

        if LOG_REGISTERS:
            limit = 10 if start == 2 else 8
            shown = values if count <= limit else values[:limit] + ["..."]
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

        # Сортировка - первой: старт у неё тот же, что у съёмки (регистр
        # 1 = 2), и решить, что именно стартовало, надо до capture.
        sorter.on_write(start, values)
        # Команды съёмки и служебные регистры.
        capture.on_write(start, values)
        # Проверка механики: регистр 70 - номер клапана, открыть его по SPI.
        valves.on_write(start, values)

        return [build_write_response(start, count)]

    return []


def open_socket():
    """
    UDP-сокет сервера.

    Если сервер запущен systemd по первому пришедшему пакету (dsp.socket),
    сокет уже открыт и лежит в дескрипторе 3 - берём его, пакет, который
    разбудил сервер, в нём и ждёт. Иначе открываем сами.

    При загрузке системы сервер может стартовать раньше, чем сетевой карте
    назначен адрес 192.168.2.101, и bind упадёт. Тогда ждём и повторяем,
    а не завершаемся.
    """
    if (os.environ.get("LISTEN_PID") == str(os.getpid())
            and int(os.environ.get("LISTEN_FDS", "0")) >= 1):
        sock = socket.socket(fileno=3)
        print(f"сокет получен от systemd: {sock.getsockname()}")
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        waited = 0
        while True:
            try:
                sock.bind((IP_ADDRESS, PORT))
                break
            except OSError as e:
                if e.errno != errno.EADDRNOTAVAIL:
                    raise
                if waited % 30 == 0:
                    print(f"адреса {IP_ADDRESS} пока нет на сетевой карте, "
                          f"жду ({waited} с)")
                time.sleep(1)
                waited += 1
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 22)
    return sock


def main():
    import logger
    logger.start()          # всё напечатанное - ещё и в logs/<дата>.log

    load_valves()          # если есть сохранённая конфигурация - поднимаем её
    capture.prepare()      # камера поднимается в фоне, до первого запроса
    valves.prepare()       # плата клапанов по SPI; без неё сервер работает
    sorter.prepare()       # модель грузится в фоне, до первого старта

    sock = open_socket()
    host, port = sock.getsockname()[:2]
    print(f"слушаю {host}:{port}, адрес устройства {UNIT}")
    print(f"тип камеры {CAMERA_TYPE}, "
          f"поле зрения {REGISTERS[FOV_LEFT_REG]}..{REGISTERS[FOV_RIGHT_REG]}, "
          f"клапанов {REGISTERS[VALVE_COUNT_REG]}")

    # Python отдаёт процессор между потоками раз в 5 мс. Пока потоки
    # камеры и сортировки считают, главный поток после каждой отправки
    # датаграммы мог ждать своей очереди до 5 мс - на снимке из 2161
    # фрагмента это складывается в секунды. 1 мс - сервер отвечает быстрее,
    # камере это почти ничего не стоит.
    sys.setswitchinterval(0.001)

    while True:
        data, addr = sock.recvfrom(65535)
        t_recv = time.perf_counter()
        if _skip_duplicate(data, t_recv):
            continue
        # Одна неудачная обработка не должна останавливать сервер. Раньше
        # любое исключение - в разборе, в сборке ответа или в отправке -
        # завершало программу, и сервер переставал отвечать совсем.
        # Теперь ошибка печатается с трассировкой (и попадает в лог), а
        # сервер ждёт следующий запрос.
        try:
            first = True
            frames = 0
            t_first = None
            for frame in handle(data):
                if FRAME_DELAY and not first:
                    time.sleep(FRAME_DELAY)
                send(sock, frame, addr)
                if first:
                    t_first = time.perf_counter()
                first = False
                frames += 1
            _report_time(data, frames, t_recv, t_first)
        except Exception:
            print(f"ОШИБКА при обработке запроса от {addr[0]}:{addr[1]}, "
                  f"{len(data)} байт: {data[:16].hex()}")
            traceback.print_exc()


def _is_big_read(data):
    if len(data) < 13 or data[2] != FUNC_READ:
        return False
    count = struct.unpack(">I", data[7:11])[0]
    return count > MAX_REGS_PER_FRAME * TIMED_FRAMES


def _skip_duplicate(data, now):
    """True - это повтор большого чтения, на которое только что ответили."""
    if not DUP_BIG_READ_SEC or not _is_big_read(data):
        return False
    prev, when = _last_big
    if data == prev and now - when < DUP_BIG_READ_SEC:
        start = struct.unpack(">I", data[3:7])[0]
        if LOG_REGISTERS:
            print(f"чтение: регистр {start} - повтор того же запроса через "
                  f"{(now - when) * 1000:.0f} мс после ответа, второй раз не шлю")
        return True
    return False


def _report_time(data, frames, t_recv, t_first):
    """Сколько занял ответ, если он долгий или большой."""
    global _last_big
    now = time.perf_counter()
    total = (now - t_recv) * 1000
    if _is_big_read(data):
        _last_big = (data, now)
    if not LOG_REGISTERS:
        return
    if frames < TIMED_FRAMES and total < SLOW_REPLY_MS:
        return
    start = struct.unpack(">I", data[3:7])[0] if len(data) >= 7 else -1
    first = (t_first - t_recv) * 1000 if t_first else total
    print(f"  ответ на регистр {start}: {frames} фрагм. за {total:.0f} мс, "
          f"первый через {first:.1f} мс"
          f"{'  <- МЕДЛЕННО' if frames < TIMED_FRAMES else ''}")


def send(sock, frame, addr):
    """
    Отправка с повтором. При длинной очереди датаграмм - снимок это 2161
    кадр подряд - Linux может ответить "нет места в буфере" (ENOBUFS).
    Это временно: подождать и повторить, а не терять кадр и не падать.
    """
    for attempt in range(50):
        try:
            sock.sendto(frame, addr)
            return
        except (BlockingIOError, InterruptedError):
            time.sleep(0.001)
        except OSError as e:
            if e.errno in (errno.ENOBUFS, errno.EAGAIN):
                time.sleep(0.001 * (attempt + 1))
                continue
            raise
    print(f"кадр не ушёл после 50 попыток ({len(frame)} байт)")


if __name__ == "__main__":
    main()