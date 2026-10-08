"""
Самопроверка: поднимает сервер в фоновом потоке и опрашивает его клиентом.
Проверяет короткое чтение, фрагментацию, выход за границы, битую CRC
и чужой адрес устройства.

    python3 test_roundtrip.py
"""

import socket
import struct
import threading
import time

from dsp_protocol import (
    build_request, build_response_frames, crc16_modbus,
    parse_request, parse_response_frame, verify_crc,
)
from dsp_server import DspUdpServer, RegisterBank
from dsp_client import DspUdpClient, ResponseTimeout

HOST = "127.0.0.1"
PORT = 5502
UNIT = 1

ok_count = 0
fail_count = 0


def check(name, cond, extra=""):
    global ok_count, fail_count
    if cond:
        ok_count += 1
        print(f"  OK   {name}" + (f"  {extra}" if extra else ""))
    else:
        fail_count += 1
        print(f"  FAIL {name}" + (f"  {extra}" if extra else ""))


# ---------------------------------------------------------------- сервер

bank = RegisterBank(size=4096)
bank.fill_demo()

srv = DspUdpServer(bind_ip=HOST, port=PORT, unit=UNIT, bank=bank,
                   dsp_state=0x05, max_per_frame=250, verbose=False)
srv.start()
t = threading.Thread(target=srv.serve_forever, daemon=True)
t.start()
time.sleep(0.2)

print("=" * 70)
print("1. Проверка кадров без сети")
print("=" * 70)

req = build_request(UNIT, 0x12345678, 0x0000ABCD)
check("длина запроса 13 байт", len(req) == 13, f"len={len(req)}")
check("CRC запроса сходится", verify_crc(req))
print(f"       hex: {req.hex(' ')}")
p = parse_request(req)
check("адрес и количество разобраны",
      p["start"] == 0x12345678 and p["count"] == 0xABCD,
      f"start=0x{p['start']:X} count={p['count']}")

frames = build_response_frames(UNIT, 0x05, 0, list(range(10)), max_per_frame=250)
f0 = frames[0]
check("ответ на 10 регистров = 34 байта", len(f0) == 10 * 2 + 14, f"len={len(f0)}")
check("формула n = count*2+14 соблюдена", len(f0) == 10 * 2 + 14)
check("CRC ответа сходится", verify_crc(f0))
print(f"       hex: {f0.hex(' ')}")

print()
print("=" * 70)
print("2. Короткое чтение по сети")
print("=" * 70)

with DspUdpClient(HOST, PORT, UNIT, timeout=1.0, debug=False) as c:
    regs, dsp = c.read_registers(0, 8)
    check("получено 8 регистров", len(regs) == 8, f"len={len(regs)}")
    check("значения верные", regs == list(range(8)), f"{regs}")
    check("состояние DSP передано", dsp == 0x05, f"dsp=0x{dsp:02X}")

    regs, dsp = c.read_registers(100, 4)
    check("чтение со смещения 100", regs == [235, 0x4248, 0x0000, 0xFFFF], f"{regs}")

print()
print("=" * 70)
print("3. Фрагментация (1000 регистров по 250 на кадр)")
print("=" * 70)

with DspUdpClient(HOST, PORT, UNIT, timeout=1.0, debug=True) as c:
    t0 = time.monotonic()
    regs, dsp = c.read_registers(0, 1000)
    dt = time.monotonic() - t0
    check("собрано 1000 регистров", len(regs) == 1000, f"len={len(regs)}")
    # эталон: fill_demo кладёт i в регистр i, кроме 100..103
    expected = list(range(1000))
    expected[100:104] = [235, 0x4248, 0x0000, 0xFFFF]
    check("порядок не нарушен", regs == expected,
          "" if regs == expected else f"первое расхождение на индексе "
          f"{next(i for i, (a, b) in enumerate(zip(regs, expected)) if a != b)}")
    check("границы фрагментов склеены верно",
          regs[249:251] == [249, 250] and regs[499:501] == [499, 500],
          f"{regs[249:251]} {regs[499:501]}")
    check("уложились в таймаут", dt < 1.0, f"{dt*1000:.0f} мс")

print()
print("=" * 70)
print("4. Граничные случаи")
print("=" * 70)

# выход за границы банка (4096 регистров)
with DspUdpClient(HOST, PORT, UNIT, timeout=1.0, debug=False) as c:
    regs, dsp = c.read_registers(4090, 20)
    check("вне границ: ответ получен", len(regs) == 20, f"len={len(regs)}")
    check("вне границ: флаг в DSP поднят", dsp & 0x02, f"dsp=0x{dsp:02X}")

# битая CRC -- сервер должен молчать
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.settimeout(0.5)
bad = bytearray(build_request(UNIT, 0, 4))
bad[-1] ^= 0xFF
s.sendto(bytes(bad), (HOST, PORT))
try:
    s.recvfrom(65535)
    check("битая CRC: сервер промолчал", False, "пришёл ответ")
except socket.timeout:
    check("битая CRC: сервер промолчал", True)

# чужой адрес устройства
s.sendto(build_request(99, 0, 4), (HOST, PORT))
try:
    s.recvfrom(65535)
    check("чужой unit: сервер промолчал", False, "пришёл ответ")
except socket.timeout:
    check("чужой unit: сервер промолчал", True)

# мусор вместо кадра
s.sendto(b"\x00\x01\x02", (HOST, PORT))
try:
    s.recvfrom(65535)
    check("мусор: сервер промолчал", False, "пришёл ответ")
except socket.timeout:
    check("мусор: сервер промолчал", True)
s.close()

# таймаут клиента на мёртвом адресе
with DspUdpClient(HOST, PORT + 1, UNIT, timeout=0.2, retries=2, debug=False) as c:
    try:
        c.read_registers(0, 4)
        check("нет устройства: поднято исключение", False)
    except ResponseTimeout:
        check("нет устройства: поднято исключение", True)
    except OSError:
        check("нет устройства: поднято исключение", True, "(ICMP port unreachable)")

print()
print("=" * 70)
print("5. Отбрасывание опоздавшей датаграммы")
print("=" * 70)

with DspUdpClient(HOST, PORT, UNIT, bind_ip=HOST, timeout=1.0, debug=False) as c:
    # подкладываем в буфер ответ на совсем другой диапазон
    stale = build_response_frames(UNIT, 0x00, 9000, [777] * 4)[0]
    inj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inj.bind((HOST, 0))
    inj.sendto(stale, c.sock.getsockname())
    inj.close()
    time.sleep(0.1)
    regs, dsp = c.read_registers(0, 4)
    check("мусорный фрагмент не попал в данные", regs == [0, 1, 2, 3], f"{regs}")

srv.stop()
time.sleep(0.3)

print()
print("=" * 70)
print(f"ИТОГО: успешно {ok_count}, провалено {fail_count}")
print("=" * 70)
raise SystemExit(1 if fail_count else 0)
