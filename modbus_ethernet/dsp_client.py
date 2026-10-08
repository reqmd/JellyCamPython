"""
Клиент (опросчик) модифицированного Modbus поверх UDP.

Привязывается к конкретному локальному IP, что гарантирует отправку
через нужный сетевой адаптер независимо от таблицы маршрутов.

Запуск:
    python dsp_client.py --host 192.168.2.11 --bind 192.168.2.100 --start 0 --count 10
"""

import argparse
import socket
import time

from dsp_protocol import (
    CrcError, ProtocolError, build_request, parse_response_frame,
)


class ResponseTimeout(ProtocolError):
    pass


class DspUdpClient:
    def __init__(self, host, port=502, unit=1,
                 bind_ip=None, bind_port=0,
                 timeout=0.5, retries=3,
                 crc_big_endian=True, check_crc=True, debug=False):
        self.peer = (host, port)
        self.unit = unit
        self.bind_ip = bind_ip
        self.bind_port = bind_port
        self.timeout = timeout          # межфрагментный таймаут
        self.retries = retries
        self.crc_be = crc_big_endian
        self.check_crc = check_crc
        self.debug = debug
        self.sock = None

    def log(self, *a):
        if self.debug:
            print(*a, flush=True)

    # --- сокет -------------------------------------------------------

    def open(self):
        self.close()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        if self.bind_ip or self.bind_port:
            s.bind((self.bind_ip or "0.0.0.0", self.bind_port))
            self.log(f"сокет привязан к {s.getsockname()}")
        self.sock = s

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    def _drain(self):
        """Выбрасывает опоздавшие датаграммы от прошлых запросов."""
        self.sock.setblocking(False)
        try:
            while True:
                try:
                    stale, _ = self.sock.recvfrom(65535)
                    self.log(f"  drop stale: {stale[:12].hex(' ')}...")
                except BlockingIOError:
                    break
                except OSError:
                    break
        finally:
            self.sock.setblocking(True)

    # --- одна попытка ------------------------------------------------

    def _attempt(self, start, count):
        chunks = {}
        got_last = False
        dsp = None
        deadline = time.monotonic() + self.timeout

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, None

            self.sock.settimeout(remaining)
            try:
                dgram, src = self.sock.recvfrom(65535)
            except socket.timeout:
                return None, None

            if src[0] != self.peer[0]:
                continue
            try:
                f = parse_response_frame(dgram, self.crc_be, self.check_crc)
            except CrcError as e:
                self.log(f"  {e}")
                continue
            if f["unit"] != self.unit:
                continue
            if not (start <= f["start"] < start + count):
                self.log(f"  фрагмент start={f['start']} вне диапазона, отброшен")
                continue

            chunks[f["start"]] = f["regs"]
            dsp = f["dsp"]
            got_last = got_last or f["last"]
            deadline = time.monotonic() + self.timeout
            self.log(f"  RX фрагмент start={f['start']} count={f['count']} "
                     f"last={int(f['last'])} dsp=0x{f['dsp']:02X}")

            total = sum(len(v) for v in chunks.values())
            if got_last and total >= count:
                break
            if total > count:
                raise ProtocolError("получено больше регистров, чем запрошено")

        regs, cursor = [], start
        for s in sorted(chunks):
            if s != cursor:
                raise ProtocolError(
                    f"разрыв в данных: ожидался адрес {cursor}, пришёл {s}")
            regs.extend(chunks[s])
            cursor += len(chunks[s])

        return regs, dsp

    # --- публичный API -----------------------------------------------

    def read_registers(self, start: int, count: int):
        if self.sock is None:
            self.open()
        req = build_request(self.unit, start, count, self.crc_be)

        last_err = None
        for attempt in range(1, self.retries + 1):
            self._drain()
            self.log(f"TX (попытка {attempt}): {req.hex(' ')}")
            self.sock.sendto(req, self.peer)
            try:
                regs, dsp = self._attempt(start, count)
            except ProtocolError as e:
                last_err = e
                self.log(f"  {e}")
                continue
            if regs is not None:
                return regs, dsp

        raise ResponseTimeout(
            f"нет полного ответа на [{start}, {count}] за {self.retries} попыток"
            + (f"; последняя ошибка: {last_err}" if last_err else ""))


def main():
    p = argparse.ArgumentParser(description="Опросчик устройства (Modbus UDP, модифицированный)")
    p.add_argument("--host", required=True, help="IP устройства, например 192.168.2.11")
    p.add_argument("--port", type=int, default=502)
    p.add_argument("--bind", default=None, help="локальный IP, например 192.168.2.100")
    p.add_argument("--unit", type=int, default=1)
    p.add_argument("--start", type=lambda x: int(x, 0), default=0)
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--timeout", type=float, default=0.5)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--crc-le", action="store_true")
    p.add_argument("--no-crc-check", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    with DspUdpClient(args.host, args.port, args.unit,
                      bind_ip=args.bind, timeout=args.timeout,
                      retries=args.retries,
                      crc_big_endian=not args.crc_le,
                      check_crc=not args.no_crc_check,
                      debug=not args.quiet) as c:
        regs, dsp = c.read_registers(args.start, args.count)
        print(f"\nDSP = 0x{dsp:02X}, получено {len(regs)} регистров")
        for i in range(0, len(regs), 8):
            addr = args.start + i
            row = " ".join(f"{v:5d}" for v in regs[i:i + 8])
            print(f"  [{addr:6d}] {row}")


if __name__ == "__main__":
    main()
