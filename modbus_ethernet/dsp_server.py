"""
Сервер (эмулятор устройства) модифицированного Modbus поверх UDP.

Слушает UDP-порт, принимает запросы на чтение, отвечает значениями
регистров. Длинные ответы режет на фрагменты.

Запуск:
    python dsp_server.py --bind 192.168.2.11 --port 502 --unit 1
"""

import argparse
import socket
import struct
import threading
import time

from dsp_protocol import (
    FUNC_READ, MAX_REGS_PER_FRAME, CrcError, ProtocolError,
    build_response_frames, parse_request,
)


class RegisterBank:
    """Хранилище регистров. Потокобезопасное."""

    def __init__(self, size: int = 65536):
        self.size = size
        self._regs = [0] * size
        self._lock = threading.Lock()

    def read(self, start: int, count: int):
        with self._lock:
            if start < 0 or start + count > self.size:
                return None            # выход за границы
            return self._regs[start:start + count]

    def write(self, start: int, values):
        with self._lock:
            for i, v in enumerate(values):
                if 0 <= start + i < self.size:
                    self._regs[start + i] = v & 0xFFFF

    def fill_demo(self):
        """Узнаваемые данные, чтобы на глаз видеть сдвиги адреса."""
        with self._lock:
            for i in range(self.size):
                self._regs[i] = i & 0xFFFF
            # несколько осмысленных значений
            self._regs[100] = 235          # температура 23.5 C
            self._regs[101] = 0x4248       # float32 50.5 в формате ABCD
            self._regs[102] = 0x0000
            self._regs[103] = 0xFFFF


class DspUdpServer:
    def __init__(self, bind_ip="0.0.0.0", port=502, unit=1,
                 bank=None, dsp_state=0x00,
                 max_per_frame=MAX_REGS_PER_FRAME,
                 crc_big_endian=True, check_crc=True,
                 frame_delay=0.0, verbose=True):
        self.bind_ip = bind_ip
        self.port = port
        self.unit = unit
        self.bank = bank or RegisterBank()
        self.dsp_state = dsp_state
        self.max_per_frame = max_per_frame
        self.crc_be = crc_big_endian
        self.check_crc = check_crc
        self.frame_delay = frame_delay      # пауза между фрагментами, с
        self.verbose = verbose
        self.sock = None
        self._stop = threading.Event()

    def log(self, *a):
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    def start(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        s.bind((self.bind_ip, self.port))
        s.settimeout(0.5)
        self.sock = s
        self.log(f"слушаю {self.bind_ip}:{self.port}, адрес устройства {self.unit}")

    def stop(self):
        self._stop.set()

    def serve_forever(self):
        if self.sock is None:
            self.start()
        try:
            while not self._stop.is_set():
                try:
                    dgram, src = self.sock.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError:
                    break
                try:
                    self._handle(dgram, src)
                except Exception as e:
                    self.log(f"ошибка обработки от {src}: {e}")
        finally:
            if self.sock:
                self.sock.close()
                self.sock = None

    def _handle(self, dgram: bytes, src):
        self.log(f"RX {src[0]}:{src[1]} <- {dgram.hex(' ')}")

        try:
            req = parse_request(dgram, self.crc_be, self.check_crc)
        except CrcError as e:
            self.log(f"  {e} -- кадр отброшен молча")
            return
        except ProtocolError as e:
            self.log(f"  {e} -- кадр отброшен молча")
            return

        # чужой адрес устройства: 0 считаем широковещательным, на него не отвечаем
        if req["unit"] != self.unit:
            self.log(f"  запрос адресован устройству {req['unit']}, игнорирую")
            return

        start, count = req["start"], req["count"]
        self.log(f"  чтение: start={start} (0x{start:X}), count={count}")

        if count == 0:
            self.log("  count=0, игнорирую")
            return

        regs = self.bank.read(start, count)
        if regs is None:
            # Поведение при выходе за границы в протоколе не описано.
            # Здесь возвращаем нули и поднимаем флаг в состоянии DSP.
            self.log("  диапазон вне границ -- возвращаю нули, DSP |= 0x02")
            regs = [0] * count
            dsp = self.dsp_state | 0x02
        else:
            dsp = self.dsp_state

        frames = build_response_frames(
            self.unit, dsp, start, regs, self.max_per_frame, self.crc_be)

        for i, frame in enumerate(frames, 1):
            self.sock.sendto(frame, src)
            head = frame[:16].hex(' ')
            tail = frame[-2:].hex(' ')
            self.log(f"  TX фрагмент {i}/{len(frames)}, {len(frame)} байт: "
                     f"{head}... crc={tail}")
            if self.frame_delay and i < len(frames):
                time.sleep(self.frame_delay)


def main():
    p = argparse.ArgumentParser(description="Эмулятор устройства (Modbus UDP, модифицированный)")
    p.add_argument("--bind", default="0.0.0.0", help="локальный IP (например 192.168.2.11)")
    p.add_argument("--port", type=int, default=502)
    p.add_argument("--unit", type=int, default=1, help="адрес устройства")
    p.add_argument("--dsp", type=lambda x: int(x, 0), default=0x00, help="состояние DSP")
    p.add_argument("--max-per-frame", type=int, default=MAX_REGS_PER_FRAME)
    p.add_argument("--crc-le", action="store_true", help="CRC младшим байтом вперёд")
    p.add_argument("--no-crc-check", action="store_true", help="не проверять CRC запроса")
    p.add_argument("--frame-delay", type=float, default=0.0, help="пауза между фрагментами, с")
    args = p.parse_args()

    bank = RegisterBank()
    bank.fill_demo()

    srv = DspUdpServer(
        bind_ip=args.bind, port=args.port, unit=args.unit, bank=bank,
        dsp_state=args.dsp, max_per_frame=args.max_per_frame,
        crc_big_endian=not args.crc_le, check_crc=not args.no_crc_check,
        frame_delay=args.frame_delay,
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nостановлен")


if __name__ == "__main__":
    main()
