#!/usr/bin/env python3
"""
Управление платой клапанов через SPI (Orange Pi 5 Plus, /dev/spidev0.0).

Формат кадра — 10 байт, CS удерживается низким на всю посылку:

    [Duration][Data7][Data6][Data5][Data4][Data3][Data2][Data1][Data0][0x99]

    Duration  — длительность работы клапанов, 0..31
    Data[7]   — клапаны 63..56   (старший байт идёт первым)
    ...
    Data[0]   — клапаны 7..0
    0x99      — контрольный байт (153)

Клапаны нумеруются с нуля: 0..63.
"""

import sys
import time

try:
    import spidev
except ImportError:
    sys.exit("Нет модуля spidev. Установите: sudo apt install python3-spidev")


BUS = 0
DEVICE = 0
SPEED = 20_000_000     # 20 МГц, рекомендованное значение (допустимо 1..35 МГц)
MODE = 1               # CPOL=0, CPHA=1. Если не работает — попробуйте 3
CHECKSUM = 0x99        # 153
FRAME_LEN = 10
MAX_DURATION = 10
VALVE_COUNT = 64


class ValveBoard:
    def __init__(self, bus=BUS, device=DEVICE, speed=SPEED, mode=MODE,
                 verbose=False):
        self.verbose = verbose
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = speed
        self.spi.mode = mode
        self.spi.bits_per_word = 8
        self.spi.lsbfirst = False          # MSB first
        # CS опускается один раз на весь кадр и поднимается в конце
        self.spi.cshigh = False

    # ---------------------------------------------------------- сборка кадра

    @staticmethod
    def build_frame(valves, duration):
        """
        valves   — что открыть: набор номеров 0..63, либо целое 64-битное число
        duration — 0..31
        Возвращает список из 10 байт.
        """
        if not 0 <= duration <= MAX_DURATION:
            raise ValueError(f"duration вне диапазона 0..{MAX_DURATION}: {duration}")

        if isinstance(valves, int):
            mask = valves
        else:
            mask = 0
            for v in valves:
                if not 0 <= v < VALVE_COUNT:
                    raise ValueError(f"клапан вне диапазона 0..63: {v}")
                mask |= 1 << v

        if not 0 <= mask < (1 << VALVE_COUNT):
            raise ValueError("маска не помещается в 64 бита")

        # Data[7] первым -> старший байт маски идёт первым
        data = [(mask >> (8 * i)) & 0xFF for i in range(7, -1, -1)]

        frame = [duration] + data + [CHECKSUM]
        assert len(frame) == FRAME_LEN
        return frame

    # ------------------------------------------------------------- отправка

    def send(self, valves, duration):
        frame = self.build_frame(valves, duration)
        if self.verbose:
            print("TX: " + " ".join(f"{b:02X}" for b in frame))
        # xfer2 держит CS низким на всю посылку и поднимает его после
        self.spi.xfer2(frame)
        return frame

    def open_valves(self, valves, duration):
        """Открыть указанные клапаны на заданную длительность."""
        return self.send(valves, duration)

    def close_all(self):
        """Все клапаны закрыть."""
        return self.send(0, 0)

    def close(self):
        try:
            self.close_all()
        finally:
            self.spi.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ------------------------------------------------------------------ сценарии

def test_single(board, duration=2, pause=0.5):
    """По очереди открыть каждый клапан — бегущая волна."""
    print(f"Прогон клапанов 0..63, duration={duration}")
    for v in range(VALVE_COUNT):
        print(f"  клапан {v}")
        board.open_valves([v], duration)
        time.sleep(pause)
    board.close_all()


def test_groups(board, duration=2, pause=1.0):
    """Открывать группами по 8 — проверка порядка байт."""
    print(f"Прогон группами по 8, duration={duration}")
    for g in range(8):
        valves = list(range(g * 8, g * 8 + 8))
        print(f"  группа {g}: клапаны {valves[0]}..{valves[-1]}")
        board.open_valves(valves, duration)
        time.sleep(pause)
    board.close_all()


def test_edges(board, duration=2, pause=1.0):
    """
    Крайние клапаны по одному. Если порядок байт перепутан,
    клапан 0 и клапан 63 поменяются местами — это сразу видно.
    """
    for v in (0, 1, 62, 63):
        print(f"  клапан {v}")
        board.open_valves([v], duration)
        time.sleep(pause)
    board.close_all()


def test_speeds(mode=MODE, duration=2, repeats=300, pause=0.02):
    """
    Прогон одного и того же кадра на разных частотах SCK.

    Чем выше частота, тем короче кадр и тем уже яма на CS.
    Смотрите осциллографом ширину ямы и следите, с какой частоты
    плата перестаёт реагировать — рабочей считайте предыдущую.
    """
    speeds = [1_000_000, 2_000_000, 5_000_000,
              10_000_000, 20_000_000, 35_000_000]

    for sp in speeds:
        # ширина кадра без учёта межбайтовых пауз
        theory_us = FRAME_LEN * 8 / sp * 1e6
        print(f"\n--- {sp/1e6:>4.0f} МГц   (80 тактов ~ {theory_us:.1f} мкс) ---")
        try:
            board = ValveBoard(speed=sp, mode=mode)
        except Exception as e:
            print(f"  не удалось открыть на этой частоте: {e}")
            continue

        try:
            actual = board.spi.max_speed_hz
            if actual != sp:
                print(f"  драйвер выставил {actual/1e6:.2f} МГц вместо запрошенной")
            for _ in range(repeats):
                board.open_valves(0xFFFFFFFFFFFFFFFF, duration)
                time.sleep(pause)
        finally:
            board.close()
        time.sleep(1.0)

    print("\nГотово. Рабочей берите частоту на ступень ниже той,")
    print("на которой плата перестала отвечать.")


def test_burst(board, count=50, duration=2):
    """
    Пачка кадров подряд без пауз между ними.
    Удобно, чтобы поймать форму CS на осциллографе без триггера.
    """
    print(f"Пачка из {count} кадров без пауз")
    for _ in range(count):
        board.open_valves(0xFFFFFFFFFFFFFFFF, duration)
    board.close_all()


# ---------------------------------------------------------------------- main

def main():
    import argparse

    p = argparse.ArgumentParser(description="Управление платой клапанов по SPI")
    p.add_argument("--mode", type=int, default=MODE, choices=[0, 1, 2, 3],
                   help=f"режим SPI (по умолчанию {MODE})")
    p.add_argument("--speed", type=int, default=SPEED,
                   help=f"частота SCK в Гц (по умолчанию {SPEED})")
    p.add_argument("--duration", type=int, default=10,
                   help="длительность 0..31 (по умолчанию 10)")
    p.add_argument("--valves", type=str, default=None,
                   help="номера клапанов через запятую, например 0,5,63")
    p.add_argument("--test", choices=["single", "groups", "edges",
                                      "speeds", "burst"],
                   help="запустить тестовый сценарий")
    p.add_argument("--off", action="store_true", help="закрыть все клапаны")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="печатать отправляемые байты")
    args = p.parse_args()

    if args.test == "speeds":
        test_speeds(mode=args.mode, duration=args.duration)
        return

    with ValveBoard(speed=args.speed, mode=args.mode,
                    verbose=args.verbose) as board:
        actual = board.spi.max_speed_hz
        print(f"SPI: mode={args.mode}, {actual/1e6:.2f} МГц, MSB first")
        if actual != args.speed:
            print(f"  (запрошено {args.speed/1e6:.2f} МГц, "
                  f"драйвер округлил до ближайшей доступной)")

        if args.off:
            board.close_all()
            print("Все клапаны закрыты.")
            return

        if args.test == "single":
            test_single(board, args.duration)
        elif args.test == "groups":
            test_groups(board, args.duration)
        elif args.test == "edges":
            test_edges(board, args.duration)
        elif args.test == "burst":
            test_burst(board, duration=args.duration)
        elif args.valves:
            valves = [int(x) for x in args.valves.split(",") if x.strip()]
            board.open_valves(valves, args.duration)
            print(f"Открыты клапаны {valves} на duration={args.duration}")
        else:
            p.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОстановлено, закрываю клапаны.")
        try:
            with ValveBoard() as b:
                b.close_all()
        except Exception:
            pass