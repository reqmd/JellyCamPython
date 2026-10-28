#!/usr/bin/env python3
"""
Стабильная картинка на осциллографе: непрерывный поток одинаковых кадров.

Настройка осциллографа:
  - триггер по СПАДАЮЩЕМУ фронту на CS (пин 24), режим Normal
  - развёртка: 2 мкс/дел на 20 МГц, 20 мкс/дел на 1 МГц
  - каналы: CS (пин 24), SCK (пин 23), MOSI (пин 19)

Запуск:   python3 scope_test.py
          python3 scope_test.py --speed 1000000
          python3 scope_test.py --mode 3 --gap 0.002
"""

import argparse
import sys
import time

try:
    import spidev
except ImportError:
    sys.exit("Нет модуля spidev: sudo apt install python3-spidev")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--speed", type=int, default=20_000_000)
    p.add_argument("--mode", type=int, default=1, choices=[0, 1, 2, 3])
    p.add_argument("--gap", type=float, default=0.001,
                   help="пауза между кадрами в секундах (по умолчанию 1 мс)")
    p.add_argument("--bits", type=int, default=8, choices=[8, 16],
                   help="8 или 16 бит на слово")
    args = p.parse_args()

    # Кадр с чередующимися битами: MOSI хорошо виден как меандр.
    # Duration=21 (0x15), все клапаны 0xAA..., контрольный байт 0x99.
    frame = [0x15] + [0xAA] * 8 + [0x99]

    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = args.speed
    spi.mode = args.mode
    spi.bits_per_word = args.bits
    spi.lsbfirst = False

    actual = spi.max_speed_hz
    period_ns = 1e9 / actual
    print(f"mode={args.mode}  {actual/1e6:.2f} МГц  "
          f"период такта {period_ns:.0f} нс  bits={args.bits}")
    print(f"кадр: {' '.join(f'{b:02X}' for b in frame)}")
    print(f"80 тактов без пауз ~ {80 * period_ns / 1000:.2f} мкс")
    print(f"пауза между кадрами {args.gap * 1000:.1f} мс")
    print("\nCtrl+C для остановки\n")

    n = 0
    try:
        while True:
            spi.xfer2(list(frame))
            n += 1
            time.sleep(args.gap)
    except KeyboardInterrupt:
        print(f"\nОтправлено кадров: {n}")
    finally:
        spi.close()


if __name__ == "__main__":
    main()