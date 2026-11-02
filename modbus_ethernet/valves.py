#!/usr/bin/env python3
"""
Управление платой клапанов через SPI (Orange Pi 5 Plus, /dev/spidev0.0).

Формат кадра - 10 байт, CS удерживается низким на всю посылку:

    [Duration][Data7][Data6][Data5][Data4][Data3][Data2][Data1][Data0][0x99]

    Duration  - длительность работы клапанов, 0..MAX_DURATION
    Data[7]   - клапаны 63..56   (старший байт идёт первым)
    ...
    Data[0]   - клапаны 7..0
    0x99      - контрольный байт (153)

Клапаны на плате нумеруются с нуля: 0..63.

Файл работает двумя способами.

1. Из сервера (dsp_protocol.py) - проверка механики. Приложение пишет в
   регистр 70 номер клапана, сервер открывает этот клапан на плате:

       регистр 70 = N  ->  клапан N - VALVE_NUMBER_BASE на плате
       регистр 70 = 0  ->  все клапаны закрыть

   Если SPI недоступен (нет модуля spidev, нет прав на /dev/spidev0.0),
   сервер продолжает работать как обычно, а в лог пишется причина.

2. Из командной строки - тесты платы, как раньше:

       python3 valves.py --valves 0,5,63
       python3 valves.py --test edges
       python3 valves.py --off
"""

import time

try:
    import spidev
except ImportError:            # без SPI сервер должен работать дальше
    spidev = None

BUS = 0
DEVICE = 0
SPEED = 20_000_000     # 20 МГц, рекомендованное значение (допустимо 1..35 МГц)
MODE = 1               # CPOL=0, CPHA=1. Если не работает - попробуйте 3
CHECKSUM = 0x99        # 153
FRAME_LEN = 10
# Длительность в кадре - 5 бит, 0..31 по формату платы. Если плата на
# значениях больше 10 перестаёт реагировать, верните 10: такой предел
# стоял в исходной версии файла.
MAX_DURATION = 31
VALVE_COUNT = 32

# --------------------------------------------------------------------------
# Проверка механики из приложения
# --------------------------------------------------------------------------

TEST_REG = 70          # регистр, в который приложение пишет номер клапана
# С какого числа приложение считает клапаны. 1 - клапан "1" в приложении
# это клапан 0 на плате (так обычно и нумерует интерфейс: 1..64). Если
# при проверке срабатывает соседний клапан, поставьте 0.
VALVE_NUMBER_BASE = 1
# Сколько номеров приложения приходится на один клапан платы. Приложение
# пишет номера подряд, 1..64, а клапанов (и светодиодов на ленте) 32: на
# каждый клапан по два номера. Делитель 2 сводит 1..64 к клапанам 0..31 -
# проверено на машине, лента пробегается целиком.
VALVE_DIVISOR = 1
# На сколько открывать клапан при проверке, 0..MAX_DURATION. Приложение
# шлёт команды примерно раз в 125 мс, на клапан приходится две подряд.
# Подобрать на глаз можно и без приложения:
#     python3 valves.py --test single --duration 20 --pause 0.25
TEST_DURATION = 20
# Через сколько секунд снова пробовать открыть SPI, если не вышло.
RETRY_SEC = 5.0

INSTALL_HINT = ("модуль spidev не установлен в этот Python. Установите "
                "именно в тот, которым запускается сервер: "
                "python3 -m pip install spidev")


class ValveBoard:
    def __init__(self, bus=BUS, device=DEVICE, speed=SPEED, mode=MODE,
                 verbose=False):
        if spidev is None:
            raise RuntimeError(INSTALL_HINT)
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
        valves   - что открыть: набор номеров 0..63, либо целое 64-битное число
        duration - 0..MAX_DURATION
        Возвращает список из 10 байт.
        """
        if not 0 <= duration <= MAX_DURATION:
            raise ValueError(f"duration вне диапазона 0..{MAX_DURATION}: "
                             f"{duration}")

        if isinstance(valves, int):
            mask = valves
        else:
            mask = 0
            for v in valves:
                if not 0 <= v < VALVE_COUNT:
                    raise ValueError(f"клапан вне диапазона 0..31: {v}")
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
        # xfer2 держит CS низким на всю посылку и поднимает его после.
        # Передаём копию: xfer2 записывает в переданный список принятые
        # по MISO байты, и в логе вместо отправленного кадра оказывались
        # одни FF.
        self.spi.xfer2(list(frame))
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


# --------------------------------------------------------------------------
# Для сервера: одна плата на всю программу, открывается при первой команде
# --------------------------------------------------------------------------

_board = None
_last_error = None
_retry_at = 0.0


def _get_board():
    """
    Плата или None. Ошибка открытия печатается один раз, повторно - только
    если она сменилась, чтобы не засыпать лог одинаковыми строками.
    Пробуем снова не чаще раза в RETRY_SEC: права на устройство могли
    выдать, а модуль - доустановить, не перезапуская сервер.
    """
    global _board, _last_error, _retry_at
    if _board is not None:
        return _board
    now = time.monotonic()
    if now < _retry_at:
        return None
    try:
        _board = ValveBoard()
        print(f"клапаны: SPI открыт, /dev/spidev{BUS}.{DEVICE}, "
              f"{_board.spi.max_speed_hz / 1e6:.0f} МГц, mode {MODE}")
        _last_error = None
        return _board
    except Exception as e:
        msg = str(e) or type(e).__name__
        if msg != _last_error:
            hint = ""
            if isinstance(e, PermissionError):
                hint = (" - у пользователя нет прав на устройство, "
                        "см. группу spi или правило udev")
            elif isinstance(e, FileNotFoundError):
                hint = " - SPI не включён в системе (overlay в armbianEnv)"
            print(f"клапаны: SPI недоступен: {msg}{hint}")
            _last_error = msg
        _retry_at = now + RETRY_SEC
        return None


def prepare():
    """При старте сервера: открыть SPI и сказать, получилось ли."""
    _get_board()


def fire(valve, duration=TEST_DURATION):
    """Открыть один клапан платы (0..63). True, если кадр ушёл."""
    global _board
    board = _get_board()
    if board is None:
        return False
    try:
        frame = board.open_valves([valve], duration)
    except Exception as e:
        # Устройство могло пропасть - закрываем, при следующей команде
        # откроем заново.
        print(f"клапаны: ошибка отправки по SPI: {e}")
        try:
            board.spi.close()
        except Exception:
            pass
        _board = None
        return False
    print(f"клапаны: открыт клапан {valve} на {duration}, кадр "
          + " ".join(f"{b:02X}" for b in frame))
    return True


def close_all():
    """Закрыть все клапаны. True, если кадр ушёл."""
    board = _get_board()
    if board is None:
        return False
    try:
        board.close_all()
    except Exception as e:
        print(f"клапаны: ошибка отправки по SPI: {e}")
        return False
    print("клапаны: все закрыты")
    return True


def on_write(start, values):
    """
    Запись из приложения. Реагируем только на регистр TEST_REG: его
    значение - номер клапана, который надо открыть.
    """
    if not start <= TEST_REG < start + len(values):
        return
    number = values[TEST_REG - start]
    if number == 0:
        print(f"клапаны: регистр {TEST_REG} = 0, закрываю все")
        close_all()
        return
    valve = (number - VALVE_NUMBER_BASE) // VALVE_DIVISOR
    if not 0 <= valve < VALVE_COUNT:
        print(f"клапаны: регистр {TEST_REG} = {number} - такого клапана нет "
              f"на плате (выходит {valve}, а там 0..{VALVE_COUNT - 1}; "
              f"VALVE_NUMBER_BASE = {VALVE_NUMBER_BASE}, VALVE_DIVISOR = "
              f"{VALVE_DIVISOR}), пропускаю")
        return
    print(f"клапаны: регистр {TEST_REG} = {number} -> клапан {valve} "
          f"на плате")
    fire(valve)


def shutdown():
    """При остановке сервера: закрыть клапаны и устройство."""
    global _board
    if _board is not None:
        try:
            _board.close()
        except Exception:
            pass
        _board = None


# ------------------------------------------------------------------ сценарии

def test_single(board, duration=2, pause=0.5):
    """По очереди открыть каждый клапан - бегущая волна."""
    print(f"Прогон клапанов 0..63, duration={duration}")
    for v in range(VALVE_COUNT):
        print(f"  клапан {v}")
        board.open_valves([v], duration)
        time.sleep(pause)
    board.close_all()


def test_groups(board, duration=2, pause=1.0):
    """Открывать группами по 8 - проверка порядка байт."""
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
    клапан 0 и клапан 63 поменяются местами - это сразу видно.
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
    плата перестаёт реагировать - рабочей считайте предыдущую.
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
    import sys

    if spidev is None:
        sys.exit(INSTALL_HINT)

    p = argparse.ArgumentParser(description="Управление платой клапанов по SPI")
    p.add_argument("--mode", type=int, default=MODE, choices=[0, 1, 2, 3],
                   help=f"режим SPI (по умолчанию {MODE})")
    p.add_argument("--speed", type=int, default=SPEED,
                   help=f"частота SCK в Гц (по умолчанию {SPEED})")
    p.add_argument("--duration", type=int, default=MAX_DURATION,
                   help=f"длительность 0..{MAX_DURATION} "
                        f"(по умолчанию {MAX_DURATION})")
    p.add_argument("--pause", type=float, default=0.5,
                   help="пауза между клапанами в тестах single, секунд "
                        "(приложение шлёт раз в ~0.125)")
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
            test_single(board, args.duration, args.pause)
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