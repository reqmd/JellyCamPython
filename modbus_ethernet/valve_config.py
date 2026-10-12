import json
import os
import time

# --------------------------------------------------------------------------
# Конфигурация клапанов: приём, расчёт поля зрения, сохранение на диск
# --------------------------------------------------------------------------

# Куда сохраняется конфигурация клапанов, присланная приложением.
# Файл читается при запуске, поэтому настройки переживают перезапуск сервера.
VALVE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "valves.json")


### НАДО ПЕРЕПИСАТЬ ПОД ИМПОРТ ИЗ JSON !!!!!!!!!!!!!!!!!!!

CAMERA_TYPE = 10        # регистр 41006
FOV_LEFT_REG = 20106    # поле зрения, левая граница
FOV_RIGHT_REG = 20107   # поле зрения, правая граница
VALVE_COUNT_REG = 20108 # число клапанов
VALVE_BASE = 20109      # с этого регистра идут позиции клапанов, по одному на клапан
VALVE_MAX = 128         # потолок числа клапанов
VALVE_W_COUNT_REG = 20237
VALVE_W_BASE = 20238
FOV_LEFT = 1            # начальное поле зрения, пока приложение не прислало клапаны
FOV_RIGHT = 2047
VALVE_COUNT = 64        # начальное число клапанов
FOV_MARGIN = 10         # запас, который формула добавляет за крайними клапанами

REGISTERS = {
    99: 1,
    41006: CAMERA_TYPE,
    FOV_LEFT_REG: FOV_LEFT,
    FOV_RIGHT_REG: FOV_RIGHT,
    VALVE_COUNT_REG: VALVE_COUNT,
    VALVE_W_COUNT_REG: VALVE_COUNT,
}


def compute_fov(coords):
    """
    Поле зрения по крайним клапанам:
        левая  = 2 * coord[0]  - coord[1]  - запас
        правая = 2 * coord[-1] - coord[-2] + запас
    Формула отступает на один шаг за крайний клапан и добавляет запас.
    Результат может оказаться отрицательным - это нормально, в регистр
    он кладётся обрезанным по нулю, а в файл пишется как есть.
    """
    if len(coords) >= 2:
        left = 2 * coords[0] - coords[1] - FOV_MARGIN
        right = 2 * coords[-1] - coords[-2] + FOV_MARGIN
    elif len(coords) == 1:
        left = coords[0] - FOV_MARGIN
        right = coords[0] + FOV_MARGIN
    else:
        return None, None
    return left, right


def clamp_u16(value: int) -> int:
    return 0 if value < 0 else (0xFFFF if value > 0xFFFF else value)


def save_valves(count, coords, left, right):
    """Сохраняет конфигурацию в JSON рядом со скриптом."""
    data = {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "valve_count": count,
        "coords": coords,
        "fov_left": left,
        "fov_right": right,
    }
    tmp = VALVE_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, VALVE_FILE)          # замена одним движением
        print(f"  конфигурация сохранена в {VALVE_FILE}")
    except OSError as e:
        print(f"  не удалось сохранить {VALVE_FILE}: {e}")


def load_valves():
    """Читает сохранённую конфигурацию при запуске, если файл есть."""
    if not os.path.exists(VALVE_FILE):
        return False
    try:
        with open(VALVE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        coords = [int(c) for c in data["coords"]]
        count = int(data["valve_count"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"файл {VALVE_FILE} повреждён, беру настройки по умолчанию: {e}")
        return False

    if not 1 <= count <= VALVE_MAX or len(coords) != count:
        print(f"файл {VALVE_FILE} противоречив, беру настройки по умолчанию")
        return False

    apply_valves(count, coords, announce="восстановлено из файла")
    return True


def apply_valves(count, coords, announce=None):
    """Кладёт конфигурацию в карту регистров и пересчитывает поле зрения."""
    for count_reg, base in ((VALVE_COUNT_REG, VALVE_BASE), (VALVE_W_COUNT_REG, VALVE_W_BASE)):
        REGISTERS[count_reg] = count
        for i in range(VALVE_MAX):
            REGISTERS[base + i] = coords[i] if i < count else 0

    left, right = compute_fov(coords)
    REGISTERS[FOV_LEFT_REG] = clamp_u16(left)
    REGISTERS[FOV_RIGHT_REG] = clamp_u16(right)

    if announce:
        print(f"клапаны ({announce}): {count} шт., "
              f"крайние {coords[0]} .. {coords[-1]}")
    print(f"  поле зрения: {left} .. {right}", end="")
    if left != clamp_u16(left) or right != clamp_u16(right):
        print(f"   (в регистрах {clamp_u16(left)} .. {clamp_u16(right)}, "
              f"обрезано до границ uint16)")
    else:
        print()
    return left, right


def store_valve_write(count_reg, base):
    """
    Вызывается после записи в блок клапанов. Берёт из карты регистров
    число клапанов и координаты, пересчитывает поле зрения, пишет файл.
    """
    count = REGISTERS.get(count_reg, 0)
    if not 1 <= count <= VALVE_MAX:
        print(f"  число клапанов {count} вне диапазона 1..{VALVE_MAX}, "
              f"конфигурация не сохранена")
        return

    coords = [REGISTERS.get(base + i, 0) for i in range(count)]
    print(f"  клапанов: {count}, координаты: "
          f"{coords[:4]}{' ...' if count > 4 else ''}"
          f"{f' {coords[-1]}' if count > 4 else ''}")

    left, right = apply_valves(count, coords)
    save_valves(count, coords, left, right)

