"""
Конфигурация устройства и работа с клапанами.

Настройки лежат в отдельном файле рядом со скриптом:
    config.yaml  - если есть и установлен PyYAML (можно писать комментарии)
    config.json  - иначе; создаётся автоматически при первом запуске

Карта регистров REGISTERS объявлена ЗДЕСЬ и только здесь. dsp_protocol
импортирует этот же объект, поэтому запись и чтение работают с одними
данными. Заводить второй словарь с таким же именем в другом файле нельзя:
это будут два разных словаря, и записанное в один не увидит другой.
"""

import json
import os
import time

try:
    import yaml                      # необязательная зависимость
except ImportError:
    yaml = None

HERE = os.path.dirname(os.path.abspath(__file__))

CONFIG_JSON = os.path.join(HERE, "config.json")
CONFIG_YAML = os.path.join(HERE, "config.yaml")


# --------------------------------------------------------------------------
# Значения по умолчанию. Файл конфигурации накладывается поверх них,
# поэтому в нём можно держать только то, что отличается.
# --------------------------------------------------------------------------

DEFAULTS = {
    "registers": {
        "camera_type": 41006,        # тип камеры
        "fov_left": 20106,           # поле зрения, левая граница
        "fov_right": 20107,          # поле зрения, правая граница
        "valve_count_read": 20108,   # число клапанов, блок чтения
        "valve_base_read": 20109,    # координаты клапанов, блок чтения
        "valve_count_write": 20237,  # число клапанов, блок записи
        "valve_base_write": 20238,   # координаты клапанов, блок записи
    },
    "device": {
        "camera_type_value": 10,     # что отдавать в регистре типа камеры
        "valve_max": 128,            # потолок числа клапанов
        "fov_margin": 10,            # запас за крайними клапанами
    },
    "defaults": {
        "fov_left": 1,               # поле зрения до первой записи клапанов
        "fov_right": 2047,
        "valve_count": 64,           # начальное число клапанов
    },
    "storage": {
        "valve_file": "valves.json",  # куда писать присланную конфигурацию
    },
}


def _merge(base: dict, over: dict) -> dict:
    """Накладывает over поверх base, на два уровня вложенности."""
    result = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key].update(value)
        else:
            result[key] = value
    return result


def _read_config_file():
    """Читает config.yaml или config.json. Возвращает словарь или None."""
    if os.path.exists(CONFIG_YAML):
        if yaml is None:
            print(f"найден {CONFIG_YAML}, но модуль PyYAML не установлен "
                  f"(pip3 install pyyaml); беру настройки по умолчанию")
            return None
        try:
            with open(CONFIG_YAML, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            print(f"не читается {CONFIG_YAML}: {e}; беру настройки по умолчанию")
            return None

    if os.path.exists(CONFIG_JSON):
        try:
            with open(CONFIG_JSON, encoding="utf-8") as f:
                return json.load(f) or {}
        except (OSError, ValueError) as e:
            print(f"не читается {CONFIG_JSON}: {e}; беру настройки по умолчанию")
            return None

    return None


def _write_default_config():
    """Создаёт config.json со значениями по умолчанию, если файла ещё нет."""
    try:
        with open(CONFIG_JSON, "w", encoding="utf-8") as f:
            json.dump(DEFAULTS, f, ensure_ascii=False, indent=2)
        print(f"создан файл настроек {CONFIG_JSON}")
    except OSError as e:
        print(f"не удалось создать {CONFIG_JSON}: {e}")


def _validate(cfg: dict) -> dict:
    """Проверяет значения и приводит их к целым числам."""
    for section in ("registers", "device", "defaults"):
        for key, value in cfg[section].items():
            try:
                cfg[section][key] = int(value)
            except (TypeError, ValueError):
                print(f"настройка {section}.{key} = {value!r} не число, "
                      f"беру значение по умолчанию {DEFAULTS[section][key]}")
                cfg[section][key] = DEFAULTS[section][key]

    reg = cfg["registers"]
    vmax = cfg["device"]["valve_max"]

    # Блоки клапанов не должны налезать друг на друга.
    read_end = reg["valve_base_read"] + vmax
    if reg["valve_count_write"] < read_end:
        print(f"блок записи ({reg['valve_count_write']}) начинается внутри "
              f"блока чтения (до {read_end - 1}); проверьте config")

    if reg["valve_base_read"] != reg["valve_count_read"] + 1:
        print(f"координаты блока чтения должны идти сразу за счётчиком: "
              f"{reg['valve_count_read']} и {reg['valve_base_read']}")
    if reg["valve_base_write"] != reg["valve_count_write"] + 1:
        print(f"координаты блока записи должны идти сразу за счётчиком: "
              f"{reg['valve_count_write']} и {reg['valve_base_write']}")

    if not 1 <= cfg["defaults"]["valve_count"] <= vmax:
        print(f"начальное число клапанов {cfg['defaults']['valve_count']} "
              f"вне диапазона 1..{vmax}, беру {vmax}")
        cfg["defaults"]["valve_count"] = vmax

    return cfg


def load_config() -> dict:
    """Читает файл настроек, накладывает его поверх умолчаний, проверяет."""
    raw = _read_config_file()
    if raw is None:
        if not os.path.exists(CONFIG_YAML) and not os.path.exists(CONFIG_JSON):
            _write_default_config()
        raw = {}
    return _validate(_merge(DEFAULTS, raw))


CONFIG = load_config()

# --------------------------------------------------------------------------
# Константы, собранные из конфигурации. Имена прежние, поэтому
# dsp_protocol.py менять не нужно.
# --------------------------------------------------------------------------

CAMERA_TYPE = CONFIG["device"]["camera_type_value"]
VALVE_MAX = CONFIG["device"]["valve_max"]
FOV_MARGIN = CONFIG["device"]["fov_margin"]

CAMERA_TYPE_REG = CONFIG["registers"]["camera_type"]
FOV_LEFT_REG = CONFIG["registers"]["fov_left"]
FOV_RIGHT_REG = CONFIG["registers"]["fov_right"]
VALVE_COUNT_REG = CONFIG["registers"]["valve_count_read"]
VALVE_BASE = CONFIG["registers"]["valve_base_read"]
VALVE_W_COUNT_REG = CONFIG["registers"]["valve_count_write"]
VALVE_W_BASE = CONFIG["registers"]["valve_base_write"]

FOV_LEFT = CONFIG["defaults"]["fov_left"]
FOV_RIGHT = CONFIG["defaults"]["fov_right"]
VALVE_COUNT = CONFIG["defaults"]["valve_count"]

# Куда сохраняется конфигурация клапанов, присланная приложением.
# Файл читается при запуске, поэтому настройки переживают перезапуск сервера.
VALVE_FILE = os.path.join(HERE, CONFIG["storage"]["valve_file"])


# --------------------------------------------------------------------------
# Карта регистров. Всё, чего нет в словаре, равно нулю.
# --------------------------------------------------------------------------

REGISTERS = {
    99: 1,                          # связь с устройством установлена
    CAMERA_TYPE_REG: CAMERA_TYPE,
    FOV_LEFT_REG: FOV_LEFT,
    FOV_RIGHT_REG: FOV_RIGHT,
    VALVE_COUNT_REG: VALVE_COUNT,
    VALVE_W_COUNT_REG: VALVE_COUNT,
}

# Стартовая раскладка: клапаны равномерно по полю зрения, каждый в центре
# своей полосы. Заполняются оба блока - и для чтения, и для записи.
_step = (FOV_RIGHT - FOV_LEFT) / VALVE_COUNT
for _i in range(VALVE_COUNT):
    _pos = round(FOV_LEFT + (_i + 0.5) * _step)
    REGISTERS[VALVE_BASE + _i] = _pos
    REGISTERS[VALVE_W_BASE + _i] = _pos


# --------------------------------------------------------------------------
# Расчёт поля зрения и сохранение конфигурации клапанов
# --------------------------------------------------------------------------

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
    """Сохраняет конфигурацию клапанов рядом со скриптом."""
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
    """Читает сохранённую конфигурацию клапанов при запуске, если файл есть."""
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
    """Кладёт конфигурацию в оба блока и пересчитывает поле зрения."""
    for count_reg, base in ((VALVE_COUNT_REG, VALVE_BASE),
                            (VALVE_W_COUNT_REG, VALVE_W_BASE)):
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
    Вызывается после записи в блок клапанов. Берёт число клапанов и
    координаты из того блока, куда писали, пересчитывает поле зрения,
    раскладывает по обоим блокам и пишет файл.
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