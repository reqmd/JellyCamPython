"""
Сортировка: найти предмет, распознать его сетью и отбить клапаном.

Как это идёт по шагам:

    1. Приложение по кнопке "Старт" пишет настройки и регистр 1 = 2.
       Перед этим идут блоки в 120999..., 390000, 9402000..., 9700000 -
       по ним сортировка отличается от съёмки (у съёмки перед стартом
       отдельные записи в 5, 13, 14, 15, а после - метка 65278 в 401999).
    2. Детектор камеры (line_camera.py) находит серию строк с предметами.
       Каждый предмет в ней вырезается отдельно: рамка по ширине и маска
       на чёрном фоне - так же, как готовились картинки для обучения.
    3. Сеть на NPU (модель и классы из models.yaml) называет класс.
    4. Если класс в списке отбиваемых (config.json, sorting.reject_classes),
       выбираются клапаны: зона клапана тянется от середины до соседа слева
       до середины до соседа справа, координаты присылает приложение.
       Предмет отбивают все клапаны, в зону которых он попал.
    5. Клапаны открываются через задержку из регистра 11 после того, как
       передний край предмета прошёл под камерой, на время из регистра 10.
       Оба регистра в миллисекундах.

Регистр 1 с любым значением, кроме 2, останавливает сортировку; метка
съёмки в 401999 - тоже.

Классификация и открытие клапанов идут в своих потоках: детекция камеры
их не ждёт, а клапаны открываются по часам, а не когда сеть освободится.

Настройки - секция "sorting" в config.json (или config.yaml) рядом со
скриптом. Достаточно указать то, что отличается от SORT_DEFAULTS ниже:

    "sorting": {
        "reject_classes": ["bad"]
    }
"""

import heapq
import itertools
import os
import queue
import threading
import time

import cv2
import numpy as np

import valves
from valve_config import (CONFIG, HERE, REGISTERS, VALVE_BASE,
                          VALVE_COUNT_REG)

# --------------------------------------------------------------------------
# Регистры
# --------------------------------------------------------------------------

COMMAND_REG = 1             # 2 - старт, остальное - стоп
COMMAND_START = 2
DURATION_REG = 10           # сколько держать клапан открытым, мс
DELAY_REG = 11              # через сколько после камеры открывать, мс
SHOOT_TAG_REG = 401999      # метка съёмки
SHOOT_TAG = 65278

# Чем начало сортировки отличается от начала съёмки - по логам 14:01 и
# 14:09. Признак сортировки - блоки, которые приложение пишет только перед
# ней. Признак съёмки - ОТДЕЛЬНЫЕ записи в 5, 13, 14, 15: внутри общего
# блока "регистр 2, количество 58" эти регистры есть в обоих режимах.
SORT_MARK_STARTS = {120999, 390000, 9700000}
SORT_MARK_RANGES = ((121000, 125096), (9402000, 9406096))
SHOOT_MARK_STARTS = {5, 13, 14, 15}

# --------------------------------------------------------------------------
# Настройки
# --------------------------------------------------------------------------

SORT_DEFAULTS = {
    # Имена классов, которые надо отбивать - как папки при обучении
    # (они же записаны в models.yaml, поле classes).
    "reject_classes": [],
    # Где лежит models.yaml. Путь к модели внутри него считается от
    # папки, где лежит сам models.yaml.
    "model_config": "../models.yaml",
    # Отбивать, только если сеть уверена хотя бы настолько. 0 - всегда.
    "min_confidence": 0.0,
    # Ядро NPU: 0, 1, 2, или -1 - пусть выбирает сам.
    "npu_core": 0,
    # Сколько миллисекунд в одной единице длительности кадра платы.
    # Регистр 10 приходит в миллисекундах, в кадр уходит 0..31 единиц.
    "ms_per_duration_unit": 1.0,
    # Решение может опоздать: предмет уже пролетел мимо клапанов. Если
    # опоздание меньше этого, клапан открывается сразу; если больше -
    # не открывается совсем, а в сводке растёт счётчик опозданий.
    "late_limit_ms": 20,
    # Команды, которые должны уйти почти одновременно, сливаются в один
    # кадр: так плата получает их вместе, а не одну поверх другой.
    "merge_ms": 2,
    # Как часто печатать сводку, секунд.
    "summary_sec": 10,
    # Печатать ли каждый отбитый предмет. Сводка печатается в любом случае.
    "log_each": True,
    # Сохранять вырезки в папку - проверить, что сеть видит то же, что
    # при обучении. Пустая строка - не сохранять. Не больше save_limit
    # картинок за одну сортировку.
    "save_dir": "",
    "save_limit": 500,
}

CFG = dict(SORT_DEFAULTS)
CFG.update(CONFIG.get("sorting") or {})


# --------------------------------------------------------------------------
# Сеть
# --------------------------------------------------------------------------

def resize_pad(image, size, fill=(0, 0, 0)):
    """
    Ресайз по длинной стороне и паддинг до квадрата - тот же, что в
    linux/train.py и linux/test.py. Иначе сеть увидит не то, на чём
    училась.
    """
    h, w = image.shape[:2]
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


class Classifier:
    """Модель .rknn на NPU. Создавать и вызывать из одного потока."""

    def __init__(self):
        try:
            import yaml
        except ImportError:
            raise RuntimeError("нет модуля PyYAML для чтения models.yaml: "
                               "python3 -m pip install pyyaml")
        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            raise RuntimeError("нет rknn-toolkit-lite2 (модуль rknnlite) "
                               "в этом Python")

        path = CFG["model_config"]
        if not os.path.isabs(path):
            path = os.path.normpath(os.path.join(HERE, path))
        if not os.path.exists(path):
            raise RuntimeError(f"нет файла {path}")
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.size = int(cfg["input_size"][2])
        self.classes = list(cfg["classes"])
        model = cfg["rknn"]
        if not os.path.isabs(model):
            model = os.path.normpath(os.path.join(os.path.dirname(path),
                                                  model))
        if not os.path.exists(model):
            raise RuntimeError(f"нет модели {model}")

        cores = {0: RKNNLite.NPU_CORE_0, 1: RKNNLite.NPU_CORE_1,
                 2: RKNNLite.NPU_CORE_2, -1: RKNNLite.NPU_CORE_AUTO}
        self.rknn = RKNNLite()
        if self.rknn.load_rknn(model) != 0:
            raise RuntimeError(f"load_rknn не открыл {model}")
        if self.rknn.init_runtime(
                core_mask=cores.get(int(CFG["npu_core"]),
                                    RKNNLite.NPU_CORE_0)) != 0:
            raise RuntimeError("init_runtime не запустил NPU")
        self.model = model

    def __call__(self, crop_bgr):
        """Вырезка BGR -> (класс, уверенность 0..1)."""
        padded = resize_pad(crop_bgr, self.size)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        # В NPU идёт uint8 без деления на 255: нормализацию делает сам
        # NPU по mean/std, заданным при сборке модели.
        out = self.rknn.inference(inputs=[np.expand_dims(rgb, 0)])
        logits = np.asarray(out[0], dtype=np.float64).ravel()
        idx = int(np.argmax(logits))
        e = np.exp(logits - logits.max())
        return self.classes[idx], float(e[idx] / e.sum())


# --------------------------------------------------------------------------
# Клапаны по координатам
# --------------------------------------------------------------------------

def valve_zones():
    """
    Зоны клапанов по координатам из приложения: [(левая, правая, индекс)].

    Граница между соседями - посередине. У крайних клапанов зона уходит
    наружу на столько же, на сколько к соседу. Индекс - номер клапана в
    приложении минус один.
    """
    count = int(REGISTERS.get(VALVE_COUNT_REG, 0) or 0)
    pts = sorted((int(REGISTERS.get(VALVE_BASE + i, 0) or 0), i)
                 for i in range(count))
    pts = [p for p in pts if p[0] > 0]
    zones = []
    for k, (c, i) in enumerate(pts):
        if len(pts) == 1:
            left, right = float("-inf"), float("inf")
        else:
            left = ((pts[k - 1][0] + c) / 2 if k
                    else c - (pts[1][0] - c) / 2)
            right = ((c + pts[k + 1][0]) / 2 if k + 1 < len(pts)
                     else c + (c - pts[k - 1][0]) / 2)
        zones.append((left, right, i))
    return zones


def board_valve(index):
    """
    Индекс клапана в приложении -> номер клапана на плате, или None.
    Пересчёт тот же, что при проверке механики (valves.on_write), чтобы
    клапан N в приложении был одним и тем же в обоих режимах.
    """
    v = (index + 1 - valves.VALVE_NUMBER_BASE) // valves.VALVE_DIVISOR
    return v if 0 <= v < valves.VALVE_COUNT else None


def valves_for(x0, x1, zones):
    """Клапаны платы, в зону которых попал предмет [x0, x1)."""
    out = set()
    for left, right, i in zones:
        if x0 < right and x1 > left:
            v = board_valve(i)
            if v is not None:
                out.add(v)
    return sorted(out)


# --------------------------------------------------------------------------
# Состояние
# --------------------------------------------------------------------------

_lock = threading.Lock()
_active = False
_session = 0                 # номер сортировки: старые события не трогать
_next_mode = None            # "sorting" / "shooting" - что подготовлено
_duration_ms = 0
_delay_ms = 0
_units = 1                   # длительность в единицах платы
_saved = 0

_objects = queue.Queue(maxsize=256)
_events = []                 # куча (когда, номер, клапаны, единиц, сессия)
_events_cv = threading.Condition(_lock)
_seq = itertools.count()
_threads = []

_classifier = None
_classifier_error = None


class _Stats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.seen = 0            # предметов пришло от камеры
        self.by_class = {}       # класс -> сколько
        self.rejected = 0        # из них в списке отбиваемых
        self.low_conf = 0        # в списке, но сеть не уверена
        self.no_valve = 0        # отбить нечем: вне зон клапанов
        self.fired = 0           # кадров на плату
        self.late = 0            # открыто с опозданием
        self.too_late = 0        # не открыто: опоздание больше предела
        self.at_edge = 0         # обрезаны краем полосы
        self.dropped = 0         # очередь переполнена
        self.unknown = 0         # не классифицировано: нет модели
        self.latency = []        # мс от камеры до решения

    def line(self):
        cls = ", ".join(f"{k} {v}" for k, v in sorted(self.by_class.items()))
        lat = ""
        if self.latency:
            lat = (f", решение через {np.mean(self.latency):.0f} мс после "
                   f"камеры (макс {max(self.latency):.0f})")
        extra = [f"{name} {val}" for name, val in (
            ("неуверенно", self.low_conf), ("вне клапанов", self.no_valve),
            ("с опозданием", self.late), ("опоздали, не отбиты", self.too_late),
            ("у края полосы", self.at_edge), ("очередь полна", self.dropped),
            ("без модели", self.unknown)) if val]
        return (f"предметов {self.seen} ({cls or 'нет'}), отбито "
                f"{self.rejected}, кадров на плату {self.fired}"
                + (", " + ", ".join(extra) if extra else "") + lat)


_stats = _Stats()
_total = _Stats()            # за всю сортировку, от старта до стопа
_summary_at = time.monotonic()


def _count(name, n=1):
    setattr(_stats, name, getattr(_stats, name) + n)
    setattr(_total, name, getattr(_total, name) + n)


# --------------------------------------------------------------------------
# Старт и стоп
# --------------------------------------------------------------------------

def _read_params():
    """Длительность и задержка из регистров 10 и 11."""
    global _duration_ms, _delay_ms, _units
    _duration_ms = int(REGISTERS.get(DURATION_REG, 0) or 0)
    _delay_ms = int(REGISTERS.get(DELAY_REG, 0) or 0)
    per = float(CFG["ms_per_duration_unit"]) or 1.0
    _units = max(1, min(valves.MAX_DURATION, round(_duration_ms / per)))
    if _duration_ms / per > valves.MAX_DURATION:
        print(f"сортировка: длительность {_duration_ms} мс больше, чем "
              f"помещается в кадр ({valves.MAX_DURATION} единиц по "
              f"{per:g} мс), отправляю {valves.MAX_DURATION}")


def _start():
    global _active, _session, _saved
    import capture
    import line_camera
    if capture.MODE != capture.MODE_LINE_CAMERA:
        print("сортировка: не запущена - в capture.py режим съёмки не "
              "линейная камера (MODE = 1, картинка с диска)")
        return
    with _lock:
        _active = True
        _session += 1
        _events.clear()
        _saved = 0
    _read_params()
    _stats.reset()
    _total.reset()
    line_camera.set_object_sink(_on_objects)

    zones = valve_zones()
    reject = list(CFG["reject_classes"])
    print(f"сортировка: старт. Открывать на {_duration_ms} мс (в кадр "
          f"{_units}), через {_delay_ms} мс после камеры; отбиваем "
          f"{reject or 'НИЧЕГО - список reject_classes пуст'}; клапанов с "
          f"координатами {len(zones)}")
    if _classifier_error:
        print(f"сортировка: сеть недоступна - {_classifier_error}. "
              f"Предметы будут считаться, но не отбиваться")
    elif _classifier is not None:
        unknown = [c for c in reject if c not in _classifier.classes]
        if unknown:
            print(f"сортировка: классов {unknown} нет в модели (есть "
                  f"{_classifier.classes}) - они не отбьются никогда")
    if not line_camera.ready():
        print("сортировка: камера ещё не готова, предметы пойдут, как "
              "только она поднимется")


def _stop(why):
    global _active
    import line_camera
    with _lock:
        if not _active:
            return
        _active = False
        _events.clear()
        _events_cv.notify_all()
    line_camera.set_object_sink(None)
    print(f"сортировка: стоп ({why}). Итого: {_total.line()}")


def is_active():
    return _active


# --------------------------------------------------------------------------
# Записи из приложения
# --------------------------------------------------------------------------

def on_write(start, values):
    """Вызывается сервером на каждую запись, до capture.on_write."""
    global _next_mode
    n = len(values)

    if start in SORT_MARK_STARTS or any(lo <= start < hi
                                        for lo, hi in SORT_MARK_RANGES):
        _next_mode = "sorting"
    elif start in SHOOT_MARK_STARTS and n <= 4:
        _next_mode = "shooting"

    if start <= COMMAND_REG < start + n:
        cmd = values[COMMAND_REG - start]
        if cmd == COMMAND_START:
            # Приложение шлёт каждую запись дважды - второй старт
            # уже идущей сортировки ничего не меняет.
            if _next_mode == "sorting" and not _active:
                _start()
            _next_mode = None
        elif _active:
            _stop(f"регистр {COMMAND_REG} = {cmd}")

    if start <= SHOOT_TAG_REG < start + n and \
            values[SHOOT_TAG_REG - start] == SHOOT_TAG and _active:
        _stop("пришла метка съёмки")

    if _active and (start <= DURATION_REG < start + n
                    or start <= DELAY_REG < start + n):
        was = (_duration_ms, _delay_ms)
        _read_params()
        if (_duration_ms, _delay_ms) != was:
            print(f"сортировка: теперь открывать на {_duration_ms} мс, "
                  f"через {_delay_ms} мс")


# --------------------------------------------------------------------------
# Предметы от камеры
# --------------------------------------------------------------------------

def _on_objects(objs, at_edge):
    """Из потока детекции: только положить в очередь и вернуться."""
    if at_edge:
        _count("at_edge", at_edge)
    if not objs:
        return
    try:
        _objects.put_nowait((_session, time.monotonic(), objs))
    except queue.Full:
        _count("dropped", len(objs))


def _save_crop(crop, name, conf):
    global _saved
    if not CFG["save_dir"] or _saved >= int(CFG["save_limit"]):
        return
    d = CFG["save_dir"]
    if not os.path.isabs(d):
        d = os.path.join(HERE, d)
    d = os.path.join(d, name)
    os.makedirs(d, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(os.path.join(d, f"{stamp}_{_saved:04d}_{conf:.2f}.png"),
                crop)
    _saved += 1


def _decide(obj, received):
    """Классифицировать один предмет и, если надо, назначить клапаны."""
    _count("seen")
    if _classifier is None:
        _count("unknown")
        return
    name, conf = _classifier(obj["crop"])
    _stats.by_class[name] = _stats.by_class.get(name, 0) + 1
    _total.by_class[name] = _total.by_class.get(name, 0) + 1
    _save_crop(obj["crop"], name, conf)

    now = time.monotonic()
    t_front = obj["t_front"] if obj["t_front"] is not None else received
    lat = (now - t_front) * 1000
    _stats.latency.append(lat)
    _total.latency.append(lat)

    if name not in CFG["reject_classes"]:
        return
    if conf < float(CFG["min_confidence"]):
        _count("low_conf")
        return
    hit = valves_for(obj["x0"], obj["x1"], valve_zones())
    if not hit:
        _count("no_valve")
        if CFG["log_each"]:
            print(f"сортировка: {name} {conf:.2f} на x {obj['x0']}..."
                  f"{obj['x1']} - вне зон клапанов, отбить нечем")
        return

    due = t_front + _delay_ms / 1000
    wait = (due - now) * 1000
    if wait < -float(CFG["late_limit_ms"]):
        _count("too_late")
        if CFG["log_each"]:
            print(f"сортировка: {name} {conf:.2f} - решение опоздало на "
                  f"{-wait:.0f} мс, предмет уже пролетел; увеличьте "
                  f"задержку (регистр {DELAY_REG}) или проверьте нагрузку")
        return
    if wait < 0:
        _count("late")
    _count("rejected")
    with _lock:
        heapq.heappush(_events, (due, next(_seq), hit, _units, _session))
        _events_cv.notify()
    if CFG["log_each"]:
        print(f"сортировка: {name} {conf:.2f} на x {obj['x0']}..."
              f"{obj['x1']} -> клапаны {hit} через {max(wait, 0):.0f} мс")


def _worker():
    """Поток сети: грузит модель один раз, затем разбирает очередь."""
    global _classifier, _classifier_error, _summary_at
    try:
        _classifier = Classifier()
        print(f"сортировка: модель {_classifier.model}, классы "
              f"{_classifier.classes}, вход {_classifier.size}x"
              f"{_classifier.size}")
    except Exception as e:
        _classifier_error = str(e) or type(e).__name__
        print(f"сортировка: сеть недоступна - {_classifier_error}")

    while True:
        try:
            session, received, objs = _objects.get(timeout=1.0)
        except queue.Empty:
            session = None
        if session is not None and _active and session == _session:
            for obj in objs:
                try:
                    _decide(obj, received)
                except Exception as e:
                    print(f"сортировка: ошибка при разборе предмета: {e!r}")

        now = time.monotonic()
        if now - _summary_at >= float(CFG["summary_sec"]):
            if _active and _stats.seen:
                print(f"сортировка за {now - _summary_at:.0f} с: "
                      f"{_stats.line()}")
            _stats.reset()
            _summary_at = now


def _firer():
    """
    Поток клапанов: ждёт момента и отправляет кадр. Команды, которые
    приходятся на одно и то же время с точностью до merge_ms, уходят
    одним кадром.
    """
    merge = float(CFG["merge_ms"]) / 1000
    while True:
        with _lock:
            while not _events:
                _events_cv.wait()
            due = _events[0][0]
            now = time.monotonic()
            if due > now:
                _events_cv.wait(timeout=due - now)
                continue
            batch, units = set(), 0
            while _events and _events[0][0] <= now + merge:
                _, _, hit, u, session = heapq.heappop(_events)
                if session == _session and _active:
                    batch.update(hit)
                    units = max(units, u)
        if batch:
            if valves.fire_many(sorted(batch), units):
                _count("fired")


def prepare():
    """При старте сервера: потоки сети и клапанов, загрузка модели."""
    if _threads:
        return
    for target, name in ((_worker, "сортировка"), (_firer, "клапаны")):
        t = threading.Thread(target=target, daemon=True, name=name)
        t.start()
        _threads.append(t)
