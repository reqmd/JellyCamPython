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
       Задержка - в тактах по 250 мкс (настройка delay_tick_ms): 10 в
       приложении - это 2,5 мс, 40 - 10 мс. Длительность - в мс.

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
import diagnostic_capture as diag
from valve_config import (CONFIG, HERE, REGISTERS, VALVE_BASE,
                          VALVE_COUNT_REG)

# --------------------------------------------------------------------------
# Регистры
# --------------------------------------------------------------------------

COMMAND_REG = 1             # 2 - старт, остальное - стоп
COMMAND_START = 2
DELAY_REG = 11              # через сколько после камеры открывать, такты
DURATION_REG = 10           # сколько держать клапан открытым, мс
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
    # Длина такта задержки (регистр 11), мс. Приложение задаёт задержку
    # не в миллисекундах, а в тактах по 250 мкс: 10 - это 2,5 мс.
    "delay_tick_ms": 0.25,
    # То же для длительности (регистр 10). Она приходит в миллисекундах.
    "duration_tick_ms": 1.0,
    # Сколько миллисекунд в одной единице длительности кадра платы (в
    # кадр уходит 0..31 единиц). 1.0 - число из регистра 10 уходит в
    # кадр как есть.
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
    # Где клапаны на снимке ЭТОЙ камеры, столбцы сенсора 0..2047. Пусто -
    # брать координаты из приложения. Приложение настроено под старую
    # камеру, и у новой масштаб, сдвиг, а то и направление могут быть
    # другими - тогда клапаны срабатывают, но соседние. Варианты:
    #   [x1, xN]        - центр первого и последнего клапана, остальные
    #                     поровну между ними; x1 > xN, если камера видит
    #                     ленту зеркально;
    #   [x1, x2, ... xN] - центр каждого клапана, по номерам.
    "valve_x": [],
    # Инверсия позиции предмета: камера видит ленту зеркально - предмет,
    # пролетающий у правого края, на снимке слева. true - позиция
    # отражается перед выбором клапана, и предмет под клапаном 1 уходит
    # клапану 32, под клапаном 2 - клапану 31 и так далее.
    "mirror_x": False,
    # Ось отражения, столбец сенсора. null - середина между первым и
    # последним клапаном: тогда номера клапанов просто идут в обратном
    # порядке. 1023.5 - середина сенсора.
    "mirror_center": None,
}

CFG = dict(SORT_DEFAULTS)
CFG.update(CONFIG.get("sorting") or {})
LOG_CFG = dict(CONFIG.get("logging") or {})
LOG_OBJECTS = bool(LOG_CFG.get("log_objects", CFG["log_each"]))
LOG_SUMMARY = bool(LOG_CFG.get("log_summary", True))
SUMMARY_SEC = max(0.1, float(LOG_CFG.get("summary_sec", CFG["summary_sec"])))
LOG_POSITIONS = bool(LOG_CFG.get("log_positions", False))
NPU_WARN_MS = float(LOG_CFG.get("npu_warn_ms", 5.0))
NPU_DETAIL_EACH = bool(LOG_CFG.get("npu_detail_each", False))
PIPELINE_TIMING = bool(LOG_CFG.get("pipeline_timing", True))
PIPELINE_TIMING_EACH = bool(LOG_CFG.get("pipeline_timing_each", True))
# Имена классов - строки (это имена папок при обучении). Если в настройках
# записано число, например 0 вместо "0", приводим к строке, иначе класс
# не совпал бы ни с одним ответом сети и молча не отбивался бы.
CFG["reject_classes"] = [str(c) for c in (CFG["reject_classes"] or [])]


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

        # Первый вывод после загрузки в разы медленнее остальных: NPU
        # прогревается. Делаем это сразу, иначе опоздает первый предмет.
        dummy = np.zeros((1, self.size, self.size, 3), np.uint8)
        for _ in range(10):
            self.rknn.inference(inputs=[dummy])
        t = time.monotonic()
        for _ in range(20):
            self.rknn.inference(inputs=[dummy])
        self.infer_ms = (time.monotonic() - t) / 20 * 1000

    def classify_timed(self, crop_bgr):
        """(class, confidence, timings): preparation / RKNN / result, ms.

        `thread_cpu_ms` measures CPU consumption by this Python thread,
        while `rknn_ms` measures elapsed wall time, including waits in the
        driver. Their difference is only a diagnostic clue, not proof of
        which device or lock caused a long call.
        """
        start = time.perf_counter_ns()
        padded = resize_pad(crop_bgr, self.size)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(rgb, 0)
        before_rknn = time.perf_counter_ns()
        cpu_before = time.thread_time_ns()
        out = self.rknn.inference(inputs=[inp])
        cpu_after = time.thread_time_ns()
        after_rknn = time.perf_counter_ns()
        logits = np.asarray(out[0], dtype=np.float64).ravel()
        idx = int(np.argmax(logits))
        e = np.exp(logits - logits.max())
        confidence = float(e[idx] / e.sum())
        end = time.perf_counter_ns()
        return (self.classes[idx], confidence, {
            'Подг': (before_rknn-start) / 1e6,
            'RKNN': (after_rknn-before_rknn) / 1e6,
            'Рез': (end-after_rknn) / 1e6,
            'RKNN_CPU': (cpu_after-cpu_before) / 1e6,
        })

    def __call__(self, crop_bgr):
        """Compatibility with existing callers."""
        name, conf, _ = self.classify_timed(crop_bgr)
        return name, conf


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
    pts = sorted(_valve_points()[0])
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


def _valve_points():
    """
    Центры клапанов: ([(x, индекс)], откуда взяты).
    Из настройки valve_x, если она задана, иначе из приложения.
    """
    vx = [float(v) for v in (CFG.get("valve_x") or [])]
    n = valves.VALVE_COUNT
    if len(vx) == 2 and n > 1:
        a, b = vx
        return ([(a + (b - a) * i / (n - 1), i) for i in range(n)],
                f"config.json, valve_x {a:g}..{b:g}")
    if len(vx) > 2:
        return [(x, i) for i, x in enumerate(vx)], "config.json, valve_x"
    count = int(REGISTERS.get(VALVE_COUNT_REG, 0) or 0)
    pts = [(int(REGISTERS.get(VALVE_BASE + i, 0) or 0), i)
           for i in range(count)]
    return [p for p in pts if p[0] > 0], "приложение"


def _mirror_axis():
    """Столбец, вокруг которого отражается позиция, или None - без инверсии."""
    if not CFG.get("mirror_x"):
        return None
    if CFG.get("mirror_center") is not None:
        return float(CFG["mirror_center"])
    xs = [x for x, _ in _valve_points()[0]]
    return (min(xs) + max(xs)) / 2 if xs else 1023.5


def object_position(obj):
    """
    Где предмет для выбора клапана: (x0, x1, описание для лога).

    x0, x1 - столбцы снимка; при инверсии - отражённые. Описание говорит,
    где предмет на снимке, который видит приложение: столбцы, центр и
    доля ширины слева направо, и куда он ушёл после инверсии.
    """
    import capture
    x0, x1 = obj["x0"], obj["x1"]
    c = (x0 + x1) / 2
    width = getattr(capture, "WIDTH", 2048) or 2048
    text = (f"на снимке x {x0}..{x1}, {c / width * 100:.0f}% ширины "
            f"слева")
    axis = _mirror_axis()
    if axis is not None:
        x0, x1 = 2 * axis - x1, 2 * axis - x0
        text += f", с инверсией x {x0:.0f}..{x1:.0f}"
    return x0, x1, text


def _span(hit):
    """Клапаны для лога: '3-5', '7' или '-'."""
    n = _numbers(hit)
    if not n:
        return "-"
    return str(n[0]) if len(n) == 1 else f"{n[0]}-{n[-1]}"


def _numbers(hit):
    """Номера клапанов для лога - как в приложении и в проверке механики."""
    return [v * valves.VALVE_DIVISOR + valves.VALVE_NUMBER_BASE for v in hit]


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

_lock = threading.RLock()  # события и согласованные счётчики сессии
_active = False
_session = 0                 # номер сортировки: старые события не трогать
_next_mode = None            # "sorting" / "shooting" - что подготовлено
_duration_ms = 0
_delay_ms = 0
_units = 1                   # длительность в единицах платы
_saved = 0
_started_at = None
_camera_base = None
_pipeline_previous = None
_logger_base = 0
_spi_inflight = 0

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
        self.seen = 0
        self.queued = 0        # объектов принято в очередь NPU
        self.by_class = {}
        self.rejected = 0       # поставлены в очередь клапанов (не факт отбивания)
        self.low_conf = 0
        self.no_valve = 0
        self.fired = 0          # успешные вызовы SPI, кадры могут объединяться
        self.spi_errors = 0
        self.late = 0           # решения, поставленные в очередь позже срока
        self.too_late = 0       # команда отменена по пределу опоздания
        self.at_edge = 0
        self.dropped = 0
        self.unknown = 0
        self.errors = 0
        self.no_time = 0
        self.latency = []
        self.parts = {k: [] for k in ("Сбор", "Дет", "КамОч", "КамОчЗак", "Возраст", "ВозрастЗак", "Сним", "СнимCPU", "ВырОч", "Выр", "Оч", "NPU", "Подг", "RKNN", "Рез", "Проч")}
        self.spi_late = []      # отклонение начала вызова SPI от due, со знаком

    def short(self):
        cls = " ".join(f"Кл{k}={v}" for k, v in sorted(self.by_class.items()))
        text = (f"объектов={self.seen} | {cls or 'нет классов'} | "
                f"OK={self.rejected - self.late} Поздно={self.late} "
                f"Не отправлено={self.too_late} Проход={self.passed()} "
                f"| SPI кадров={self.fired} ошибок={self.spi_errors}")
        bad = [f"{name}={value}" for name, value in (
            ("край", self.at_edge), ("потеряно", self.dropped),
            ("вне клапанов", self.no_valve), ("неуверенно", self.low_conf),
            ("без модели", self.unknown), ("ошибки", self.errors)) if value]
        if bad:
            text += " | " + " ".join(bad)
        if self.latency:
            text += (f" | Итог сред={np.mean(self.latency):.2f} "
                     f"p95={np.percentile(self.latency, 95):.2f} "
                     f"макс={max(self.latency):.2f} мс")
        return text

    def passed(self):
        return sum(v for k, v in self.by_class.items()
                   if k not in CFG["reject_classes"])

    def line(self):
        n = self.rejected + self.too_late + self.low_conf + self.no_valve
        on = self.rejected - self.late
        pct = lambda v: f" ({100*v/n:.1f}%)" if n else ""
        cls = " | ".join(f"Класс {k}: {v}" for k, v in sorted(self.by_class.items()))
        lines = [f"  Объектов классифицировано: {sum(self.by_class.values())} | {cls or 'нет классов'}",
                 f"  Принято в очередь: {self.queued} | разобрано: {self.seen} | "
                 f"не разобрано на стоп: {max(0, self.queued - self.seen - self.errors)}",
                 f"  Вовремя: {on}{pct(on)} | С опозданием: {self.late}{pct(self.late)} | "
                 f"Не отправлено по сроку: {self.too_late}{pct(self.too_late)} | Проход: {self.passed()}",
                 f"  Отказы: низкая уверенность={self.low_conf} вне клапанов={self.no_valve} "
                 f"край={self.at_edge} очередь полна={self.dropped} без модели={self.unknown} "
                 f"ошибки={self.errors} без метки времени={self.no_time}",
                 "  ВРЕМЯ, мс       СРЕДНЕЕ      P95      МАКС"]
        for name, values in [*self.parts.items(), ("ИТОГ", self.latency),
                             ("SPI откл.", self.spi_late)]:
            if values:
                lines.append(f"  {name:<14} {np.mean(values):8.2f} "
                             f"{np.percentile(values, 95):8.2f} {max(values):8.2f}")
            else:
                lines.append(f"  {name:<14} {'—':>8} {'—':>8} {'—':>8}")
        lines.append(f"  SPI: успешных кадров={self.fired} | ошибок={self.spi_errors}")
        return "\n".join(lines)


_stats = _Stats()
_total = _Stats()
_summary_at = time.monotonic()


def _count(name, n=1):
    with _lock:
        setattr(_stats, name, getattr(_stats, name) + n)
        setattr(_total, name, getattr(_total, name) + n)


def _camera_summary(before, after):
    if after is None:
        return "  Камера: нет данных"
    prefix = ""
    if before is None:
        before = {}
        prefix = "  Камера не была готова на старте; ниже счётчики с её запуска.\n"
    def delta(key):
        return after.get(key, 0) - before.get(key, 0)
    text = (f"  Камера: порций={delta('grabbed')} потери={delta('dropped')} "
            f"разрывы={delta('gaps')} восстановлений={delta('recoveries')} "
            f"ошибки вырезки={delta('sort_errors')} очередь={after['queue']}")
    if after.get('cut_async'):
        ncut = delta('cut_wait_n')
        wms = delta('cut_wait_sum_ms')
        text += (f"\n  ВЫРЕЗАНИЕ: принято={delta('cut_enqueued')} "
                 f"готово={delta('cut_completed')} "
                 f"резервных={delta('cut_queue_full')} "
                 f"в очереди={after.get('cut_queue', 0)} "
                 f"макс очередь={after.get('cut_pending_max', 0)} "
                 f"ожидание сред={wms/ncut if ncut else 0:.2f} мс "
                 f"макс={after.get('cut_wait_max_ms', 0):.2f} мс")
    if "stream_pieces" in after:
        n = delta('stream_pieces')
        text += (f"\n  STREAM: завершено={delta('stream_finished')} "
                 f"выдано={delta('stream_emitted')} пустых={delta('stream_empty')} "
                 f"невалидных={delta('stream_invalid')} объединено={delta('stream_merged')} "
                 f"сброшено={delta('stream_resets')} активно на стоп={after['stream_active']}")
        if n > 0:
            text += (f" | маска={delta('stream_mask_ns')/n/1e6:.3f} "
                     f"трек={delta('stream_track_ns')/n/1e6:.3f} мс/порция")
    if after.get('pipeline_timing'):
        text += "\n" + _pipeline_summary(before, after)
    return prefix + text


def _pipeline_summary(before, after):
    """Compact stage costs. Maxima are since session start, averages are deltas.

    Counters are single-writer and sampled without interrupting the detector.
    Per-stage timings may overlap: CPU is a subset of wall STREAM time,
    and mask/track/callback are subsets of STREAM wall time.
    """
    if not after or not after.get('pipeline_timing'):
        return ''
    before = before or {}
    def mean(stage):
        count = after.get(stage+'_n', 0) - before.get(stage+'_n', 0)
        ns = after.get(stage+'_sum_ns', 0) - before.get(stage+'_sum_ns', 0)
        return ns / count / 1e6 if count > 0 else 0.0
    def maxval(stage):
        return after.get(stage+'_max_ns', 0) / 1e6
    def fmt(stage):
        return f'{mean(stage):.2f}/{maxval(stage):.2f}'
    pieces = after.get('stream_pieces', 0) - before.get('stream_pieces', 0)
    if pieces > 0:
        mask = (after.get('stream_mask_ns', 0) - before.get('stream_mask_ns', 0)) / pieces / 1e6
        track = (after.get('stream_track_ns', 0) - before.get('stream_track_ns', 0)) / pieces / 1e6
    else:
        mask = track = 0.0
    ncb = after.get('stream_onrun_n', 0) - before.get('stream_onrun_n', 0)
    cb = ((after.get('stream_onrun_sum_ns', 0) - before.get('stream_onrun_sum_ns', 0))
          / ncb / 1e6 if ncb else 0.0)
    nsnap = after.get('snapshot_n', 0) - before.get('snapshot_n', 0)
    snap = ((after.get('snapshot_sum_ns', 0) - before.get('snapshot_sum_ns', 0))
            / nsnap / 1e6 if nsnap else 0.0)
    return (f'  ПРОФИЛЬ мс сред/макс*: ОчКам={fmt("cam_queue")} '
            f'Возраст={fmt("cam_age")} Граф={fmt("graph")} '
            f'STREAM={fmt("feed")} CPU={fmt("feed_cpu")} '
            f'буферОчМакс={after.get("pipeline_pending_max", 0)}\n'
            f'  STREAM мс: Маска={mask:.3f}/{after.get("stream_mask_max_ns", 0)/1e6:.3f} '
            f'Трек={track:.3f}/{after.get("stream_track_max_ns", 0)/1e6:.3f} '
            f'Колбэк={cb:.3f}/{after.get("stream_onrun_max_ns", 0)/1e6:.3f} '
            f'Сним={snap:.3f}/{after.get("snapshot_max_ns", 0)/1e6:.3f} '
            f'(колбэков={ncb}, снимков={nsnap}; *макс с начала сессии)')


# --------------------------------------------------------------------------
# Старт и стоп
# --------------------------------------------------------------------------

def _read_params():
    """Задержка из регистра 11 и длительность из регистра 10."""
    global _duration_ms, _delay_ms, _units
    _duration_ms = (int(REGISTERS.get(DURATION_REG, 0) or 0)
                    * (float(CFG["duration_tick_ms"]) or 1.0))
    _delay_ms = (int(REGISTERS.get(DELAY_REG, 0) or 0)
                 * (float(CFG["delay_tick_ms"]) or 1.0))
    per = float(CFG["ms_per_duration_unit"]) or 1.0
    _units = max(1, min(valves.MAX_DURATION, round(_duration_ms / per)))
    if _duration_ms / per > valves.MAX_DURATION:
        print(f"сортировка: длительность {_duration_ms:g} мс больше, чем "
              f"помещается в кадр ({valves.MAX_DURATION} единиц по "
              f"{per:g} мс), отправляю {valves.MAX_DURATION}")


def _params_text():
    """Регистры 10 и 11 вместе с тем, сколько это в миллисекундах."""
    return (f"открывать на {_duration_ms:g} мс (регистр {DURATION_REG} = "
            f"{REGISTERS.get(DURATION_REG, 0)}, в кадр {_units}), через "
            f"{_delay_ms:g} мс после камеры (регистр {DELAY_REG} = "
            f"{REGISTERS.get(DELAY_REG, 0)})")


def _check_layout(zones):
    """
    Где стоят клапаны и сходится ли их число с платой.

    Раскладку присылает приложение (запись в 20237), сервер хранит её в
    valves.json. Если приложение настроено на другую машину или другой тип
    камеры, число и шаг клапанов будут не те - и предметы станут отбиваться
    не своими клапанами. Снаружи это похоже на "клапаны бьют не туда",
    поэтому говорим об этом прямо, на старте.
    """
    n = len(zones)
    if not n:
        print("сортировка: ВНИМАНИЕ - координат клапанов нет, отбивать "
              "нечем. Пришлите раскладку из приложения")
        return
    import line_camera
    pts, source = _valve_points()
    xs = sorted(x for x, _ in pts)
    step = (xs[-1] - xs[0]) / (n - 1) if n > 1 else 0
    first = min(pts, key=lambda p: p[1])[0]
    last = max(pts, key=lambda p: p[1])[0]
    if LOG_POSITIONS:
        print(f"сортировка: клапаны по координатам ({source}) - {n} шт., "
              f"клапан 1 на x {first:.0f}, клапан {n} на x {last:.0f}, шаг "
              f"{step:.1f}; на плате подключено {valves.VALVE_COUNT}; "
              f"предметы ищутся в столбцах {line_camera.X0}..{line_camera.X1}")
        axis = _mirror_axis()
        print("сортировка: инверсия позиции " +
              ("выключена (mirror_x)" if axis is None else
               f"ВКЛЮЧЕНА, ось x {axis:.1f}: предмет слева на снимке отбивается "
               f"клапанами справа"))
    if n != valves.VALVE_COUNT:
        print(f"сортировка: ВНИМАНИЕ - приложение прислало {n} клапанов, а "
              f"на плате {valves.VALVE_COUNT}. Зоны стоят не на своих "
              f"местах, часть из них ведёт на несуществующие клапаны: "
              f"предметы будут отбиваться не там или не отбиваться. "
              f"Проверьте тип камеры и число клапанов в приложении и "
              f"пришлите раскладку заново")
    if xs[0] < 0 or xs[-1] > 2047:
        print(f"сортировка: ВНИМАНИЕ - координаты клапанов выходят за "
              f"0..2047, то есть за ширину сенсора: в приложении, похоже, "
              f"выбрана камера с другой матрицей")


def _start():
    global _active, _session, _saved, _started_at, _camera_base, _pipeline_previous, _logger_base, _summary_at, _stats, _total
    import capture
    import line_camera
    if capture.MODE != capture.MODE_LINE_CAMERA:
        print("сортировка: не запущена - в capture.py режим съёмки не "
              "линейная камера (MODE = 1, картинка с диска)")
        return
    import logger
    with _lock:
        _session += 1
        _events.clear()
        _saved = 0
        _read_params()
        _stats = _Stats()
        _total = _Stats()
        _started_at = _summary_at = time.monotonic()
        if PIPELINE_TIMING:
            line_camera.reset_sorting_profile()
        _camera_base = line_camera.sorting_metrics()
        _pipeline_previous = _camera_base
        _logger_base = logger.dropped_count()
        _active = True
    session = _session
    zones = valve_zones()
    reject = list(CFG["reject_classes"])
    print(f"=== СОРТИРОВКА: СТАРТ | сессия={session} ===")
    cam = _camera_base or {}
    print(f"  Камера: {cam.get('width', '?')}x{cam.get('chunk_rows', '?')} | "
          f"LINES={line_camera.LINES} PROC_ROWS={line_camera.PROC_ROWS} | "
          f"задано {line_camera.ROWS_PER_SEC:g} строк/с | "
          f"выдержка={line_camera.EXPOSURE_MS:g} мс")
    print(f"  Вырезание: {'поток + очередь' if (line_camera.CUT_ASYNC and line_camera.STREAM_TRACKING) else 'синхронно'} | "
          f"очередь={line_camera.CUT_QUEUE_SIZE} | "
          f"NPU-предупреждение от {NPU_WARN_MS:g} мс | "
          f"профиль стадий={'ON' if PIPELINE_TIMING else 'OFF'}")
    print(f"  Детектор: {'STREAM' if line_camera.STREAM_TRACKING else 'SERIES'} | "
          f"ROI={line_camera.X0}..{line_camera.X1} TH={line_camera.TH} "
          f"GAP={line_camera.GAP_ROWS} MIN_ROWS={line_camera.MIN_ROWS} MIN_AREA={line_camera.MIN_AREA} "
          f"LINK_X={line_camera.STREAM_LINK_X} MIN_RUN={line_camera.STREAM_MIN_RUN}")
    model = _classifier
    print(f"  Модель: {f'{model.size}x{model.size} | классы={model.classes}' if model is not None else 'не готова'} "
          f"| отбиваем={reject}")
    print(f"  Клапаны: {len(zones)} | задержка={_delay_ms:.2f} мс | импульс={_duration_ms:.2f} мс "
          f"| инверсия={'ON' if CFG['mirror_x'] else 'OFF'}")
    _check_layout(zones)
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
    diag_params = {k:getattr(line_camera,k) for k in
        ('X0','X1','LINES','PROC_ROWS','FIXED_RATE','ROWS_PER_SEC','TH','GAP_ROWS',
         'MIN_ROWS','MIN_AREA','STREAM_LINK_X','STREAM_MIN_RUN','OPEN_K','CLOSE_K','EDGE_OLD','MEDIAN_KERNEL')}
    diag_params.update(delay_ms=_delay_ms, duration_ms=_duration_ms,
                       reject_classes=reject, mirror_x=bool(CFG['mirror_x']))
    diag.start(session, line_camera._camera, diag_params)
    line_camera.set_object_sink(lambda objs, edge: _on_objects(objs, edge, session))


def _stop(why):
    global _active
    import line_camera
    import logger
    with _lock:
        if not _active:
            return
        _active = False
        cancelled = len(_events)
        _events.clear()
        _events_cv.notify_all()
        after = line_camera.sorting_metrics()
        line_camera.set_object_sink(None)
        # Снимок ровно на переходе в ожидание. Уже начавшийся SPI-вызов
        # может завершиться позже; показываем это отдельно.
        session = _session
        elapsed = time.monotonic() - _started_at
        total, before = _total, _camera_base
        inflight = _spi_inflight
        logs_dropped = logger.dropped_count() - _logger_base
    print(f"=== СОРТИРОВКА: СТОП ({why}) | сессия={session} ===")
    print(f"  Длительность: {elapsed:.2f} сек")
    print(total.line())
    print(_camera_summary(before, after))
    print(f"  На стоп: отменено событий клапанов={cancelled} "
          f"SPI в работе={inflight} | пропущено строк лога={logs_dropped}")
    diag.stop()
    if why == f"регистр {COMMAND_REG} = 1":
        print("=== КАМЕРА: РЕЖИМ ОЖИДАНИЯ ===")
    else:
        print("=== СОРТИРОВКА НЕАКТИВНА ===")


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
        with _lock:
            was = (_duration_ms, _delay_ms)
            _read_params()
            if (_duration_ms, _delay_ms) != was:
                print(f"сортировка: теперь {_params_text()}")


# --------------------------------------------------------------------------
# Предметы от камеры
# --------------------------------------------------------------------------

def _on_objects(objs, at_edge, session):
    """Из потока детекции: только положить в очередь и вернуться."""
    with _lock:
        if not _active or session != _session:
            return
        if at_edge:
            _count("at_edge", at_edge)
        if not objs:
            return
        try:
            _objects.put_nowait((session, time.monotonic(), objs))
            _count("queued", len(objs))
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


def _decide(obj, received, session):
    """Классификация и прежняя политика планирования; новый только лог."""
    with _lock:
        if not _active or session != _session:
            return
        model = _classifier
        if model is None:
            _count("seen")
            _count("unknown")
            return
    crop = obj["crop"]
    t_nn = time.monotonic()
    if hasattr(model, 'classify_timed'):
        name, conf, npu_detail = model.classify_timed(crop)
    else:
        name, conf = model(crop)
        npu_detail = {}
    t_nn_end = time.monotonic()
    nn_ms = (t_nn_end - t_nn) * 1000
    if nn_ms >= NPU_WARN_MS:
        print(f'  NPU ДОЛГО: {nn_ms:.2f} мс | '
              f'подготовка={npu_detail.get("Подг", 0):.2f} '
              f'RKNN={npu_detail.get("RKNN", nn_ms):.2f} '
              f'результат={npu_detail.get("Рез", 0):.2f} '
              f'CPU в RKNN={npu_detail.get("RKNN_CPU", 0):.2f} мс')
    _save_crop(crop, name, conf)

    # Срок по-прежнему отсчитывается от t_front. Измеряем решение
    # после вычисления клапанов, непосредственно перед планированием.
    actual_front = obj.get("t_front")
    t_front = actual_front if actual_front is not None else received
    t_last = obj.get("t_last")
    t_found = obj.get("t_found")
    cut_end = obj.get("t_cut_end")
    parts = {"NPU": nn_ms, "Оч": (t_nn - (cut_end if cut_end is not None else received)) * 1000}
    parts.update({k: npu_detail[k] for k in ("Подг", "RKNN", "Рез") if k in npu_detail})
    if obj.get("cut_ms") is not None:
        parts["Выр"] = obj["cut_ms"]
    if obj.get("cut_wait_ms") is not None:
        parts["ВырОч"] = obj['cut_wait_ms']
    if PIPELINE_TIMING:
        # Do not shadow `name`: it is the RKNN class label used to decide
        # whether the object must be rejected by the valves.
        for metric_name, key in (("КамОч", "camera_queue_ms"),
                                 ("КамОчЗак", "camera_finish_queue_ms"),
                                 ("Возраст", "camera_age_ms"),
                                 ("ВозрастЗак", "camera_finish_age_ms"),
                                 ("Сним", "snapshot_ms"),
                                 ("СнимCPU", "snapshot_cpu_ms")):
            if obj.get(key) is not None:
                parts[metric_name] = obj[key]
    if actual_front is not None and t_last is not None:
        parts["Сбор"] = (t_last - actual_front) * 1000
        if t_found is not None:
            parts["Дет"] = (t_found - t_last) * 1000
    x0, x1, where = object_position(obj)
    hit = valves_for(x0, x1, valve_zones())

    with _lock:
        if not _active or session != _session:
            return
        now = time.monotonic()
        lat = (now - t_front) * 1000
        if actual_front is not None and all(k in parts for k in ("Сбор", "Дет", "Выр", "Оч", "NPU")):
            # The detailed NPU stages are a *breakdown* of NPU, not extra
            # elapsed time; do not subtract them twice from the residual.
            parts["Проч"] = lat - sum(parts.get(k, 0) for k in
                ("Сбор", "Дет", "Сним", "ВырОч", "Выр", "Оч", "NPU"))
        _count("seen")
        _stats.by_class[name] = _stats.by_class.get(name, 0) + 1
        _total.by_class[name] = _total.by_class.get(name, 0) + 1
        if actual_front is not None:
            _stats.latency.append(lat)
            _total.latency.append(lat)
        else:
            _count("no_time")
        for k, v in parts.items():
            _stats.parts[k].append(v)
            _total.parts[k].append(v)
        number = _total.seen
        due = t_front + _delay_ms / 1000
        wait = (due - now) * 1000
        event_id = None
        if name not in CFG["reject_classes"]:
            status = "ПРОХОД"
        elif conf < float(CFG["min_confidence"]):
            _count("low_conf")
            status = "НЕ ОТПРАВЛЕН: уверенность"
        elif not hit:
            _count("no_valve")
            status = "НЕ ОТПРАВЛЕН: вне клапанов"
        elif wait < -float(CFG["late_limit_ms"]):
            _count("too_late")
            status = f"НЕ ОТПРАВЛЕН +{-wait:.2f}"
        else:
            _count("rejected")
            event_id = next(_seq)
            heapq.heappush(_events, (due, event_id, hit, _units, session))
            _events_cv.notify()
            if wait < 0:
                _count("late")
                status = f"ПОЗДНО +{-wait:.2f}"
            else:
                status = f"OK +{wait:.2f}"
        if actual_front is None:
            status += " [нет метки камеры; срок от очереди]"
        rec = diag.current()
        if rec is not None and rec.session == session:
            try:
                rec.decision(obj.get('_diag_entry'), dict(number=number, track_id=obj.get('track_id'),
                    x0=obj['x0'], x1=obj['x1'], row0=obj['row0'], row1=obj['row1'], area=obj['area'],
                    cls=str(name), confidence=float(conf), parts=parts, latency_ms=lat,
                    status=status, valves=list(hit), event_id=event_id, due=due, decision=now))
            except Exception as e:
                rec.error(e)
        if LOG_OBJECTS:
            fmt = lambda k: f"{parts[k]:.2f}" if k in parts else "—"
            stages = " ".join(f"{k}={fmt(k)}" for k in
                ("Сбор", "Дет", "ВырОч", "Выр", "Оч", "NPU"))
            if PIPELINE_TIMING and PIPELINE_TIMING_EACH:
                stages += ' {' + ' '.join(f'{k}={fmt(k)}' for k in
                                       ("КамОч", "КамОчЗак", "Возраст", "Сним")) + '}'
            if NPU_DETAIL_EACH and npu_detail:
                stages += ' [' + ' '.join(f'{k}={fmt(k)}' for k in
                                      ("Подг", "RKNN", "Рез")) + ']'
            if abs(parts.get("Проч", 0)) >= .1:
                stages += f" Проч={fmt('Проч')}"
            total = f"{lat:.2f}" if actual_front is not None else "—"
            tail = f" | клапаны={_span(hit)} [{where}]" if LOG_POSITIONS else ""
            print(f"#{number:04d} | Кл={name} {conf*100:.1f}% | {stages} | "
                  f"Итог={total} мс | {status}{tail}")


def _worker():
    """Поток сети: грузит модель один раз, затем разбирает очередь."""
    global _classifier, _classifier_error
    try:
        _classifier = Classifier()
        print(f"сортировка: модель {_classifier.model}, классы "
              f"{_classifier.classes}, вход {_classifier.size}x"
              f"{_classifier.size}, вывод {_classifier.infer_ms:.1f} мс")
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
                    _decide(obj, received, session)
                except Exception as e:
                    with _lock:
                        if _active and session == _session:
                            _count("errors")
                    print(f"сортировка: ошибка при разборе предмета: {e!r}")



def _summary_worker():
    """Статистика отдельно от NPU: percentile не задерживает следующий crop."""
    global _summary_at, _stats, _pipeline_previous
    import line_camera
    while True:
        time.sleep(min(.25, SUMMARY_SEC))
        diag.poll_manual()
        snapshot = None
        now = time.monotonic()
        if now - _summary_at >= SUMMARY_SEC:
            with _lock:
                now = time.monotonic()
                if now - _summary_at >= SUMMARY_SEC:
                    if LOG_SUMMARY and _active and (_stats.seen or _stats.dropped or _stats.spi_errors or _stats.errors or _stats.at_edge or _stats.fired):
                        snapshot = (_session, now - _summary_at, _stats)
                    _stats = _Stats()
                    _summary_at = now
        if snapshot is not None:
            session, elapsed, stats = snapshot
            line = f"СВОДКА {elapsed:.1f} с | {stats.short()}"
            now_pipeline = line_camera.sorting_metrics() if PIPELINE_TIMING else None
            with _lock:
                if _active and session == _session:
                    if now_pipeline is not None and _pipeline_previous is not None:
                        line += "\n" + _pipeline_summary(_pipeline_previous, now_pipeline)
                        _pipeline_previous = now_pipeline
                    print(line)


def _firer():
    """
    Поток клапанов: ждёт момента и отправляет кадр. Команды, которые
    приходятся на одно и то же время с точностью до merge_ms, уходят
    одним кадром.
    """
    global _spi_inflight
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
            batch_dues = []
            batch_ids = []
            batch_session = _session
            while _events and _events[0][0] <= now + merge:
                event_due, event_id, hit, u, session = heapq.heappop(_events)
                if session == _session and _active:
                    batch.update(hit)
                    units = max(units, u)
                    batch_dues.append(event_due)
                    batch_ids.append(event_id)
            if batch:
                _spi_inflight += 1
        if batch:
            t_spi = time.monotonic()
            ok = False
            try:
                ok = valves.fire_many(sorted(batch), units)
            except Exception as exc:
                print(f"SPI ERROR: {exc!r}")
            finally:
                with _lock:
                    _spi_inflight -= 1
                    if _active and batch_session == _session:
                        _count("fired" if ok else "spi_errors")
                        if ok:
                            values = [(t_spi - d) * 1000 for d in batch_dues]
                            _stats.spi_late.extend(values)
                            _total.spi_late.extend(values)
            rec = diag.current()
            if rec is not None and rec.session == batch_session:
                try:
                    rec.spi(dict(event_ids=batch_ids, valves=sorted(batch), units=units,
                                 due=batch_dues, t_spi=t_spi, ok=bool(ok)))
                except Exception as e:
                    rec.error(e)
            if not ok:
                print("SPI ERROR: команда клапанам не отправлена")


def prepare():
    """При старте сервера: потоки сети и клапанов, загрузка модели."""
    if _threads:
        return
    for target, name in ((_worker, "сортировка"), (_firer, "клапаны"),
                         (_summary_worker, "сводки")):
        t = threading.Thread(target=target, daemon=True, name=name)
        t.start()
        _threads.append(t)