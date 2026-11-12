#!/usr/bin/env python3
"""
Проверки и калибровки линейной камеры - отдельно от рабочего кода
line_camera.py, чтобы тот не разрастался. Сервер при этом остановить:
камера одна.

    python3 camera_tools.py --calibrate          ВСЯ калибровка одной командой:
                                                 усиление, чёрное, выравнивание,
                                                 сама пишет config.json
    python3 camera_tools.py --selftest           детекция без железа
    python3 camera_tools.py --graph              живой график строки (нужен экран)
    python3 camera_tools.py                      снимать кадры с предметами в png
    python3 camera_tools.py --snap [сек]         что видит камера: snap.png, маска,
                                                 шумные столбцы, подсказка по полосе
    python3 camera_tools.py --exposure-sweep [--steps 50,100,..] [--trigger 0]
                                                 какую выдержку держит частота
    python3 camera_tools.py --gain-sweep [--steps ..]   подбор усиления
    python3 camera_tools.py --black-sweep        подбор уровня чёрного (объектив закрыт)
    python3 camera_tools.py --sensitivity-sweep  какие 8 бит из 12
    python3 camera_tools.py --color-info         настройки цвета камеры и чёрное
    python3 camera_tools.py --dark               чёрное по пикселям -> dark.npy
    python3 camera_tools.py --flat [--target 156]   выравнивание -> flat.npy

Старые вызовы через line_camera.py с этими ключами тоже работают.
Рабочие параметры (выдержка, частота, порог и прочее) берутся из
line_camera.py и config.json, здесь их нет.
"""

import os
import sys
import time
from ctypes import byref, c_float, c_int, create_string_buffer

import cv2
import numpy as np

import line_camera as lc
from line_camera import (
    DARK_FILE,
    EXPOSURE_MS,
    FIXED_RATE,
    FLAT_FILE,
    FLAT_PIECES,
    FRAME_HEIGHT,
    GAIN,
    GAP_ROWS,
    GRAPH_MAX,
    IDX,
    LINES,
    LineCamera,
    MEDIAN_KERNEL,
    MIN_ROW_PIXELS,
    ObjectDetector,
    SENSOR_ROWS,
    SENSOR_WIDTH,
    SKIP_PIECES,
    TH,
    X0,
    X1,
    _flat_settings,
    _gain_to_sdk,
    _row_median,
    arm,
    begin,
    failed,
    live_graph,
    ready,
    status,
    stop,
    take,
)


def selftest():
    """Проверяет детекцию, высоту кадра и центрирование без камеры."""
    rng = np.random.default_rng(0)
    w, ch = SENSOR_WIDTH, 3
    piece_rows = LINES * SENSOR_ROWS
    full_bg = np.full((w, ch), 90, np.uint8)

    def stream(det, specs):
        """specs: список (строк, есть ли объект). Возвращает кадры."""
        out = []
        for rows, obj, *rest in specs:
            drift = rest[0] if rest else 0
            done = 0
            while done < rows:
                n = min(piece_rows, rows - done)
                p = rng.integers(85, 96, size=(n, w, ch), dtype=np.int16)
                p += drift                      # общий дрейф подсветки
                if obj:
                    p[:, X0 + 150:X0 + 450] = rng.integers(
                        180, 230, size=(n, 300, ch))
                p = np.clip(p, 0, 255).astype(np.uint8)
                if p.shape[0] < piece_rows:     # порция всегда целая
                    pad = np.repeat(p[-1:], piece_rows - p.shape[0], axis=0)
                    p = np.vstack([p, pad])
                out.extend(det.feed(p))
                done += piece_rows
        return out

    def object_rows(frame):
        band = frame[:, X0:X1].astype(np.int16)
        d = band - full_bg[X0:X1].astype(np.int16)
        d -= np.median(d[:, ::8], axis=1, keepdims=True).astype(np.int16)
        fg = (np.abs(d) > TH).any(axis=2).sum(axis=1)
        rows = np.flatnonzero(fg >= 20)
        return (int(rows[0]), int(rows[-1])) if rows.size else None

    ok = True
    print(f"порция {piece_rows} строк, кадр {FRAME_HEIGHT} строк, "
          f"разрыв серии {GAP_ROWS} строк\n")

    for length in (64, 128, 256):
        det = ObjectDetector(full_bg, piece_rows)
        found = stream(det, [(512, False), (length, True), (1536, False)])
        if len(found) != 1:
            print(f"  объект {length:4d} строк: кадров {len(found)}, ожидался 1")
            ok = False
            continue
        frame = found[0]
        span = object_rows(frame)
        if span is None:
            print(f"  объект {length:4d} строк: в кадре его нет")
            ok = False
            continue
        top, bottom = span
        offset = (top + bottom) / 2 - FRAME_HEIGHT / 2
        good = (frame.shape == (FRAME_HEIGHT, w, ch) and abs(offset) <= 24)
        ok = ok and good
        print(f"  объект {length:4d} строк -> кадр {frame.shape[0]}x"
              f"{frame.shape[1]}, строки {top}..{bottom}, "
              f"смещение центра {offset:+.0f}  {'ок' if good else 'ПЛОХО'}")

    # Главная проверка: дрейф подсветки не должен приниматься за объект.
    det = ObjectDetector(full_bg, piece_rows)
    drifted = stream(det, [(512, False, 0), (1024, False, 25),
                           (1024, False, -25), (512, False, 0)])
    print(f"\n  дрейф подсветки на +-25 уровней: кадров {len(drifted)} "
          f"(ожидалось 0)")
    ok = ok and len(drifted) == 0

    # Шум без объекта тоже не должен срабатывать.
    det = ObjectDetector(full_bg, piece_rows)
    noise = stream(det, [(4096, False)])
    print(f"  чистый фон 4096 строк: кадров {len(noise)} (ожидалось 0)")
    ok = ok and len(noise) == 0

    # Два объекта подряд не должны потеряться и не должны задвоиться.
    det = ObjectDetector(full_bg, piece_rows)
    two = stream(det, [(512, False), (128, True), (768, False),
                       (128, True), (1536, False)])
    print(f"  два объекта подряд: кадров {len(two)} (ожидалось 2)")
    ok = ok and len(two) == 2

    print("\nдетектор:", "работает" if ok else "НЕ РАБОТАЕТ")
    return ok


def show_graph():
    """
    Живой график яркости строки, как в graph.py. Нужен дисплей.

    Берёт ту же строку, что уходит приложению, поэтому годится для
    настройки освещения и усиления: что видно здесь, то увидит и
    приложение в блоке 106.
    """
    import matplotlib.pyplot as plt

    begin()
    while not ready() and not failed():
        time.sleep(0.1)
    if failed():
        raise SystemExit(1)

    x = np.arange(SENSOR_WIDTH)
    plt.ion()
    fig, ax = plt.subplots()
    lines = [ax.plot(x, np.zeros(SENSOR_WIDTH), c + '-', label=n)[0]
             for c, n in (('r', 'R'), ('g', 'G'), ('b', 'B'))]
    ax.set_ylim(0, GRAPH_MAX)
    ax.set_xlim(0, SENSOR_WIDTH)
    ax.set_xlabel('Линия пикселей')
    ax.set_ylabel('Яркость')
    ax.legend()

    while plt.fignum_exists(fig.number):
        flat = live_graph()
        if flat is None:
            plt.pause(0.05)
            continue
        rgb = flat.reshape(SENSOR_WIDTH, 3)
        for i, ln in enumerate(lines):
            ln.set_ydata(rgb[:, i])
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.02)
    stop()


def exposure_sweep():
    """
    Какую выдержку камера выдерживает при FIXED_RATE:
        python3 camera_tools.py --exposure-sweep

    Сервер остановить - камера одна. Для каждой выдержки снимается 300
    порций и меряется, как часто они на самом деле приходят. Если камера не
    успевает, она пропускает считывания: порция приходит в 2-3 раза реже.
    Рабочая выдержка - самая большая, при которой частота держится.
    """
    def arg(name, default=None):
        if name in sys.argv:
            return int(sys.argv[sys.argv.index(name) + 1])
        return default

    cam = LineCamera()
    cam.open()
    ksj = cam.ksj

    # Служебное время сенсора на считывание: гашение по строке и по кадру
    # (параметры KSJ_HORIZONTALBLANK = 9, KSJ_VERTICALBLANK = 10) и ширина
    # считываемой области. Показываем, что стоит, и даём попробовать другое.
    names = {9: "гашение по строке (HORIZONTALBLANK)",
             10: "гашение по кадру (VERTICALBLANK)"}
    for code, flag in ((9, "--hblank"), (10, "--vblank")):
        lo, hi, now = c_int(), c_int(), c_int()
        rc1 = ksj.KSJ_GetParamRange(IDX, code, byref(lo), byref(hi))
        rc2 = ksj.KSJ_GetParam(IDX, code, byref(now))
        want = arg(flag)
        line = (f"  {names[code]}: сейчас {now.value}, диапазон "
                f"{lo.value}..{hi.value}" if rc1 >= 0 and rc2 >= 0
                else f"  {names[code]}: камера не отдаёт (коды {rc1}, {rc2})")
        if want is not None:
            rc = ksj.KSJ_SetParam(IDX, code, want)
            ksj.KSJ_GetParam(IDX, code, byref(now))
            line += f" -> ставлю {want}: код {rc}, теперь {now.value}"
        print(line)
    trig = arg("--trigger")
    if trig is not None:
        # 0 - камера снимает сама так быстро, как позволяет сенсор (своя
        # частота при данной выдержке), 3 - по FIXED_RATE.
        rc = ksj.KSJ_TriggerModeSet(IDX, trig)
        print(f"  режим запуска {trig}: код {rc}"
              + (" - камера снимает со своей предельной частотой"
                 if trig == 0 else ""))
    fr = c_float()
    if ksj.KSJ_GetFixedFrameRateEx(IDX, byref(fr)) >= 0:
        print(f"  частота в камере: {fr.value:.0f}")
    width = arg("--width")
    if width:
        # Только для замера: остальной код рассчитан на всю ширину 2048.
        start = max(0, min(SENSOR_WIDTH - width, (X0 + X1) // 2 - width // 2))
        rc = ksj.KSJ_CaptureSetFieldOfViewEx(IDX, start, 0, width,
                                             SENSOR_ROWS, 0, 0, LINES)
        w, h, bits = c_int(), c_int(), c_int()
        ksj.KSJ_CaptureGetSizeEx(IDX, byref(w), byref(h), byref(bits))
        cam.width, cam.height = w.value, h.value
        print(f"  ширина считывания {width} с {start}: код {rc}, камера "
              f"отдаёт {w.value}x{h.value}")

    buf = create_string_buffer(cam.width * cam.height * cam.channels)
    expect = LINES / FIXED_RATE * 1000
    period_us = 1e6 / FIXED_RATE
    print(f"\nFIXED_RATE = {FIXED_RATE:.0f}: на одно считывание "
          f"{period_us:.0f} мкс, порция должна приходить раз в "
          f"{expect:.1f} мс")
    print("выдержка, мкс | в камере, мкс | порция раз в, мс | считываний/с | "
          "держит | яркость ленты B G R")
    best = None
    try:
        steps = (50, 100, 125, 150, 175, 200, 210, 220, 230, 240, 250)
        if "--steps" in sys.argv:
            steps = [int(v) for v in
                     sys.argv[sys.argv.index("--steps") + 1].split(",")]
        for us in steps:
            if us > period_us:
                break
            ksj.KSJ_ExposureTimeSet(IDX, us / 1000)
            got = c_float()
            got_us = (got.value * 1000
                      if ksj.KSJ_ExposureTimeGet(IDX, byref(got)) >= 0
                      else float("nan"))
            for _ in range(SKIP_PIECES + 4):
                ksj.KSJ_CaptureRgbData(IDX, buf)
            t0 = time.perf_counter()
            n = 300
            for _ in range(n):
                ksj.KSJ_CaptureRgbData(IDX, buf)
            cycle = (time.perf_counter() - t0) / n * 1000
            piece = np.frombuffer(buf, np.uint8).reshape(
                cam.height, cam.width, cam.channels)
            band = piece[:, X0:X1] if cam.width == SENSOR_WIDTH else piece
            level = band.reshape(-1, 3).mean(axis=0)
            ok = cycle <= expect * 1.15
            if ok:
                best = us
            print(f"{us:13d} | {got_us:13.1f} | {cycle:16.2f} | "
                  f"{LINES / cycle * 1000:12.0f} "
                  f"| {'да' if ok else 'НЕТ':6s} | "
                  + " ".join(f"{v:3.0f}" for v in level))
    finally:
        cam.close()
    if best:
        print(f"\nсамая большая выдержка, при которой камера держит "
              f"{FIXED_RATE:.0f}: {best} мкс. Поставьте EXPOSURE_MS = "
              f"{best / 1000:g} (лучше с запасом на ступень ниже)")
    else:
        print("\nни при одной выдержке камера не держит FIXED_RATE")


def snapshot(seconds=3.0):
    """
    Что видит камера, без детекции: python3 camera_tools.py --snap [секунд]

    Сервер остановить. Камера снимает фон (лента пустая), затем по Enter
    пишет seconds секунд - в это время бросьте несколько предметов. Из
    записи берётся кусок 1024 строки там, где предметов больше всего:
        snap.png       - картинка, как её видит детектор (с выравниванием,
                         если оно включено);
        snap_mask.png  - белым то, что детектор счёл отличием от фона
                         при нынешнем пороге TH.
    И печатается, насколько предметы отличаются от фона и насколько шумит
    пустая лента - по этому видно, какой порог TH поставить.
    """
    lc._detect_enabled = False
    begin()
    while not ready() and not failed():
        time.sleep(0.1)
    if failed():
        raise SystemExit(1)
    cam = lc._camera
    while cam.detector is None:
        time.sleep(0.05)
    bg = cam.detector.full_bg.astype(np.int16)
    input(f"\nНажмите Enter и сразу бросайте предметы - пишу {seconds:g} с...")
    cam.tap_max = int(seconds / cam.piece_sec)
    cam.tap = []
    while len(cam.tap) < cam.tap_max:
        time.sleep(0.05)
    pieces, cam.tap = cam.tap, None
    process = cam.process
    stop()

    # Те же поправки, что в работе: чёрное, выравнивание, матрица.
    pieces = [process(p_) for p_ in pieces]
    flat = cam.flat2
    rows = np.concatenate(pieces)
    # То же отличие от фона, что считает детектор: по каналам, за вычетом
    # общего сдвига строки, максимум по трём каналам.
    d = rows[:, X0:X1].astype(np.int16) - bg[X0:X1]
    d -= _row_median(d[:, ::8])
    np.abs(d, out=d)
    dm = np.maximum(np.maximum(d[..., 0], d[..., 1]), d[..., 2])
    busy = (dm > TH).sum(axis=1)

    print(f"\nзаписано {rows.shape[0]} строк ({seconds:g} с); порог TH = {TH}, "
          f"строка занята от {MIN_ROW_PIXELS} пикселей")
    level = rows[:, X0:X1].reshape(-1, 3).mean(axis=0)
    print("яркость ленты B G R: " + " ".join(f"{v:.0f}" for v in level)
          + (" (с выравниванием)" if flat is not None else ""))
    empty = dm[busy < MIN_ROW_PIXELS]
    full = busy >= MIN_ROW_PIXELS
    print(f"занятых строк: {full.sum()} из {rows.shape[0]}")
    if empty.size:
        print(f"шум пустой ленты: 99% пикселей отличаются от фона меньше чем "
              f"на {np.percentile(empty, 99):.0f}, 99.9% - на "
              f"{np.percentile(empty, 99.9):.0f}")
    # Предметы: самые сильные пиксели в самых занятых строках.
    top = np.argsort(busy)[-64:]
    obj = dm[top]
    strong = obj[obj > np.percentile(obj, 90)]
    if busy.max() >= MIN_ROW_PIXELS and strong.size:
        print(f"предметы: самые заметные пиксели отличаются от фона на "
              f"{np.median(strong):.0f} (медиана), в целом по строкам с "
              f"предметами - на {np.percentile(obj, 75):.0f} и больше у "
              f"четверти пикселей")
    else:
        print(f"ни одной занятой строки: при пороге TH = {TH} предметы от "
              f"фона не отличаются. Самые сильные отличия во всей записи: "
              f"{np.percentile(dm, 99.99):.0f} (0.01% пикселей), "
              f"{np.percentile(dm, 99.9):.0f} (0.1%) - если это предметы, "
              f"порог надо опустить ниже; посмотрите snap.png")
    # Столбцы, которые "заняты" почти всегда - это не предметы, а шумный
    # край за освещённой лентой. Если они в полосе, серия не кончается.
    always = np.flatnonzero((dm > TH).mean(axis=0) > 0.05) + X0
    if always.size:
        parts, start = [], always[0]
        for a_, b_ in zip(always, always[1:]):
            if b_ - a_ > 20:
                parts.append((start, a_))
                start = b_
        parts.append((start, always[-1]))
        print("ВНИМАНИЕ: столбцы, которые отличаются от фона почти всё время: "
              + ", ".join(f"{a_}..{b_}" for a_, b_ in parts)
              + f". Если из-за них занята каждая строка ({full.mean():.0%} "
              f"строк заняты), детектор не захватит ни одного предмета - "
              f"сузьте полосу: camera.band в config (сейчас {X0}..{X1})")
    # Освещённая лента - самый длинный сплошной участок, где яркость
    # столбца выше половины наибольшей (сглажено по 31 столбцу).
    lit = rows.mean(axis=0).sum(axis=1)
    lit = np.convolve(lit, np.ones(31) / 31, "same")
    on = (lit > lit.max() * 0.5).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.r_[0, on, 0]))
    if edges.size >= 2:
        runs = list(zip(edges[0::2], edges[1::2]))
        a_, b_ = max(runs, key=lambda t: t[1] - t[0])
        lo, hi = a_ + 15, b_ - 15
        # Шумные столбцы внутри ленты тоже не берём: от середины ленты
        # до ближайшего шумного столбца с каждой стороны.
        mid = (lo + hi) // 2
        left = [c for c in always if lo <= c < mid]
        right = [c for c in always if mid <= c < hi]
        if left:
            lo = max(left) + 10
        if right:
            hi = min(right) - 10
        print(f"освещённая лента на снимке примерно {a_}..{b_}; полоса "
              f"детектора без шумных краёв: camera.band = [{lo}, {hi}]")
    sat = (rows[:, X0:X1] >= 250).mean(axis=(0, 1)) * 100
    if (sat > 20).any():
        print("ВНИМАНИЕ: каналы упираются в 255: "
              + " ".join(f"{n_} {v:.0f}%" for n_, v in zip("BGR", sat))
              + " пикселей полосы. В этом канале предметы светлее ленты не "
              "отличить. Если включено выравнивание - flat.npy снят при "
              "другой выдержке, снимите заново (--flat)")
    noise = np.percentile(empty, 99.9) if empty.size else 0
    print(f"порог TH должен быть выше шума ({noise:.0f}) и заметно ниже того, "
          f"насколько отличаются предметы; сейчас TH = {TH}")

    k = int(np.argmax(np.convolve(busy, np.ones(256), "same")))
    a = max(0, min(rows.shape[0] - 1024, k - 512))
    out = rows[a:a + 1024]
    mask = np.zeros(out.shape[:2], np.uint8)
    mask[:, X0:X1] = (dm[a:a + 1024] > TH) * 255
    here = os.path.dirname(os.path.abspath(__file__))
    if MEDIAN_KERNEL > 1:
        cv2.imwrite(os.path.join(here, "snap_raw.png"), out)
        out = cv2.medianBlur(np.ascontiguousarray(out), MEDIAN_KERNEL)
    cv2.imwrite(os.path.join(here, "snap.png"), out)
    cv2.imwrite(os.path.join(here, "snap_mask.png"), mask)
    print(f"сохранены snap.png (с медианой {MEDIAN_KERNEL}, как снимок в "
          f"приложении), snap_raw.png (без неё) и snap_mask.png "
          f"(строки {a}..{a + 1024})")


def _measure(ksj, buf, cam, n=60):
    """n порций подряд (после 4 на прогрев) -> строки полосы, (k, 3)."""
    for _ in range(SKIP_PIECES):
        ksj.KSJ_CaptureRgbData(IDX, buf)
    out = []
    for _ in range(n):
        ksj.KSJ_CaptureRgbData(IDX, buf)
        out.append(np.frombuffer(buf, np.uint8).reshape(
            cam.height, cam.width, cam.channels)[:, X0:X1].copy())
    return np.concatenate(out).reshape(-1, 3)


def gain_sweep():
    """
    Подбор усиления: python3 camera_tools.py --gain-sweep [--steps 40,60,..]

    Сервер остановить, лента пустая, объектив открыт, выравнивание и прочие
    поправки на картинку не влияют - мерится сырое. Для каждого усиления
    (0..255, больше - ярче): яркость ленты в центре и у краёв, сколько
    пикселей в 255 и шум от строки к строке. Нужное - самое большое, при
    котором самый яркий канал в центре ещё не выше ~220 и в 255 почти
    ничего.
    """
    cam = LineCamera()
    cam.open()
    ksj = cam.ksj
    buf = create_string_buffer(cam.width * cam.height * cam.channels)
    steps = list(range(0, 256, 20)) + [255]
    if "--steps" in sys.argv:
        steps = [int(v) for v in
                 sys.argv[sys.argv.index("--steps") + 1].split(",")]
    segs = c_int()
    ksj.KSJ_AWAIBA_GetSegmentNum(IDX, byref(segs))
    w = X1 - X0
    print(f"выдержка {EXPOSURE_MS * 1000:.0f} мкс, частота {FIXED_RATE:.0f}")
    print("усиление | центр B G R | края B G R | в 255, % | шум B G R")
    best = None
    try:
        for g in steps:
            for sg in range(segs.value):
                ksj.KSJ_AWAIBA_SetGain(IDX, sg, _gain_to_sdk(g))
            for _ in range(SKIP_PIECES):
                ksj.KSJ_CaptureRgbData(IDX, buf)
            rows = []
            for _ in range(60):
                ksj.KSJ_CaptureRgbData(IDX, buf)
                rows.append(np.frombuffer(buf, np.uint8).reshape(
                    cam.height, cam.width, cam.channels)[:, X0:X1].copy())
            st = np.concatenate(rows).astype(np.float32)
            centre = st[:, w // 3:2 * w // 3].mean(axis=(0, 1))
            edges = np.concatenate([st[:, :w // 10], st[:, -w // 10:]],
                                   axis=1).mean(axis=(0, 1))
            sat = (st >= 255).mean() * 100
            noise = st[0::2].std(axis=0).mean(axis=0)
            print(f"{g:8d} | " + " ".join(f"{v:3.0f}" for v in centre)
                  + " | " + " ".join(f"{v:3.0f}" for v in edges)
                  + f" | {sat:8.1f} | "
                  + " ".join(f"{v:4.1f}" for v in noise))
            if centre.max() <= 220 and sat < 1 and (best is None or g > best):
                best = g
    finally:
        cam.close()
    if best is not None:
        print(f"\nсамое большое усиление без пересвета: {best}. В config.json, "
              f"раздел camera: \"gain\": {best}. Затем заново --black-sweep, "
              f"--dark и --flat")


def black_sweep():
    """
    Подбор уровня чёрного: python3 camera_tools.py --black-sweep

    Сервер остановить, объектив ЗАКРЫТЬ. Для каждого значения уровня
    чёрного (одно на оба сегмента) меряется, что камера даёт в темноте.
    Нужное значение - при котором чёрное чуть выше нуля (около 4-8) и
    нулей почти нет: тогда тёмные оттенки не обрезаются, а диапазон для
    светлых почти не теряется.
    """
    cam = LineCamera()
    cam.open()
    ksj = cam.ksj
    buf = create_string_buffer(cam.width * cam.height * cam.channels)
    input("\nЗакройте объектив крышкой и нажмите Enter...")
    segs = c_int()
    ksj.KSJ_AWAIBA_GetSegmentNum(IDX, byref(segs))
    print("уровень | чёрное B G R (среднее) | нулей B G R, %")
    print("(чем больше уровень, тем темнее чёрное)")
    good = []
    try:
        for level in range(0, 256, 8):
            for sg in range(segs.value):
                ksj.KSJ_AWAIBA_SetBlackLevel(IDX, sg, level)
            d = _measure(ksj, buf, cam)
            mean = d.mean(axis=0)
            zero = (d == 0).mean(axis=0) * 100
            print(f"{level:7d} | " + " ".join(f"{v:5.1f}" for v in mean)
                  + "     | " + " ".join(f"{v:5.1f}" for v in zero))
            # По самому тёмному каналу: он первым уходит в 0. Остальные
            # каналы при этом выше - их смещение уберёт вычитание чёрного.
            if (zero < 1).all() and mean.min() >= 3:
                good.append((abs(mean.min() - 6), level))
    finally:
        cam.close()
    if good:
        best = min(good)[1]
        print(f"\nподходит уровень {best}: в config.json, раздел camera - "
              f"\"black_level\": {best}. Затем заново --dark и --flat")
    else:
        print("\nни одно значение не дало чёрное 4-15 без нулей - пришлите "
              "таблицу")


def sensitivity_sweep():
    """
    Какие 8 бит из 12 брать: python3 camera_tools.py --sensitivity-sweep

    Сервер остановить, лента пустая, объектив открыт. Камера оцифровывает
    12 бит, а отдаёт 8; режим чувствительности выбирает, какие именно.
    Более чувствительный режим - это яркость без увеличения выдержки (то
    есть без потери частоты) и больше уровней в тёмных тонах; цена -
    раньше упирается в 255.
    """
    cam = LineCamera()
    cam.open()
    ksj = cam.ksj
    if not hasattr(ksj, "KSJ_SensitivitySetMode"):
        cam.close()
        raise SystemExit("в этой версии SDK нет KSJ_SensitivitySetMode")
    buf = create_string_buffer(cam.width * cam.height * cam.channels)
    expect = LINES / FIXED_RATE * 1000
    print("режим | код | лента B G R | в 255, % | порция раз в, мс")
    try:
        for mode in range(5):
            rc = ksj.KSJ_SensitivitySetMode(IDX, mode)
            t0 = time.perf_counter()
            d = _measure(ksj, buf, cam, n=100)
            cycle = (time.perf_counter() - t0) / (100 + SKIP_PIECES) * 1000
            sat = (d >= 255).mean() * 100
            print(f"{mode:5d} | {rc:3d} | " + " ".join(f"{v:3.0f}" for v in
                                                    d.mean(axis=0))
                  + f" | {sat:8.1f} | {cycle:.2f}"
                  + ("" if cycle <= expect * 1.15 else "  не держит"))
    finally:
        cam.close()
    print("\nВыберите режим, где лента ярче, но в 255 почти ничего: "
          "\"sensitivity\": N в разделе camera. Затем заново --dark и --flat")


def calibrate_dark():
    """
    Чёрное камеры: python3 camera_tools.py --dark

    Сервер остановить. Закрыть объектив крышкой (свет можно не трогать).
    По каждому столбцу, каналу и ряду сенсора - средний уровень. Потом
    включить camera.dark_subtract и заново снять выравнивание (--flat).
    """
    lc._detect_enabled = False
    begin()
    while not ready() and not failed():
        time.sleep(0.1)
    if failed():
        raise SystemExit(1)
    input("\nЗакройте объектив крышкой и нажмите Enter...")
    cam = lc._camera
    cam.tap = []
    while len(cam.tap) < FLAT_PIECES:
        time.sleep(0.05)
    pieces, cam.tap = cam.tap, None
    stop()
    stack = np.concatenate(pieces).astype(np.float32)
    dark = np.stack([stack[0::2].mean(axis=0), stack[1::2].mean(axis=0)])
    np.save(DARK_FILE, dark.astype(np.float32))
    band = stack[:, X0:X1].reshape(-1, 3)
    print(f"\nсохранено {DARK_FILE}")
    print("чёрное B G R: среднее "
          + " ".join(f"{v:.1f}" for v in band.mean(axis=0))
          + ", нулей % "
          + " ".join(f"{v:.0f}" for v in (band == 0).mean(axis=0) * 100))
    if ((band == 0).mean(axis=0) > 0.3).any():
        print("ВНИМАНИЕ: чёрное в каком-то канале во многом обрезано в 0 - "
              "тёмные оттенки там теряются ещё в камере, вычитание этого не "
              "вернёт. Уровень чёрного камеры надо поднять (BLACK_LEVEL)")
    if band.mean() > 25:
        print("ВНИМАНИЕ: чёрное слишком светлое - объектив закрыт неплотно?")
    print("Дальше: camera.dark_subtract = true и заново --flat")


def color_info():
    """
    Что камера делает с цветом: python3 camera_tools.py --color-info

    Только читает настройки, ничего не меняет. Сервер остановить. Затем
    просит закрыть объектив (или выключить свет) и меряет уровень чёрного
    по каналам - от него зависит, как передаются тёмные предметы.
    """
    cam = LineCamera()
    cam.open()
    ksj = cam.ksj
    v = c_int()

    def q(code, name):
        rc = ksj.KSJ_QueryFunction(IDX, code, byref(v))
        print(f"  {name}: {v.value if rc >= 0 else 'нет ответа'}")

    print("\nвозможности камеры (KSJ_QueryFunction):")
    q(1, "разрядность АЦП, бит")
    q(23, "передача 16 бит на пиксель (1 - есть)")
    q(36, "аппаратный баланс белого, матрица (1 - есть)")
    q(37, "аппаратная цветовая матрица CCM (1 - есть)")
    q(32, "аппаратная гамма (1 - есть)")
    q(22, "выбор 8 бит из 12 - чувствительность (1 - есть)")

    print("\nнастройки цвета сейчас:")
    wb_names = ["выключен", "программный, пресет", "программный, авто раз",
                "программный, авто постоянно", "программный, вручную",
                "аппаратный, пресет", "аппаратный, авто раз",
                "аппаратный, авто постоянно", "аппаратный, вручную"]
    if ksj.KSJ_WhiteBalanceGet(IDX, byref(v)) >= 0:
        print(f"  баланс белого: {wb_names[v.value] if 0 <= v.value < 9 else v.value}")
    m3 = (c_float * 3)()
    if ksj.KSJ_WhiteBalanceMatrixGet(IDX, m3) >= 0:
        print("  множители баланса белого R G B: "
              + " ".join(f"{x:.3f}" for x in m3))
    ccm_names = ["выключена", "программная, пресет", "программная, вручную",
                 "аппаратная, пресет", "аппаратная, вручную"]
    if ksj.KSJ_ColorCorrectionGet(IDX, byref(v)) >= 0:
        print(f"  цветовая матрица: "
              f"{ccm_names[v.value] if 0 <= v.value < 5 else v.value}")
    m9 = (c_float * 9)()
    if ksj.KSJ_ColorCorrectionMatrixGet(IDX, m9) >= 0:
        for r_ in range(3):
            print("    " + " ".join(f"{m9[r_ * 3 + c_]:7.3f}" for c_ in range(3)))
    if ksj.KSJ_GetParam(IDX, 4, byref(v)) >= 0:
        print(f"  гамма (KSJ_GAMMA): {v.value}")
    segs = c_int()
    if ksj.KSJ_AWAIBA_GetSegmentNum(IDX, byref(segs)) >= 0:
        for sg in range(segs.value):
            g, b_ = c_int(), c_int()
            ksj.KSJ_AWAIBA_GetGain(IDX, sg, byref(g))
            ksj.KSJ_AWAIBA_GetBlackLevel(IDX, sg, byref(b_))
            lo, hi, blo, bhi = c_int(), c_int(), c_int(), c_int()
            ksj.KSJ_AWAIBA_GetGainRange(IDX, sg, byref(lo), byref(hi))
            ksj.KSJ_AWAIBA_GetBlackLevelRange(IDX, sg, byref(blo), byref(bhi))
            print(f"  сегмент {sg}: усиление {g.value} (диапазон {lo.value}.."
                  f"{hi.value}; камера показывает 255 - заданное, задано "
                  f"{_flat_settings().get('gain') or GAIN}), уровень чёрного "
                  f"{b_.value} (диапазон {blo.value}..{bhi.value})")

    buf = create_string_buffer(cam.width * cam.height * cam.channels)
    input("\nЗакройте объектив крышкой (или выключите свет) и нажмите "
          "Enter...")
    rows = []
    for _ in range(SKIP_PIECES + 100):
        ksj.KSJ_CaptureRgbData(IDX, buf)
        rows.append(np.frombuffer(buf, np.uint8).reshape(
            cam.height, cam.width, cam.channels).copy())
    cam.close()
    dark = np.concatenate(rows[SKIP_PIECES:])[:, X0:X1].reshape(-1, 3)
    print("чёрное по каналам B G R:")
    print("  среднее " + " ".join(f"{x:5.1f}" for x in dark.mean(axis=0)))
    print("  доля нулей, % " + " ".join(f"{x:5.1f}" for x in
                                       (dark == 0).mean(axis=0) * 100))
    zero = (dark == 0).mean(axis=0)
    if (zero > 0.3).any():
        print("ВНИМАНИЕ: чёрное в каком-то канале обрезано в 0 - тёмные "
              "оттенки в нём пропадают, и тёмные предметы теряют цвет. "
              "Уровень чёрного надо поднять, чтобы чёрное было около 4-8")


def calibrate_flat(target=None):
    """
    Калибровка ровной яркости: python3 camera_tools.py --flat [--target 156]

    Сервер на это время остановить - камера одна. Под камерой - пустая
    ровная лента (серая или та, что стоит), без предметов и теней. Выдержку
    и свет выставить заранее: калибровка верна только для них.

    По каждому столбцу и каналу берётся медиана яркости ленты, и
    коэффициент = цель / медиана. После этого пустая лента везде даёт
    цель (156): на графике - прямая линия. При flat_per_channel = true
    каналы выравниваются и между собой: серая лента станет серой
    (156, 156, 156). При false - один коэффициент на столбец, по средней
    яркости, цвет ленты не меняется.
    """
    cfg = _flat_settings()
    target = float(target if target is not None else cfg["flat_target"])
    top = float(cfg["flat_max_gain"])
    per_channel = bool(cfg["flat_per_channel"])

    # Детекция при калибровке не нужна: она только сыпала бы в консоль
    # "объект N" и отнимала процессор у захвата.
    lc._detect_enabled = False
    begin()
    while not ready() and not failed():
        time.sleep(0.1)
    if failed():
        raise SystemExit(1)
    input(f"\nПод камерой должна быть пустая ровная лента. Цель - {target:g}."
          f" Нажмите Enter...")
    cam = lc._camera
    cam.tap = []
    while len(cam.tap) < FLAT_PIECES:
        time.sleep(0.05)
    pieces, cam.tap = cam.tap, None
    stop()

    # Порции чётной высоты, поэтому чётные строки всего набора - один ряд
    # сенсора, нечётные - другой. У рядов разная чувствительность (отсюда
    # была "сетка" на снимке), поэтому коэффициенты - свои на каждый ряд.
    stack = np.concatenate(pieces[::2])                  # сырые строки
    rows = [stack[0::2], stack[1::2]]
    raws = [np.median(r_, axis=0).astype(np.float32) for r_ in rows]
    if cfg.get("dark_subtract") and os.path.exists(DARK_FILE):
        # Выравнивание считается по ленте за вычетом чёрного - так же,
        # как оно потом применяется к порциям.
        dark = np.load(DARK_FILE).astype(np.float32)
        raws = [np.maximum(raw_ - dark[i], 0) for i, raw_ in enumerate(raws)]
        print("чёрное вычтено (dark.npy)")
    gains = []
    for raw_ in raws:
        if per_channel:
            level = raw_
        else:
            level = np.repeat(raw_.mean(axis=1, keepdims=True), 3, axis=1)
        g = target / np.maximum(level, 1.0)
        g[level < 8] = 1.0          # за краями ленты темноту не усиливаем
        gains.append(np.clip(g, 1 / top, top).astype(np.float32))
    gain = np.stack(gains)                               # 2, 2048, 3
    np.save(FLAT_FILE, gain)

    raw = (raws[0] + raws[1]) / 2
    after = np.clip((raws[0] * gains[0] + raws[1] * gains[1]) / 2, 0, 255)
    names = "BGR"
    band, band_after = raw[X0:X1], after[X0:X1]
    n = (X1 - X0) // 10

    def three(a):
        return [a[:n].mean(axis=0), a[(X1 - X0) // 2 - n // 2:
                                       (X1 - X0) // 2 + n // 2].mean(axis=0),
                a[-n:].mean(axis=0)]

    print(f"\nсохранено {FLAT_FILE}")
    print(f"полоса {X0}..{X1}, яркость ленты B G R: левый край / центр / "
          f"правый край")
    for label, arr in (("было ", band), ("стало", band_after)):
        print(f"  {label}: " + "  /  ".join(
            " ".join(f"{v:3.0f}" for v in t) for t in three(arr)))
    for i, nm in enumerate(names):
        spread = band_after[:, i].max() - band_after[:, i].min()
        g = gain[:, X0:X1, i]
        print(f"  {nm}: поправка x{g.min():.2f}..x{g.max():.2f}, после неё "
              f"разброс {spread:.0f} уровня")
    print("ряды сенсора и шум, B G R:")
    diff = (np.abs(raws[0][X0:X1] - raws[1][X0:X1]).mean(axis=0)
            / np.maximum(raw[X0:X1].mean(axis=0), 1) * 100)
    print("  ряды отличаются друг от друга: "
          + " ".join(f"{nm} {d:.1f}%" for nm, d in zip(names, diff))
          + " (теперь выравнивается)")
    noise = [(r_[:, X0:X1].astype(np.float32).std(axis=0).mean(axis=0)
              * g_[X0:X1].mean(axis=0)) for r_, g_ in zip(rows, gains)]
    noise = (noise[0] + noise[1]) / 2
    print("  шум от строки к строке после поправки: "
          + " ".join(f"{nm} ±{v:.1f}" for nm, v in zip(names, noise))
          + " уровня")
    if (band >= 250).mean() > 0.02:
        print("ВНИМАНИЕ: лента местами упирается в 255 - там поправка не "
              "сработает. Уменьшите выдержку или свет и снимите заново")
    if gain[:, X0:X1].max() >= top * 0.99:
        print(f"ВНИМАНИЕ: где-то поправка упёрлась в предел x{top:g} - там "
              f"света слишком мало. Добавьте свет к краям или снизьте цель")
    elif gain[:, X0:X1].max() > 2:
        print("Заметка: края усиливаются больше чем вдвое - шум там тоже "
              "вырастет вдвое. Лучше подтянуть свет к краям")
    print("\nДальше: camera.flat_field = true в config и перезапуск сервера. "
          "Фон при старте пересоберётся сам - лента должна быть пустой")


def _grab(ksj, buf, cam, n):
    """n порций подряд, сырых, перевёрнутых как в работе: (строки, 2048, 3)."""
    for _ in range(SKIP_PIECES):
        ksj.KSJ_CaptureRgbData(IDX, buf)
    out = []
    for _ in range(n):
        ksj.KSJ_CaptureRgbData(IDX, buf)
        p_ = np.frombuffer(buf, np.uint8).reshape(cam.height, cam.width,
                                                   cam.channels)
        out.append((p_[::-1] if lc.FLIP_PIECES else p_).copy())
    return np.concatenate(out)


def _profile_png(path, raw, done, target):
    """Два графика яркости по ширине: до и после поправок."""
    h, w = 300, 1024
    img = np.full((2 * h + 30, w, 3), 255, np.uint8)
    colors = [(255, 0, 0), (0, 160, 0), (0, 0, 255)]          # B G R
    for k, (prof, title) in enumerate(((raw, "до"), (done, "после"))):
        top = k * (h + 30)
        y = top + h - int(target / 255 * (h - 10))
        cv2.line(img, (0, y), (w, y), (180, 180, 180), 1)
        cv2.line(img, (X0 * w // SENSOR_WIDTH, top),
                 (X0 * w // SENSOR_WIDTH, top + h), (200, 200, 200), 1)
        cv2.line(img, (X1 * w // SENSOR_WIDTH, top),
                 (X1 * w // SENSOR_WIDTH, top + h), (200, 200, 200), 1)
        xs = np.arange(SENSOR_WIDTH) * w // SENSOR_WIDTH
        for c in range(3):
            ys = top + h - (prof[:, c] / 255 * (h - 10)).astype(int)
            pts = np.stack([xs, ys], 1).reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(img, [pts], False, colors[c], 1)
        cv2.putText(img, "before" if k == 0 else "after", (8, top + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.imwrite(path, img)


def _save_camera_config(values):
    """Записать найденное в раздел camera файла config.json."""
    import json
    import shutil
    from valve_config import CONFIG_JSON, CONFIG_YAML
    if os.path.exists(CONFIG_YAML):
        print(f"\nесть {CONFIG_YAML} - его я не переписываю (там ваши "
              f"комментарии). Перенесите в раздел camera:")
        for k, v in values.items():
            print(f"  {k}: {json.dumps(v)}")
        return
    data = {}
    if os.path.exists(CONFIG_JSON):
        with open(CONFIG_JSON, encoding="utf-8") as f:
            data = json.load(f)
        shutil.copy(CONFIG_JSON, CONFIG_JSON + ".bak")
    cam = data.setdefault("camera", {})
    cam.update(values)
    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nзаписано в {CONFIG_JSON} (старый - config.json.bak), раздел "
          f"camera: " + ", ".join(f"{k} = {v}" for k, v in values.items()))


def calibrate():
    """
    Вся калибровка одной командой: python3 camera_tools.py --calibrate

    Сервер остановить. Выдержка и частота - из line_camera.py, свет,
    диафрагма и фокус выставлены. Три подсказки:
      1) лента пустая, объектив открыт - подбирается усиление;
      2) объектив закрыт крышкой - уровень чёрного и чёрное по пикселям;
      3) лента пустая, объектив открыт - выравнивание и проверка.
    Всё найденное записывается в config.json само (раздел camera), график
    до/после - в calib.png. Цветовая матрица не трогается.
    """
    cfg = lc._flat_settings()
    target = float(cfg.get("flat_target") or 156)
    if "--target" in sys.argv:
        target = float(sys.argv[sys.argv.index("--target") + 1])
    per_channel = bool(cfg.get("flat_per_channel", True))
    top = float(cfg.get("flat_max_gain") or 4.0)

    cam = lc.LineCamera()
    cam.open()
    ksj = cam.ksj
    buf = create_string_buffer(cam.width * cam.height * cam.channels)
    segs = c_int()
    ksj.KSJ_AWAIBA_GetSegmentNum(IDX, byref(segs))
    nseg = max(1, segs.value)

    def set_gain(g):
        for sg in range(nseg):
            ksj.KSJ_AWAIBA_SetGain(IDX, sg, _gain_to_sdk(g))

    def set_black(b):
        for sg in range(nseg):
            ksj.KSJ_AWAIBA_SetBlackLevel(IDX, sg, int(b))

    w = X1 - X0
    try:
        # 0. Держит ли камера частоту при этой выдержке.
        t0 = time.perf_counter()
        _grab(ksj, buf, cam, 100)
        cycle = (time.perf_counter() - t0) / (100 + SKIP_PIECES) * 1000
        expect = LINES / FIXED_RATE * 1000
        print(f"\nвыдержка {EXPOSURE_MS * 1000:.0f} мкс, частота "
              f"{FIXED_RATE:.0f}: порция раз в {cycle:.2f} мс (нужно "
              f"{expect:.2f})")
        if cycle > expect * 1.15:
            raise SystemExit("камера не держит частоту при этой выдержке - "
                             "уменьшите EXPOSURE_MS в line_camera.py")

        # 1. Усиление: самое большое без пересвета ленты в центре.
        input("\n[1/3] Лента пустая, объектив ОТКРЫТ. Enter...")
        black = int(cfg.get("black_level") or 124)
        set_black(black)
        lo, hi, gain = 0, 255, 0
        while lo <= hi:
            g = (lo + hi) // 2
            set_gain(g)
            st = _grab(ksj, buf, cam, 30)[:, X0:X1]
            centre = st[:, w // 3:2 * w // 3].reshape(-1, 3).mean(axis=0)
            sat = (st >= 250).mean()
            if centre.max() <= 215 and sat < 0.005:
                gain, lo = g, g + 1
            else:
                hi = g - 1
        set_gain(gain)
        print(f"  усиление {gain}")

        # 2. Чёрное: самый большой уровень, при котором самый тёмный канал
        # в темноте ещё не ниже 4 (без нулей), и чёрное по пикселям.
        input("\n[2/3] Закройте объектив крышкой. Enter...")
        lo, hi, black = 0, 255, 0
        while lo <= hi:
            b = (lo + hi) // 2
            set_black(b)
            d = _grab(ksj, buf, cam, 30)[:, X0:X1].reshape(-1, 3)
            if d.mean(axis=0).min() >= 4 and (d == 0).mean() < 0.01:
                black, lo = b, b + 1
            else:
                hi = b - 1
        set_black(black)
        d = _grab(ksj, buf, cam, FLAT_PIECES).astype(np.float32)
        dark = np.stack([d[0::2].mean(axis=0), d[1::2].mean(axis=0)])
        np.save(lc.DARK_FILE, dark)
        dm = dark[:, X0:X1].mean(axis=(0, 1))
        print(f"  уровень чёрного {black}, чёрное B G R "
              + " ".join(f"{v:.1f}" for v in dm) + f" -> {lc.DARK_FILE}")

        # 3. Выравнивание по пустой ленте и проверка.
        input("\n[3/3] Откройте объектив, лента пустая. Enter...")
        st = _grab(ksj, buf, cam, FLAT_PIECES).astype(np.float32)
        raws = [np.median(st[0::2], axis=0), np.median(st[1::2], axis=0)]
        gains = []
        for i, raw_ in enumerate(raws):
            lvl = np.maximum(raw_ - dark[i], 0)
            if not per_channel:
                lvl = np.repeat(lvl.mean(axis=1, keepdims=True), 3, axis=1)
            g = target / np.maximum(lvl, 1.0)
            g[lvl < 8] = 1.0
            gains.append(np.clip(g, 1 / top, top).astype(np.float32))
        flat = np.stack(gains)
        np.save(lc.FLAT_FILE, flat)

        cam.dark2 = np.clip(np.round(dark), 0, 255).astype(np.uint8)
        cam.flat2 = flat
        cam._tiles = {}
        check = _grab(ksj, buf, cam, 60)
        done = cam.process(check)
        raw_prof = check.mean(axis=0)
        done_prof = done.astype(np.float32).mean(axis=0)
        _profile_png(os.path.join(os.path.dirname(os.path.abspath(
            __file__)), "calib.png"), raw_prof, done_prof, target)

        band = done[:, X0:X1].astype(np.float32)
        colm = band.mean(axis=0)                            # столбцы, 3
        names = "BGR"
        print(f"\nрезультат на пустой ленте (полоса {X0}..{X1}), цель "
              f"{target:g}:")
        ok = True
        for i, nm in enumerate(names):
            spread = colm[:, i].max() - colm[:, i].min()
            checker = np.abs(colm[0::2, i] - colm[1::2, i]).mean()
            rows_ = band[:, :, i].mean(axis=1)
            rowalt = np.abs(rows_[0::2] - rows_[1::2]).mean()
            noise = band[0::2, :, i].std(axis=0).mean()
            sat = (done[:, X0:X1, i] >= 255).mean() * 100
            print(f"  {nm}: среднее {colm[:, i].mean():5.1f}, по ширине "
                  f"{colm[:, i].min():.0f}..{colm[:, i].max():.0f} (разброс "
                  f"{spread:.0f}), шахматка {checker:.1f}/{rowalt:.1f}, шум "
                  f"±{noise:.1f}, в 255 {sat:.1f}%")
            if spread > 12 or sat > 0.5:
                ok = False
        g_band = flat[:, X0:X1]
        print(f"  поправка x{g_band.min():.2f}..x{g_band.max():.2f}")
        print("  " + ("ровно" if ok else
                      "НЕ РОВНО: разброс больше 12 уровней или есть 255 - "
                      "посмотрите calib.png; чаще всего мешает свет: края "
                      "слишком тёмные или центр пересвечен"))
    finally:
        cam.close()

    _save_camera_config({"gain": gain, "black_level": black,
                         "dark_subtract": True, "flat_field": True,
                         "flat_target": int(target) if target == int(target) else target})
    print("график до/после - calib.png рядом со скриптом. Сервер можно "
          "запускать (лента пустая: фон пересоберётся сам)")


def main():
    if "--calibrate" in sys.argv:
        calibrate()
        return
    if "--gain-sweep" in sys.argv:
        gain_sweep()
        return
    if "--black-sweep" in sys.argv:
        black_sweep()
        return
    if "--sensitivity-sweep" in sys.argv:
        sensitivity_sweep()
        return
    if "--dark" in sys.argv:
        calibrate_dark()
        return
    if "--color-info" in sys.argv:
        color_info()
        return
    if "--snap" in sys.argv:
        i = sys.argv.index("--snap")
        sec = 3.0
        if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("-"):
            sec = float(sys.argv[i + 1])
        snapshot(sec)
        return
    if "--exposure-sweep" in sys.argv:
        exposure_sweep()
        return
    if "--flat" in sys.argv:
        target = None
        if "--target" in sys.argv:
            target = float(sys.argv[sys.argv.index("--target") + 1])
        calibrate_flat(target)
        return
    if "--selftest" in sys.argv:
        raise SystemExit(0 if selftest() else 1)
    if "--graph" in sys.argv:
        show_graph()
        return

    begin()
    while not ready() and not failed():
        time.sleep(0.1)
    if failed():
        raise SystemExit(1)
    print("жду предметы, Ctrl+C для выхода")
    arm()
    saved = 0
    try:
        while True:
            frame = take()
            if frame is not None:
                name = "frame_%03d.png" % saved
                cv2.imwrite(name, frame)
                saved += 1
                print(f"  сохранено {name}  [{status()}]")
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n" + status())
    finally:
        stop()


if __name__ == "__main__":
    main()