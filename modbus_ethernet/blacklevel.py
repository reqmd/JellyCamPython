"""
Чёрный уровень сегментов Awaiba: прочитать, выровнять, измерить решётку.

Запуск (как и сервер, с путём к плагинам SDK):

    LD_LIBRARY_PATH=/opt/ksjapi/arm64 python3 blacklevel.py            показать
    LD_LIBRARY_PATH=/opt/ksjapi/arm64 python3 blacklevel.py --auto     авто по всем
    LD_LIBRARY_PATH=/opt/ksjapi/arm64 python3 blacklevel.py --equalize вариант 2
    LD_LIBRARY_PATH=/opt/ksjapi/arm64 python3 blacklevel.py --set 120  всем одно
    LD_LIBRARY_PATH=/opt/ksjapi/arm64 python3 blacklevel.py --seg 1 --set 140
    LD_LIBRARY_PATH=/opt/ksjapi/arm64 python3 blacklevel.py --sweep    перебор
    LD_LIBRARY_PATH=/opt/ksjapi/arm64 python3 blacklevel.py --raw      сырой кадр

После любого изменения снимается кадр и печатается решётка: разница между
рядами сенсора и между столбцами в зелёном канале. Цель - привести обе
к нулю. Камеру надо навести на ровный освещённый участок ленты.

Внимание: регистры сенсора держат значение до снятия питания, а не до
перезапуска программы. Если убираете вызов авто-калибровки, камеру нужно
обесточить, иначе останутся старые значения.
"""

import argparse
import sys
from ctypes import (CDLL, byref, c_float, c_int, c_uint, c_ushort,
                    create_string_buffer)

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from line_camera import (EXPOSURE_MS, FIXED_RATE, GAIN, IDX, LIB, LINES,
                         SDK_DIR, SENSOR_ROWS, SENSOR_WIDTH, TRIGGER_MODE,
                         X0, X1)

MEASURE_FRAMES = 8          # кадров на одно измерение, для устойчивости


# --------------------------------------------------------------------------
# Камера: та же последовательность, что в line_camera.py, но без детекции
# --------------------------------------------------------------------------

def open_camera(auto_black: bool):
    ksj = CDLL(f"{SDK_DIR}/{LIB}")
    ksj.KSJ_ExposureTimeSet.argtypes = (c_int, c_float)
    ksj.KSJ_SetFixedFrameRateEx.argtypes = (c_int, c_float)
    ksj.KSJ_CaptureSetFieldOfViewEx.argtypes = (c_int,) * 7 + (c_ushort,)
    ksj.KSJ_CaptureSetTimeOut.argtypes = (c_int, c_uint)

    ksj.KSJ_Init()
    if ksj.KSJ_DeviceGetCount() < 1:
        ksj.KSJ_UnInit()
        sys.exit("камера не найдена")
    if hasattr(ksj, "KSJ_Open"):
        ksj.KSJ_Open(IDX)

    ksj.KSJ_TriggerModeSet(IDX, TRIGGER_MODE)
    ksj.KSJ_SetFixedFrameRateEx(IDX, FIXED_RATE)
    ksj.KSJ_ExposureTimeSet(IDX, EXPOSURE_MS)
    ksj.KSJ_CaptureSetFieldOfViewEx(IDX, 0, 0, SENSOR_WIDTH, SENSOR_ROWS,
                                    0, 0, LINES)
    ksj.KSJ_CaptureSetTimeOut(IDX, int(LINES / FIXED_RATE * 2000 + 5000))

    segs = c_int()
    if ksj.KSJ_AWAIBA_GetSegmentNum(IDX, byref(segs)) < 0:
        segs.value = 0
    for sgm in range(segs.value):
        ksj.KSJ_AWAIBA_SetGain(IDX, sgm, GAIN)
        if auto_black:
            ksj.KSJ_AWAIBA_AutoBlackLevel(IDX, sgm)

    w, h, bits = c_int(), c_int(), c_int()
    ksj.KSJ_CaptureGetSizeEx(IDX, byref(w), byref(h), byref(bits))
    return ksj, segs.value, w.value, h.value, bits.value // 8


# --------------------------------------------------------------------------
# Чтение и запись чёрного уровня
# --------------------------------------------------------------------------

def read_levels(ksj, segs):
    """Текущее значение и допустимый диапазон по каждому сегменту."""
    out = []
    for sgm in range(segs):
        val, lo, hi = c_int(-1), c_int(-1), c_int(-1)
        ok_v = ksj.KSJ_AWAIBA_GetBlackLevel(IDX, sgm, byref(val)) >= 0
        ok_r = ksj.KSJ_AWAIBA_GetBlackLevelRange(
            IDX, sgm, byref(lo), byref(hi)) >= 0
        out.append((sgm,
                    val.value if ok_v else None,
                    (lo.value, hi.value) if ok_r else None))
    return out


def show_levels(ksj, segs):
    rows = read_levels(ksj, segs)
    print(f"\nсегментов: {segs}")
    print(f"{'сегмент':>8s}{'чёрный':>10s}{'диапазон':>16s}")
    for sgm, val, rng in rows:
        v = "нет" if val is None else str(val)
        r = "нет" if rng is None else f"{rng[0]}..{rng[1]}"
        print(f"{sgm:8d}{v:>10s}{r:>16s}")
    vals = [v for _, v, _ in rows if v is not None]
    if len(vals) > 1:
        print(f"разброс между сегментами: {max(vals) - min(vals)}")
    return rows


def set_all(ksj, segs, value, only=None):
    for sgm in range(segs):
        if only is not None and sgm != only:
            continue
        rc = ksj.KSJ_AWAIBA_SetBlackLevel(IDX, sgm, int(value))
        if rc < 0:
            print(f"  сегмент {sgm}: запись {value} не удалась, код {rc}")


# --------------------------------------------------------------------------
# Измерение решётки
# --------------------------------------------------------------------------

def grab(ksj, w, h, ch, n=MEASURE_FRAMES):
    """Среднее нескольких порций, чтобы шум не мешал мерить решётку."""
    buf = create_string_buffer(w * h * ch)
    acc, got = None, 0
    for _ in range(n * 3):
        if got >= n:
            break
        if ksj.KSJ_CaptureRgbData(IDX, buf) < 0:
            continue
        a = np.frombuffer(buf, np.uint8).reshape(h, w, ch).astype(np.float32)
        acc = a if acc is None else acc + a
        got += 1
    if not got:
        return None
    return acc / got


def band(img, x0, x1):
    """
    Решётка в зелёном по квадрату 2x2 на полосе столбцов x0..x1.

    Зелёный физически снимается только в двух позициях квадрата из четырёх,
    остальные библиотека достраивает интерполяцией. Какая это пара - зависит
    от фазы Байера и от того, с чётной или нечётной строки начат кадр,
    поэтому меряем все четыре позиции и смотрим на их размах. Размах не
    зависит от раскладки и равен нулю только когда решётки нет.
    """
    g = img[:, x0:x1, 1]
    q = np.array([[g[0::2, 0::2].mean(), g[0::2, 1::2].mean()],
                  [g[1::2, 0::2].mean(), g[1::2, 1::2].mean()]])
    return g.mean(), g.max(), q.max() - q.min(), q[0, 0] - q[1, 1], \
        q[0, 1] - q[1, 0]


def grab_raw(ksj, w, h, n=3):
    """
    Сырой кадр, без демозаика и без обработки.

    Берём последний из нескольких: первые кадры после открытия камеры ещё
    не устоялись - в сыром дампе видно, как на 23-й паре строк чётные
    столбцы разом проседают на 40 процентов.
    """
    buf = create_string_buffer(w * h * 4)
    out = None
    for _ in range(n):
        if ksj.KSJ_CaptureRawData(IDX, buf) < 0:
            continue
        out = np.frombuffer(buf, np.uint8)[:w * h].reshape(h, w).copy()
    return out


def raw_quads(frame):
    """Четыре позиции байеровского квадрата по сырому кадру, на ленте."""
    s = frame[frame.shape[0] // 2:].astype(np.float32)   # хвост, устоялся
    b = s[:, X0:X1]
    return {(r, c): float(b[r::2, c::2].mean())
            for r in (0, 1) for c in (0, 1)}


def greens(q):
    """
    Две зелёные позиции квадрата.

    Фон синий: красный самый тёмный, синий самый яркий, зелёные - две
    средние. Раскладку Байера знать не нужно, порядок яркости устойчив.
    """
    order = sorted(q.items(), key=lambda kv: kv[1])
    return order[1], order[2]


def show_raw(frame, label=""):
    q = raw_quads(frame)
    lo, hi = greens(q)
    print(f"  {label}сырой квадрат: " +
          "  ".join(f"{k}={v:7.2f}" for k, v in sorted(q.items())))
    ratio = hi[1] / max(lo[1], 0.01)
    print(f"  зелёные {lo[0]}={lo[1]:.2f} и {hi[0]}={hi[1]:.2f}   "
          f"отношение {ratio:.4f}")
    return lo, hi, ratio


def measure(img, quiet=False):
    """
    Печатает решётку по трём полосам и возвращает размах по ленте.

    Половины кадра меряются отдельно: сегментов у сенсора два, и если
    сегмент - это группа столбцов, правка сегмента 1 сдвинет только правую
    половину. Если сегмент - ряд сенсора, сдвинутся обе одинаково.
    """
    w = img.shape[1]
    rows = (("левая половина", 100, w // 2 - 100),
            ("правая половина", w // 2 + 100, w - 100),
            ("лента", X0, X1))
    out = None
    if not quiet:
        print(f"  {'полоса':18s}{'уровень':>9s}{'размах':>9s}"
              f"{'диаг':>8s}{'анти':>8s}")
    for name, a, b in rows:
        lvl, mx, spread, diag, anti = band(img, a, b)
        if not quiet:
            print(f"  {name:18s}{lvl:9.1f}{spread:9.2f}{diag:+8.2f}{anti:+8.2f}")
        if name == "лента":
            out = spread
            if not quiet:
                if lvl < 20:
                    print("  темно - наведите камеру на освещённую ленту")
                elif mx > 250:
                    print("  зелёный упёрся в потолок - убавьте свет или GAIN")
    return out


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true",
                    help="авто-калибровка по всем сегментам")
    ap.add_argument("--equalize", action="store_true",
                    help="значение сегмента 0 проставить всем (вариант 2)")
    ap.add_argument("--set", type=int, metavar="N", help="задать значение")
    ap.add_argument("--seg", type=int, metavar="N",
                    help="только этот сегмент, вместе с --set")
    ap.add_argument("--sweep", action="store_true",
                    help="перебрать значения сегмента 1 и найти лучшее")
    ap.add_argument("--gain", action="store_true",
                    help="выровнять ряды усилением сегментов по сырому кадру")
    ap.add_argument("--repeat", type=int, metavar="N",
                    help="N измерений подряд без изменений: проверка, "
                         "насколько вообще повторяется результат")
    ap.add_argument("--raw", metavar="ФАЙЛ", nargs="?", const="raw.npy",
                    help="сохранить сырой кадр для разбора")
    args = ap.parse_args()

    # Авто-калибровку при открытии гасим, иначе она затрёт наши значения.
    ksj, segs, w, h, ch = open_camera(auto_black=args.auto)
    print(f"порция {w}x{h}, каналов {ch}")

    if segs == 0:
        print("KSJ_AWAIBA_GetSegmentNum не ответил - сегментов нет")

    mode = c_int(-1)
    if hasattr(ksj, "KSJ_BayerGetMode") and \
            ksj.KSJ_BayerGetMode(IDX, byref(mode)) >= 0:
        print(f"режим Байера: {mode.value}")

    try:
        if args.raw:
            raw = create_string_buffer(w * h * 4)
            if ksj.KSJ_CaptureRawData(IDX, raw) < 0:
                print("KSJ_CaptureRawData вернул ошибку")
            else:
                np.save(args.raw, np.frombuffer(raw, np.uint8).copy())
                print(f"сырой кадр сохранён: {args.raw} "
                      f"({w * h * 4} байт, порция {w}x{h})")
            return

        show_levels(ksj, segs)
        print("\nдо изменений:")
        img = grab(ksj, w, h, ch)
        if img is None:
            print("  кадр не получен")
            return
        measure(img)

        if args.gain:
            if segs < 2:
                print("\nсегментов меньше двух, выравнивать нечем")
                return
            print("\n=== опорное состояние, сырой кадр ===")
            frame = grab_raw(ksj, w, h)
            if frame is None:
                print("KSJ_CaptureRawData не отвечает")
                return
            lo0, hi0, ratio0 = show_raw(frame)

            # Какой сегмент какой зелёный двигает - выясняем опытом,
            # а не догадкой: меняем усиление сегмента 0 и смотрим, что поехало.
            print(f"\n=== проба: сегменту 0 усиление {GAIN - 40} ===")
            ksj.KSJ_AWAIBA_SetGain(IDX, 0, GAIN - 40)
            probe = grab_raw(ksj, w, h)
            ksj.KSJ_AWAIBA_SetGain(IDX, 0, GAIN)
            if probe is None:
                print("кадр не получен")
                return
            q0, q1 = raw_quads(frame), raw_quads(probe)
            moved = {k: q1[k] - q0[k] for k in q0}
            for k in sorted(moved):
                print(f"  {k}: {q0[k]:7.2f} -> {q1[k]:7.2f}  "
                      f"({moved[k]:+6.2f})")
            seg0_moves_lo = abs(moved[lo0[0]]) > abs(moved[hi0[0]])
            print(f"  сегмент 0 двигает зелёный {'тусклый' if seg0_moves_lo else 'яркий'}")

            # Яркий зелёный надо опустить до тусклого.
            seg = 0 if not seg0_moves_lo else 1
            print(f"\n=== перебор усиления сегмента {seg} "
                  f"(в нём яркий зелёный) ===")
            best, best_r = GAIN, abs(ratio0 - 1.0)
            for g in range(max(0, GAIN - 80), GAIN + 1, 10):
                ksj.KSJ_AWAIBA_SetGain(IDX, seg, g)
                f = grab_raw(ksj, w, h, n=2)
                if f is None:
                    continue
                print(f"  усиление {g}:")
                _, _, rr = show_raw(f, "  ")
                if abs(rr - 1.0) < best_r:
                    best, best_r = g, abs(rr - 1.0)
            ksj.KSJ_AWAIBA_SetGain(IDX, seg, best)
            print(f"\nлучшее усиление сегмента {seg}: {best} "
                  f"(отношение отличается от 1 на {best_r:.4f})")
            print(f"исходно было {abs(ratio0 - 1.0):.4f}")
            if best_r > abs(ratio0 - 1.0) * 0.5:
                print("выигрыш невелик - усиление тоже не рычаг, "
                      "остаётся двухточечная калибровка")
            else:
                print(f"пропишите GAIN отдельно: сегмент {seg} = {best}, "
                      f"второй = {GAIN}")
            print("\nпроверка по картинке:")
            img = grab(ksj, w, h, ch)
            if img is not None:
                measure(img)
            return

        if args.repeat:
            print(f"\nповторяемость: {args.repeat} измерений подряд, "
                  f"ничего не меняя")
            vals = []
            for i in range(args.repeat):
                img = grab(ksj, w, h, ch, n=4)
                if img is None:
                    continue
                lvl, _, spread, _, anti = band(img, X0, X1)
                vals.append((lvl, spread))
                print(f"  {i + 1:2d}: уровень {lvl:6.1f}   размах {spread:6.2f}"
                      f"   анти {anti:+6.2f}")
            if len(vals) > 1:
                sp = [s for _, s in vals]
                lv = [l for l, _ in vals]
                print(f"\n  размах гуляет на {max(sp) - min(sp):.2f}, "
                      f"уровень на {max(lv) - min(lv):.1f}")
                print("  любая правка меньше этого - шум, а не результат")
            return

        if args.equalize:
            rows = read_levels(ksj, segs)
            base = rows[0][1]
            if base is None:
                print("\nсегмент 0 не читается, выравнивать нечем")
            else:
                print(f"\nставлю всем сегментам значение сегмента 0: {base}")
                set_all(ksj, segs, base)
        elif args.set is not None:
            where = "всем сегментам" if args.seg is None \
                else f"сегменту {args.seg}"
            print(f"\nставлю {where} значение {args.set}")
            set_all(ksj, segs, args.set, only=args.seg)

        if args.sweep:
            rows = read_levels(ksj, segs)
            cur = rows[1][1] if segs > 1 and rows[1][1] is not None else None
            rng = rows[1][2] if segs > 1 else None
            if cur is None or rng is None:
                print("\nперебор невозможен: сегмент 1 не читается")
            else:
                print(f"\nперебор сегмента 1 вокруг {cur} "
                      f"(диапазон {rng[0]}..{rng[1]}), цель - размах к нулю")
                best, best_d, seen = cur, None, []
                for v in range(max(rng[0], cur - 48),
                               min(rng[1], cur + 48) + 1, 8):
                    set_all(ksj, segs, v, only=1)
                    img = grab(ksj, w, h, ch, n=4)
                    if img is None:
                        continue
                    print(f"  значение {v}:")
                    d = measure(img)
                    seen.append((v, d))
                    if best_d is None or d < best_d:
                        best, best_d = v, d
                set_all(ksj, segs, best, only=1)
                print(f"\nлучшее значение сегмента 1: {best} "
                      f"(размах {best_d:.2f})")
                if len(seen) > 1:
                    lo = min(d for _, d in seen)
                    hi = max(d for _, d in seen)
                    if hi - lo < 1.0:
                        print("размах не отреагировал на перебор: чёрный "
                              "уровень на эту решётку не влияет, "
                              "лечить надо сырым кадром или программно")

        if args.equalize or args.set is not None or args.auto:
            show_levels(ksj, segs)
            print("\nпосле изменений:")
            img = grab(ksj, w, h, ch)
            if img is not None:
                measure(img)
    finally:
        if hasattr(ksj, "KSJ_Close"):
            ksj.KSJ_Close(IDX)
        ksj.KSJ_UnInit()


if __name__ == "__main__":
    main()