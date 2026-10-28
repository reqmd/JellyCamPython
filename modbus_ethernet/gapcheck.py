"""
Теряются ли строки между порциями.

Лента едет непрерывно, а камера отдаёт строки порциями по LINES считываний.
Если между вызовами KSJ_CaptureRgbData есть время, когда камера не снимает,
строки за это время пропадают, и предмет на стыке порций режется.

Скрипт засекает время внутри каждого вызова и между вызовами и сравнивает
с длительностью порции по расчёту (LINES / FIXED_RATE).

Запуск, лента может быть пустой:

    LD_LIBRARY_PATH=/opt/ksjapi/arm64 python3 gapcheck.py

Режимы перебираются сами:
    как в сервере  - после вызова копия порции и очередь, как в _grabber
    без работы     - между вызовами ничего, два буфера по очереди
"""

import sys
import time
from ctypes import create_string_buffer

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from blacklevel import open_camera                      # noqa: E402
from line_camera import FIXED_RATE, IDX, LINES, SENSOR_ROWS  # noqa: E402

N = 300            # порций на режим
WARMUP = 10        # первые порции не считаем


def measure(ksj, w, h, ch, copy, pause):
    """
    copy  - после вызова делать то же, что _grabber (копия порции);
    pause - сколько секунд нарочно ждать перед следующим вызовом.
    """
    bufs = [create_string_buffer(w * h * ch) for _ in range(2)]
    calls, betweens, errors = [], [], 0
    prev_end = None
    keep = []
    for i in range(N + WARMUP):
        buf = bufs[i % 2]
        t0 = time.perf_counter()
        rc = ksj.KSJ_CaptureRgbData(IDX, buf)
        t1 = time.perf_counter()
        if rc < 0:
            errors += 1
        if i >= WARMUP:
            calls.append(t1 - t0)
            if prev_end is not None:
                betweens.append(t0 - prev_end)
        prev_end = t1                    # "между" включает всю нашу работу
        if copy:
            piece = np.frombuffer(buf, np.uint8).reshape(h, w, ch).copy()
            keep.append(piece)
            if len(keep) > 8:
                keep.pop(0)
        if pause:
            time.sleep(pause)
    return np.array(calls), np.array(betweens), errors


def report(name, calls, betweens, errors, piece_ms):
    c, b = calls * 1000, betweens * 1000
    print(f"\n{name}:")
    print(f"  внутри вызова     среднее {c.mean():6.2f} мс "
          f"(мин {c.min():.2f}, макс {c.max():.2f})")
    print(f"  между вызовами    среднее {b.mean():6.3f} мс "
          f"(макс {b.max():.3f})")
    print(f"  полный цикл       {c.mean() + b.mean():6.2f} мс, "
          f"порция по расчёту {piece_ms:.2f} мс")
    if errors:
        print(f"  ошибок захвата: {errors}")
    return c.mean(), b.mean()


CONFIGS = ((64, 8000.0), (128, 8000.0), (256, 8000.0),
           (64, 6000.0), (64, 4000.0), (64, 3000.0))


def sweep():
    """
    Какой темп камера держит на самом деле.

    При LINES = 64 и FIXED_RATE = 8000 порция должна приходить раз в 8 мс,
    а приходит раз в 16. Прогоняем несколько сочетаний и смотрим, во
    сколько раз реальный цикл длиннее расчётного. Отношение 1 - камера
    держит заданную частоту и снимает без пропусков. Отношение 2 и больше -
    не держит, и строки между кадрами пропадают.
    """
    import blacklevel
    global N
    keep_n, N = N, 120
    rows = []
    for lines, rate in CONFIGS:
        blacklevel.LINES, blacklevel.FIXED_RATE = lines, rate
        ksj, segs, w, h, ch = blacklevel.open_camera(auto_black=True)
        try:
            calls, betweens, err = measure(ksj, w, h, ch, True, 0)
        finally:
            if hasattr(ksj, "KSJ_Close"):
                ksj.KSJ_Close(IDX)
            ksj.KSJ_UnInit()
        cycle = (calls.mean() + betweens.mean()) * 1000
        calc = lines / rate * 1000
        rows.append((lines, rate, calc, cycle, lines / cycle * 1000, err))
    N = keep_n

    print(f"\n{'LINES':>6s}{'FIXED_RATE':>12s}{'расчёт':>9s}{'реально':>9s}"
          f"{'во сколько':>12s}{'считываний/с':>15s}")
    for lines, rate, calc, cycle, eff, err in rows:
        print(f"{lines:6d}{rate:12.0f}{calc:8.2f}м{cycle:8.2f}м"
              f"{cycle / calc:11.2f}x{eff:15.0f}"
              f"{'  ошибки ' + str(err) if err else ''}")

    good = [r for r in rows if r[3] / r[2] < 1.1]
    print("\nвывод:")
    if good:
        best = max(good, key=lambda r: r[1])
        print(f"  без отставания камера держит до {best[1]:.0f} считываний "
              f"в секунду (LINES = {best[0]}).")
        print(f"  поставьте FIXED_RATE = {best[1]:.0f}: тогда каждая порция "
              f"приходит вовремя и строки между порциями не пропадают.")
    else:
        print("  ни в одном сочетании камера не держит заданный темп - "
              "пришлите таблицу, разберём.")
    eff = sorted(r[4] for r in rows)
    print(f"  реальный темп во всех прогонах: {eff[0]:.0f}..{eff[-1]:.0f} "
          f"считываний в секунду")


def main():
    if "--sweep" in sys.argv:
        sweep()
        return
    ksj, segs, w, h, ch = open_camera(auto_black=True)
    piece_ms = LINES / FIXED_RATE * 1000
    rows_per_ms = FIXED_RATE * SENSOR_ROWS / 1000
    pause_ms = piece_ms / 2
    print(f"порция {w}x{h}, по расчёту {piece_ms:.2f} мс "
          f"({LINES} считываний при {FIXED_RATE:.0f} в секунду)")
    try:
        call_a, gap_a = report("как в сервере",
                               *measure(ksj, w, h, ch, True, 0)[:2],
                               0, piece_ms)
        call_b, gap_b = report(f"с нарочной паузой {pause_ms:.1f} мс",
                               *measure(ksj, w, h, ch, False,
                                        pause_ms / 1000)[:2],
                               0, piece_ms)

        print("\nвывод:")
        drop = call_a - call_b
        if drop > pause_ms * 0.6:
            # Пока мы спали, камера продолжала снимать, и следующий вызов
            # вернулся быстрее: строки за время паузы не потеряны.
            print(f"  с паузой вызов вернулся на {drop:.2f} мс быстрее - "
                  f"камера снимает непрерывно, пока мы заняты.")
            print("  потерь от нашей работы между вызовами нет. Если "
                  "предметы режутся, строки теряет сама камера между "
                  "кадрами - нужен её потоковый режим.")
        else:
            lost = gap_a * rows_per_ms
            print(f"  пауза не ускорила вызов (разница {drop:.2f} мс) - "
                  f"камера снимает только пока мы ждём в вызове.")
            print(f"  всё время между вызовами пропадает: в сервере это "
                  f"{gap_a:.3f} мс, ~{lost:.0f} строк на каждый стык порций.")
            if lost < 2:
                print("  по времени потеря меньше двух строк - мало, чтобы "
                      "объяснить срезы; теряется внутри вызова.")
    finally:
        if hasattr(ksj, "KSJ_Close"):
            ksj.KSJ_Close(IDX)
        ksj.KSJ_UnInit()


if __name__ == "__main__":
    main()