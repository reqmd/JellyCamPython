# -*- coding: utf-8 -*-
"""Анализ собранных объектов: какие они по размеру и какой вход нужен сети."""

import csv
import sys

import numpy as np

CSV_FILE = sys.argv[1] if len(sys.argv) > 1 else "objects.csv"

rows = list(csv.DictReader(open(CSV_FILE, encoding="utf-8")))
if not rows:
    sys.exit("файл пуст: сначала наберите объекты детектором")

f = lambda k: np.array([float(r[k]) for r in rows])
long_px, short_px = f("rect_w"), f("rect_h")
long_mm, short_mm = f("w_mm"), f("h_mm")
area_mm = f("area_mm2")
fill = f("fill")
aspect = long_px / np.maximum(short_px, 1)

print("объектов в выборке: %d\n" % len(rows))


def describe(name, a, unit):
    p = np.percentile(a, [5, 50, 95, 100])
    print("%-22s мин %7.1f | медиана %7.1f | 95%% %7.1f | макс %7.1f  %s"
          % (name, a.min(), p[1], p[2], p[3], unit))


describe("длинная сторона", long_mm, "мм")
describe("короткая сторона", short_mm, "мм")
describe("площадь", area_mm, "мм2")
describe("вытянутость", aspect, "раз")
describe("заполненность", fill, "доля прямоугольника")

# ------------------------------------------------------ рекомендация по входу
need = np.percentile(long_px, 95)
for candidate in (64, 96, 128, 160, 192, 224, 256, 320, 384):
    if candidate >= need:
        break

print("\nмаксимальная сторона объекта (95-й процентиль): %.0f px" % need)
print("рекомендуемый вход сети: %dx%d" % (candidate, candidate))
if candidate < need:
    print("  внимание: объекты крупнее любого стандартного входа,")
    print("  при уменьшении потеряется часть деталей")
if need / max(np.percentile(long_px, 5), 1) > 4:
    print("  объекты сильно разного размера - подумайте, важен ли")
    print("  абсолютный масштаб: если да, не нормализуйте размер,")
    print("  а вписывайте в общий холст фиксированного масштаба")

# ---------------------------------------------------------- грубая гистограмма
print("\nраспределение длинной стороны, мм:")
hist, edges = np.histogram(long_mm, bins=12)
for cnt, lo, hi in zip(hist, edges[:-1], edges[1:]):
    print("  %6.1f - %6.1f  %4d  %s" % (lo, hi, cnt, "#" * int(40 * cnt / max(hist.max(), 1))))