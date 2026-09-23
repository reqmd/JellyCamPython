# -*- coding: utf-8 -*-
"""Опрос камеры: какие режимы съёмки и способы триггера она поддерживает."""

import os
from ctypes import WinDLL, c_int, byref

SDK_DIR = r"C:\путь\к\папке\с\dll"
IDX = 0

os.add_dll_directory(SDK_DIR)
ksj = WinDLL(os.path.join(SDK_DIR, "KSJApi64.dll"))
ksj.KSJ_Init()


def q(code, label):
    """Спросить у камеры флаг поддержки. code - позиция в enum KSJ_FUNCTION."""
    v = c_int()
    rc = ksj.KSJ_QueryFunction(IDX, code, byref(v))
    state = ("да" if v.value else "нет") if rc >= 0 else "не отвечает"
    print("  %-42s %s" % (label, state))


print("поддержка режимов:")
q(11, "внешний триггер")
q(12, "софтовый триггер")
q(13, "фиксированная частота")
q(14, "  из них программная")
q(15, "  из них аппаратная")
q(34, "задержка триггера")
q(35, "многокадровая сшивка")

print("\nспособы реакции на внешний сигнал:")
q(17, "по спаду")
q(18, "по фронту")
q(19, "по высокому уровню")
q(20, "по низкому уровню")

print("\nчто принимает KSJ_TriggerModeSet на самом деле:")
names = ["0 внутренний", "1 внешний", "2 софтовый", "3 фикс. частота",
         "4 фикс. частота по уровню", "5 энкодер"]
supported = []
for m, name in enumerate(names):
    rc = ksj.KSJ_TriggerModeSet(IDX, m)
    ok = rc >= 0
    if ok:
        supported.append(m)
    print("  %-28s %s" % (name, "принят" if ok else "отказ (%d)" % rc))

ksj.KSJ_TriggerModeSet(IDX, 0)   # вернуть внутренний режим
print("\nитого доступно:", supported)
ksj.KSJ_UnInit()