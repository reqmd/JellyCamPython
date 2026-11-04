"""
Простой логгер: всё, что печатается, дублируется в файл с временем.

В консоли вывод остаётся как был. В файл каждая строка пишется с меткой
времени до секунд:

    2026-10-20 14:25:03 чтение: регистр 99, количество 1

Перехватывается и обычный print, и поток ошибок - туда попадают
трассировки исключений, в том числе из потоков камеры, и предупреждения.
Именно они нужны, когда что-то тихо перестаёт работать.

Подключение - одна строка в самом начале программы:

    import logger; logger.start()

Файлы лежат в папке logs рядом с этим файлом, у каждого запуска свой:
logs/2026-11-04_09-14-55.log - дата и время старта. Перезапуск, в том
числе автоматический после падения, начинает новый файл, прежние не
трогаются. Если программа работает через полночь, в 00:00 тоже
начинается новый файл - чтобы файлы оставались по дням.

Файлы старше LOG_KEEP_DAYS дней удаляются при запуске, чтобы папка не
росла бесконечно. 0 - не удалять ничего.
"""

import atexit
import os
import sys
import threading
import time

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_KEEP_DAYS = 30

_lock = threading.Lock()
_file = None
_day = None


def _new_name():
    """Имя файла по текущему времени: 2026-11-04_09-14-55.log."""
    name = time.strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(LOG_DIR, name + ".log")
    n = 1
    while os.path.exists(path):         # два запуска в одну секунду
        n += 1
        path = os.path.join(LOG_DIR, f"{name}_{n}.log")
    return path


def _open_for_today():
    """
    Файл текущего запуска. Открывается при первой записи; в полночь
    вместо него начинается новый.
    """
    global _file, _day
    day = time.strftime("%Y-%m-%d")
    if day != _day:
        if _file is not None:
            _file.close()
        os.makedirs(LOG_DIR, exist_ok=True)
        _file = open(_new_name(), "w", encoding="utf-8")
        _day = day
    return _file


def _cleanup():
    """Удалить логи старше LOG_KEEP_DAYS дней."""
    if not LOG_KEEP_DAYS or not os.path.isdir(LOG_DIR):
        return
    limit = time.time() - LOG_KEEP_DAYS * 86400
    for name in os.listdir(LOG_DIR):
        path = os.path.join(LOG_DIR, name)
        if name.endswith(".log") and os.path.getmtime(path) < limit:
            try:
                os.remove(path)
            except OSError:
                pass


class _Tee:
    """
    Подменяет sys.stdout или sys.stderr.

    Всё сразу уходит в консоль как есть. В файл строка пишется целиком,
    когда она закончилась: print из разных потоков иначе перемешал бы
    куски строк. Поэтому незаконченная строка копится отдельно для
    каждого потока.
    """

    def __init__(self, stream):
        self._stream = stream
        self._pending = {}              # поток -> незаконченная строка

    def write(self, text):
        self._stream.write(text)
        if not text:
            return 0
        key = threading.get_ident()
        with _lock:
            buf = self._pending.get(key, "") + text
            lines = buf.split("\n")
            self._pending[key] = lines.pop()       # хвост без перевода строки
            if lines:
                f = _open_for_today()
                stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write("".join(f"{stamp} {ln}\n" for ln in lines))
                f.flush()
        return len(text)

    def flush(self):
        self._stream.flush()

    def drain(self):
        """Дописать незаконченные строки - при выходе из программы."""
        with _lock:
            rest = [v for v in self._pending.values() if v]
            self._pending.clear()
            if rest:
                f = _open_for_today()
                stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write("".join(f"{stamp} {ln}\n" for ln in rest))
                f.flush()

    def __getattr__(self, name):
        # encoding, isatty, fileno и прочее - как у настоящего потока
        return getattr(self._stream, name)


def start():
    """Включить запись. Повторный вызов ничего не делает."""
    if isinstance(sys.stdout, _Tee):
        return
    sys.stdout = _Tee(sys.stdout)
    sys.stderr = _Tee(sys.stderr)
    atexit.register(sys.stdout.drain)
    atexit.register(sys.stderr.drain)
    _cleanup()
    with _lock:
        path = _open_for_today().name
    print(f"--- запуск, лог пишется в {path}")