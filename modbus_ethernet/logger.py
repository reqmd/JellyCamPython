"""Асинхронный лог сервера: HH:MM:SS.mmm в консоли и в файле.

print/stderr перехватываются. Метка ставится при завершении строки, а
запись и flush выполняются одним фоновым потоком. Очередь ограничена:
если диск/консоль не успевают, теряем диагностику, а не строки камеры.
Дата остаётся в имени файла; в полночь открывается новый файл.
"""
import atexit
import os
import queue
import sys
import threading
import time

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_KEEP_DAYS = 30
LOG_QUEUE_SIZE = 8192

_lock = threading.Lock()  # только короткая сборка строк/постановка в очередь
_queue = queue.Queue(maxsize=LOG_QUEUE_SIZE)
_file = None
_day = None
_thread = None
_accepting = False
_dropped = 0
_reported_dropped = 0
_out = _err = None


def _new_name(now):
    name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(now))
    path = os.path.join(LOG_DIR, name + ".log")
    n = 1
    while os.path.exists(path):
        n += 1
        path = os.path.join(LOG_DIR, f"{name}_{n}.log")
    return path


def _open_for_today(now):
    global _file, _day
    day = time.strftime("%Y-%m-%d", time.localtime(now))
    if _file is None or day != _day:
        if _file is not None:
            _file.close()
        os.makedirs(LOG_DIR, exist_ok=True)
        _file = open(_new_name(now), "w", encoding="utf-8")
        _day = day
    return _file


def _cleanup():
    if not LOG_KEEP_DAYS or not os.path.isdir(LOG_DIR):
        return
    limit = time.time() - LOG_KEEP_DAYS * 86400
    for name in os.listdir(LOG_DIR):
        path = os.path.join(LOG_DIR, name)
        if name.endswith(".log"):
            try:
                if os.path.getmtime(path) < limit:
                    os.remove(path)
            except OSError:
                pass


def _stamp(now=None):
    now = time.time() if now is None else now
    return (time.strftime("%H:%M:%S", time.localtime(now))
            + f".{int(now * 1000) % 1000:03d}")


def dropped_count():
    return _dropped


def _enqueue(stream, line):
    """Вызывается под _lock. Никогда не ждёт освобождения места."""
    global _dropped
    if not _accepting:
        return
    try:
        _queue.put_nowait((time.time(), stream, line))
    except queue.Full:
        _dropped += 1


def _write_record(now, stream, line):
    text = f"{_stamp(now)} {line}\n"
    # Не позволяем неисправной консоли или диску убить сортировку.
    try:
        stream.write(text)
        stream.flush()
    except Exception:
        pass
    try:
        f = _open_for_today(now)
        f.write(text)
        f.flush()
    except Exception as exc:
        # Не через print: это снова попало бы в очередь логов.
        try:
            _err.write(f"{_stamp()} ОШИБКА ЛОГА: {exc}\n")
            _err.flush()
        except Exception:
            pass


def _writer():
    global _reported_dropped
    while True:
        item = _queue.get()
        if item is None:
            break
        _write_record(*item)
        # Сообщаем об overflow после обработки очереди, а не из камеры.
        if _queue.empty() and _dropped != _reported_dropped:
            n = _dropped
            _write_record(time.time(), _err,
                          f"ЛОГ: пропущено строк {n - _reported_dropped} (всего {n})")
            _reported_dropped = n
    if _dropped != _reported_dropped:
        _write_record(time.time(), _err, f"ЛОГ: пропущено строк всего {_dropped}")
    if _file is not None:
        _file.close()


class _Tee:
    def __init__(self, stream):
        self._stream = stream
        self._pending = {}

    def write(self, text):
        if not text:
            return 0
        key = threading.get_ident()
        with _lock:
            buf = self._pending.get(key, "") + text
            lines = buf.split("\n")
            tail = lines.pop()
            if tail:
                self._pending[key] = tail
            else:
                self._pending.pop(key, None)
            for line in lines:
                _enqueue(self._stream, line)
        return len(text)

    def flush(self):
        # flush=True не должен синхронно писать на диск в потоке камеры.
        # Незавершённая строка остаётся до '\n' или нормального выхода.
        pass

    def drain(self):
        with _lock:
            for line in self._pending.values():
                _enqueue(self._stream, line)
            self._pending.clear()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _shutdown():
    global _accepting
    if not _accepting:
        return
    if isinstance(sys.stdout, _Tee):
        sys.stdout.drain()
    if isinstance(sys.stderr, _Tee):
        sys.stderr.drain()
    with _lock:
        _accepting = False
    # Ожидание только при выходе, не в рабочих потоках. Если вывод завис,
    # не держим завершение бесконечно.
    try:
        _queue.put(None, timeout=2)
    except queue.Full:
        return
    _thread.join(timeout=5)


def start():
    global _thread, _accepting, _out, _err
    if isinstance(sys.stdout, _Tee):
        return
    _out, _err = sys.stdout, sys.stderr
    _cleanup()
    path = _open_for_today(time.time()).name
    _accepting = True
    _thread = threading.Thread(target=_writer, daemon=True, name="логгер")
    _thread.start()
    sys.stdout = _Tee(_out)
    sys.stderr = _Tee(_err)
    atexit.register(_shutdown)
    print(f"--- запуск, лог пишется в {path}")
