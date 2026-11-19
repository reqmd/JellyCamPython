"""Session-scoped cyclic GC profiling for the sorter (CPython on Linux)."""
import gc
import os
import sys
import time


def rss_mib():
    try:
        with open("/proc/self/statm", encoding="ascii") as stream:
            pages = int(stream.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1048576
    except (OSError, ValueError, IndexError):
        return float("nan")


class SessionGC:
    def __init__(self, settings):
        self.mode = os.environ.get("JELLY_GC_MODE", settings.get("mode", "normal"))
        if self.mode not in ("normal", "young_only"):
            raise ValueError("JELLY_GC_MODE must be normal or young_only")
        if self.mode == "young_only" and (
                sys.implementation.name != "cpython" or sys.version_info >= (3, 14)):
            raise ValueError("young_only requires CPython with the classic three-generation GC")
        self.running = False

    def observe(self, phase, info):
        gen = info["generation"]
        if phase == "start":
            self.begin[gen] = time.perf_counter_ns()
        elif self.begin[gen]:
            duration = time.perf_counter_ns() - self.begin[gen]
            self.begin[gen] = 0
            self.count[gen] += 1
            self.total[gen] += duration
            self.longest[gen] = max(self.longest[gen], duration)

    def start(self):
        if self.running:
            return
        self.previous = gc.get_threshold()
        self.begin, self.count, self.total, self.longest = ([0]*3 for _ in range(4))
        self.initial_rss = rss_mib()
        gc.callbacks.append(self.observe)
        if self.mode == "young_only":
            gc.set_threshold(700, 2147483647, 2147483647)
        self.running = True
        print(f"GC START mode={self.mode} python={sys.version.split()[0]} "
              f"enabled={gc.isenabled()} thresholds={gc.get_threshold()} "
              f"rss_MiB={self.initial_rss:.2f}")

    def stop(self):
        gc.callbacks.remove(self.observe)
        gc.set_threshold(*self.previous)
        self.running = False
        before = rss_mib()
        start = time.perf_counter_ns()
        freed = gc.collect(2)
        cleanup = (time.perf_counter_ns() - start) / 1e6
        for gen, count in enumerate(self.count):
            avg = self.total[gen] / max(1, count) / 1e6
            print(f"GC gen={gen} n={count} avg_ms={avg:.3f} "
                  f"max_ms={self.longest[gen]/1e6:.3f}")
        print(f"GC STOP mode={self.mode} rss_before_cleanup_MiB={before:.2f} "
              f"delta_MiB={before-self.initial_rss:+.2f} cleanup_ms={cleanup:.2f} "
              f"collected={freed} rss_after_cleanup_MiB={rss_mib():.2f}")
