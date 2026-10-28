from valves import ValveBoard
import time

mode = 1
print(f"mode {mode}")
with ValveBoard(speed=20_000_000, mode=mode) as b:
    for duration in range(1, 5):
        for _ in range(3000):
            b.open_valves(0xFFFFFFFFFFFFFFFF, duration=duration)
            time.sleep(0.01)
        time.sleep(1)