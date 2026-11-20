"""Run beside line_camera.py, with the production server stopped."""
import argparse
import csv
import platform
import queue
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

clock = time.perf_counter_ns


def report(title, data):
    print(f"\n{title} (мс)", flush=True)
    print(f"{'операция':26} {'N':>7} {'сред':>9} {'P50':>9} {'P95':>9} {'P99':>9} {'макс':>9}")
    for key, values in data.items():
        if len(values):
            v = np.asarray(values)
            p = np.percentile(v, [50, 95, 99])
            print(f"{key:26} {len(v):7d} {v.mean():9.3f} {p[0]:9.3f} {p[1]:9.3f} {p[2]:9.3f} {v.max():9.3f}")


def report_slow_inferences(records, threshold, path):
    slow = [row for row in records if row['total_ms'] > threshold]
    print(f"\nВыбросы NPU из {len(records)} вызовов: "
          + ", ".join(f">{limit:g} мс={sum(row['total_ms'] > limit for row in records)}"
                      for limit in (3, 10, 20)))
    print(f"Первый вызов с реальной вырезкой: {records[0]['total_ms']:.3f} мс")
    print(f"Медленные вызовы >{threshold:g} мс: {len(slow)}")
    for row in slow:
        print(f"  #{row['iteration']:04d} crop={row['crop_index']} "
              f"после_предыдущего={row['idle_ms']:.3f} мс "
              f"всего={row['total_ms']:.3f} Подг={row['prep_ms']:.3f} "
              f"RKNN={row['rknn_ms']:.3f} RKNN_CPU={row['rknn_cpu_ms']:.3f} "
              f"Рез={row['result_ms']:.3f} CPU_всего={row['thread_cpu_ms']:.3f}")
    print("Самые долгие вызовы: "
          + ", ".join(f"#{row['iteration']}={row['total_ms']:.3f} мс"
                      for row in sorted(records, key=lambda row: row['total_ms'], reverse=True)[:10]))
    if path:
        with path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(slow)
        print(f"Медленные вызовы записаны в {path}")


def capture(seconds):
    import line_camera as lc
    cam = lc.LineCamera()
    stats = defaultdict(list)
    failure = []
    worker = None
    try:
        cam.open()
        cam._load_flat()
        sdk = cam.ksj

        class MeasuredSDK:
            def __getattr__(self, name):
                return getattr(sdk, name)

            def KSJ_CaptureRgbData(self, index, buf):
                start = clock()
                rc = sdk.KSJ_CaptureRgbData(index, buf)
                stats['SDK успешный вызов' if rc >= 0 else 'SDK ошибка'].append((clock()-start)/1e6)
                return rc

        cam.ksj = MeasuredSDK()
        def grab():
            try:
                cam._grabber()
            except Exception as exc:
                failure.append(exc)

        cam._running = True
        worker = threading.Thread(target=grab)
        start = clock()
        worker.start()
        previous = None
        while clock()-start < seconds*1e9 and worker.is_alive():
            try:
                seq, piece, stamp, lag, published = cam.frames.get(timeout=.1)
            except queue.Empty:
                continue
            if previous is not None:
                stats['между готовыми блоками'].append((published-previous)/1e6)
            previous = published
            stats['возраст при захвате'].append(lag)
    finally:
        cam._running = False
        if worker:
            worker.join()
        cam.close()
    report('ЗАХВАТ', stats)
    print(f"Строк/блок={cam.chunk_rows}, период блока по настройке={cam.group*cam.piece_sec*1000:.3f} мс, "
          f"потери очереди={cam.dropped}, ошибки SDK={cam.errors}")
    if failure:
        raise failure[0]
    if previous is None:
        raise RuntimeError("Камера не выдала ни одного блока")


def replay(args):
    import line_camera as lc
    from stream_detector import make_stream_detector
    image = cv2.imread(str(args.image)) if args.image else None
    if args.image and image is None:
        raise ValueError(f"Не удалось прочитать {args.image}")
    if image is None:
        image = np.zeros((512, lc.SENSOR_WIDTH, 3), np.uint8)
        image[:] = (180, 95, 30)
        for y in range(24, 480, 34):
            x = lc.X0 + (y*7) % max(1, lc.X1-lc.X0-80) + 30
            cv2.ellipse(image, (x,y), (22,7), 0, 0, 360, (55,150,210), -1)
    if args.background:
        bg = np.load(args.background, allow_pickle=False)
    else:
        bg = np.median(image, axis=0).astype(np.uint8)
    if image.shape[1] != lc.SENSOR_WIDTH or bg.shape != image.shape[1:]:
        raise ValueError("Нужны полноширинный BGR-снимок и фон формы (ширина, 3)")
    print(f"Вход={args.image or 'синтетика'}, фон={args.background or 'медиана кадра'}, "
          f"TH={lc.TH}, ROI={lc.X0}:{lc.X1}, блок={lc.PROC_ROWS} строк")
    h = lc.PROC_ROWS
    padding = (-len(image)) % h + ((lc.GAP_ROWS+h)//h+1)*h
    rows = np.concatenate((image, np.repeat(bg[None], padding, axis=0)))
    cls = make_stream_detector(lc.ObjectDetector, vars(lc))
    detect, cut, npu = defaultdict(list), defaultdict(list), defaultdict(list)
    crops = []
    snaps = []
    rounds = args.repeat if args.module in ('all','detect') else 1
    for epoch in range(rounds):
        det = cls(bg,h)
        det.frames_enabled = False
        callback_ns = 0
        def closed(lo,hi):
            nonlocal callback_ns
            start = clock()
            snap = det.snapshot_track(lo,hi)
            elapsed = clock()-start
            callback_ns += elapsed
            detect['снимок / трек'].append(elapsed/1e6)
            if epoch == 0 and snap is not None:
                snaps.append(snap)
        det.on_run = closed
        for i in range(0,len(rows),h):
            block = rows[i:i+h]
            callback_ns = 0
            start = clock()
            det.feed(block, t_end=(i+h)/lc.ROWS_PER_SEC)
            elapsed = clock()-start
            detect['feed / блок'].append(elapsed/1e6)
            detect['feed без снимка / блок'].append((elapsed-callback_ns)/1e6)
    if args.module in ('all','detect'):
        report('ДЕТЕКЦИЯ STREAM', detect)
    print(f"Снимков треков за один проход: {len(snaps)}")
    if args.module == 'detect':
        return
    for epoch in range(args.repeat if args.module in ('all','cut') else 1):
        for snap in snaps:
            start = clock()
            objects, edge = det.extract_snapshot(snap)
            cut['всё вырезание / трек'].append((clock()-start)/1e6)
            for key,value in snap['cut_profile'].items():
                cut[key].append(value)
            if epoch == 0:
                crops.extend(o['crop'] for o in objects)
    if args.module in ('all','cut'):
        report('ВЫРЕЗАНИЕ', cut)
    print(f"Вырезок за один проход: {len(crops)}")
    if not crops:
        raise ValueError("Нет вырезок для измерения. Проверьте кадр, фон и ROI.")
    if args.module == 'cut':
        return
    import sorter
    sorter.CFG['npu_core'] = args.core
    model = sorter.Classifier()
    records = []
    previous_end = None
    try:
        if args.prime_inferences:
            prime_slow = []
            for i in range(args.prime_inferences):
                start = clock()
                _, _, stages = model.classify_timed(crops[0])
                elapsed = (clock()-start)/1e6
                if elapsed > args.slow_ms:
                    prime_slow.append((i+1, elapsed, stages['RKNN'], stages['RKNN_CPU']))
            print(f"Дополнительных вызовов до замера: {args.prime_inferences}; "
                  f"медленных >{args.slow_ms:g} мс: {len(prime_slow)}")
            for number, elapsed, rknn_ms, cpu_ms in prime_slow:
                print(f"  до замера #{number:04d}: всего={elapsed:.3f} "
                      f"RKNN={rknn_ms:.3f} RKNN_CPU={cpu_ms:.3f} мс")
        for i in range(args.inferences):
            crop_index = 0 if args.fixed_crop else i % len(crops)
            crop = crops[crop_index]
            start = clock()
            cpu_start = time.thread_time_ns()
            _, _, stages = model.classify_timed(crop)
            cpu_end = time.thread_time_ns()
            end = clock()
            total_ms = (end-start)/1e6
            npu['полная классификация'].append(total_ms)
            for key,value in stages.items():
                npu[key].append(value)
            records.append(dict(iteration=i+1, crop_index=crop_index,
                                idle_ms=0 if previous_end is None else (start-previous_end)/1e6,
                                total_ms=total_ms, prep_ms=stages['Подг'],
                                rknn_ms=stages['RKNN'], rknn_cpu_ms=stages['RKNN_CPU'],
                                result_ms=stages['Рез'],
                                thread_cpu_ms=(cpu_end-cpu_start)/1e6))
            previous_end = end
            if args.pause_ms and i+1 < args.inferences:
                time.sleep(args.pause_ms/1000)
    finally:
        model.rknn.release()
    report(f'NPU ядро {args.core}, пауза {args.pause_ms} мс', npu)
    report_slow_inferences(records, args.slow_ms, args.slow_log)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('module', choices=('all','capture','detect','cut','npu'))
    p.add_argument('--image', type=Path)
    p.add_argument('--background', type=Path)
    p.add_argument('--seconds', type=float, default=5)
    p.add_argument('--repeat', type=int, default=10)
    p.add_argument('--inferences', type=int, default=500)
    p.add_argument('--prime-inferences', type=int, default=0,
                   help='Дополнительные классификации той же вырезки перед измеряемым циклом')
    p.add_argument('--core', type=int, choices=(0,1,2), default=0)
    p.add_argument('--fixed-crop', action='store_true')
    p.add_argument('--pause-ms', type=float, default=0)
    p.add_argument('--slow-ms', type=float, default=3,
                   help='Печатать вызовы классификации дольше этого порога, мс')
    p.add_argument('--slow-log', type=Path,
                   help='Дополнительно записать медленные вызовы в CSV')
    a = p.parse_args()
    if min(a.seconds,a.repeat,a.inferences)<=0 or min(a.pause_ms,a.slow_ms,a.prime_inferences)<0:
        p.error("Нужны положительные длительность/число повторов и неотрицательная пауза")
    print(f"Python {platform.python_version()}, OpenCV {cv2.__version__}, OpenCV threads={cv2.getNumThreads()}")
    if a.module in ('all','capture'):
        capture(a.seconds)
    if a.module != 'capture':
        replay(a)


if __name__ == '__main__':
    main()
