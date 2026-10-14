#!/bin/bash
#
# Запуск сервера. Вместо
#
#     LD_LIBRARY_PATH=/opt/ksjapi/arm64 python3 dsp_protocol.py
#
# достаточно:
#
#     ./run.sh                 сервер
#     ./run.sh graph           живой график с камеры (matplotlib)
#     ./run.sh camera          проверка камеры без сервера
#     ./run.sh selftest        детекция без железа
#
# Первая ошибка останавливает скрипт, необъявленная переменная - тоже.
set -euo pipefail

# Переходим в каталог самого скрипта: иначе config.json, valves.json и
# capture.png будут искаться там, откуда скрипт позвали.
cd "$(dirname "$(readlink -f "$0")")"

# Чтобы libksjapi.so нашла свои плагины.
export LD_LIBRARY_PATH="/opt/ksjapi/arm64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

case "${1:-server}" in
    server)   exec python3 dsp_protocol.py "${@:2}" ;;
    graph)    exec python3 line_camera.py --graph ;;
    camera)   exec python3 line_camera.py ;;
    selftest) exec python3 line_camera.py --selftest ;;
    *)
        echo "не знаю команды '$1'" >&2
        echo "можно: server (по умолчанию), graph, camera, selftest" >&2
        exit 1
        ;;
esac
