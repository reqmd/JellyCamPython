#!/bin/bash
#
# Установка сервера как службы systemd - чтобы не запускать его руками.
#
# Запускать от своего пользователя, НЕ через sudo (sudo скрипт попросит
# сам, когда понадобится): так он найдёт ваш Python из pyenv.
#
#     ./install_service.sh            сервер стартует при загрузке системы
#                                     и перезапускается, если упал
#     ./install_service.sh socket     порт слушает systemd, а сервер
#                                     стартует по первому пришедшему пакету
#     ./install_service.sh remove     убрать службу
#
# Управление потом:
#     systemctl status dsp            жив ли, последние строки вывода
#     sudo systemctl restart dsp      перезапуск (например, после правки кода)
#     sudo systemctl stop dsp         остановить
#     journalctl -u dsp -f            вывод сервера в реальном времени
#     (а свой лог по-прежнему пишется в logs/<дата>.log)

set -euo pipefail

NAME=dsp
DIR="$(dirname "$(readlink -f "$0")")"
UNIT_DIR=/etc/systemd/system
MODE="${1:-always}"

# Адрес и порт берём из самого сервера, чтобы не держать их в двух местах.
read -r ADDR PORT < <(cd "$DIR" && python3 -c \
    'import re,sys; s=open("dsp_protocol.py").read();
a=re.search(r"^IP_ADDRESS\s*=\s*[\x27\x22]([^\x27\x22]+)", s, re.M).group(1);
p=re.search(r"^PORT\s*=\s*(\d+)", s, re.M).group(1); print(a, p)')

if [ "$(id -u)" -eq 0 ]; then
    echo "запустите без sudo, от своего пользователя - иначе не найдётся" >&2
    echo "ваш Python из pyenv. sudo скрипт попросит сам." >&2
    exit 1
fi

if [ "$MODE" = remove ]; then
    sudo systemctl disable --now "$NAME.socket" 2>/dev/null || true
    sudo systemctl disable --now "$NAME.service" 2>/dev/null || true
    sudo rm -f "$UNIT_DIR/$NAME.service" "$UNIT_DIR/$NAME.socket"
    sudo systemctl daemon-reload
    echo "служба убрана"
    exit 0
fi

# Настоящий путь к интерпретатору: pyenv подставляет в PATH не сам Python,
# а "прокладку", которая без настроек pyenv под systemd не работает.
PYTHON="$(python3 -c 'import sys; print(sys.executable)')"
if ! "$PYTHON" -c 'import cv2, numpy' 2>/dev/null; then
    echo "в $PYTHON нет cv2 или numpy - это не тот Python, которым" >&2
    echo "запускается сервер. Активируйте нужный и повторите." >&2
    exit 1
fi
USER_NAME="$(id -un)"

echo "каталог:  $DIR"
echo "python:   $PYTHON"
echo "пользователь: $USER_NAME"
echo "адрес:    $ADDR:$PORT (UDP)"

SERVICE="[Unit]
Description=Эмулятор DSP: modbus по UDP и линейная камера
# Сеть должна быть поднята, иначе адреса $ADDR ещё нет.
Wants=network-online.target
After=network-online.target
# Не сдаваться после нескольких падений подряд, перезапускать всегда.
StartLimitIntervalSec=0

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$DIR
Environment=PYTHON=$PYTHON
# Без этого вывод в journalctl появлялся бы пачками с задержкой.
Environment=PYTHONUNBUFFERED=1
ExecStart=$DIR/run.sh server
# Упал - поднять снова через 3 секунды. Сколько угодно раз.
Restart=always
RestartSec=3
# Камере нужно время, чтобы отпустить USB при остановке.
TimeoutStopSec=10
"

if [ "$MODE" = socket ]; then
    SOCKET="[Unit]
Description=Порт эмулятора DSP: сервер стартует по первому пакету

[Socket]
ListenDatagram=$ADDR:$PORT
# Разрешить занять адрес, которого на карте ещё нет: при загрузке
# systemd открывает порты раньше, чем сеть получает адрес.
FreeBind=true

[Install]
WantedBy=sockets.target
"
    echo "$SERVICE" | sudo tee "$UNIT_DIR/$NAME.service" >/dev/null
    echo "$SOCKET"  | sudo tee "$UNIT_DIR/$NAME.socket"  >/dev/null
    sudo systemctl daemon-reload
    sudo systemctl disable --now "$NAME.service" 2>/dev/null || true
    sudo systemctl enable --now "$NAME.socket"
    echo
    echo "готово: порт $ADDR:$PORT слушает systemd, сервер стартует по"
    echo "первому пакету (например, запросу регистра 99) и дальше работает."
    echo "проверка: systemctl status $NAME.socket $NAME.service"
else
    SERVICE="$SERVICE
[Install]
WantedBy=multi-user.target
"
    echo "$SERVICE" | sudo tee "$UNIT_DIR/$NAME.service" >/dev/null
    sudo rm -f "$UNIT_DIR/$NAME.socket"
    sudo systemctl daemon-reload
    sudo systemctl enable --now "$NAME.service"
    echo
    echo "готово: сервер запущен и будет стартовать при каждой загрузке."
    echo "проверка: systemctl status $NAME"
fi