#!/bin/sh
# Пересборка и установка оверлея низкой задержки CS для SPI0 (RK3588).
#
# Запускать после обновления ядра — иначе оверлей теряется и
# постамбула CS молча возвращается со ~2 мкс к ~120 мкс.
#
# Установка один раз:
#   sudo mkdir -p /usr/local/share/spi-lowlat
#   sudo cp spi0-lowlat.dts /usr/local/share/spi-lowlat/
#   sudo cp install-spi-lowlat.sh /usr/local/sbin/
#   sudo chmod +x /usr/local/sbin/install-spi-lowlat.sh
#
# Ручной запуск:        sudo install-spi-lowlat.sh
# Для другого ядра:     sudo install-spi-lowlat.sh 6.1.0-1030-rockchip

set -e

SRC=/usr/local/share/spi-lowlat/spi0-lowlat.dts
NAME=rk3588-spi0-lowlat.dtbo
KVER="${1:-$(uname -r)}"
DEST="/lib/firmware/${KVER}/device-tree/rockchip/overlay"

[ -f "$SRC" ] || { echo "Нет исходника: $SRC"; exit 1; }

if [ ! -d "$DEST" ]; then
	echo "Нет каталога оверлеев для ядра ${KVER}: $DEST"
	exit 1
fi

command -v dtc >/dev/null || { echo "Нет dtc: apt install device-tree-compiler"; exit 1; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

dtc -@ -I dts -O dtb -o "$TMP/$NAME" "$SRC" 2>/dev/null
cp "$TMP/$NAME" "$DEST/$NAME"
echo "Установлен: $DEST/$NAME"

# Перегенерация extlinux.conf; строка overlays берётся из /etc/default/u-boot
if command -v u-boot-update >/dev/null; then
	u-boot-update
fi

# Проверка, что оба оверлея попали в загрузочный конфиг
CONF=/boot/extlinux/extlinux.conf
if [ -f "$CONF" ]; then
	if grep -q 'rk3588-spi0-lowlat.dtbo' "$CONF"; then
		echo "OK: оверлей в $CONF"
	else
		echo "ВНИМАНИЕ: оверлея нет в $CONF."
		echo "Проверьте U_BOOT_FDT_OVERLAYS в /etc/default/u-boot:"
		grep -n 'U_BOOT_FDT_OVERLAYS' /etc/default/u-boot || true
		exit 1
	fi
fi

echo
echo "Перезагрузитесь, затем проверьте:"
echo "  dmesg | grep 'feb00000.spi'   # должно быть poll=1"
