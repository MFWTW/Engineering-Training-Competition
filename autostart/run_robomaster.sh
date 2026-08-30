#!/usr/bin/env bash
# RoboMaster 主程序开机启动脚本
#
# 由 systemd 用户服务 robomaster.service 调用；
# 也可以手动运行：bash /home/xu/Engineer/autostart/run_robomaster.sh
#
# 职责：
#   1. 进入项目目录，使用项目虚拟环境（conda/bin/python3）
#   2. 配置 DISPLAY / XAUTHORITY，等待图形界面就绪
#   3. 等待 USB 摄像头枚举完成（/dev/video* 出现）
#   4. exec 启动 src.py，让 systemd 直接管理主程序进程
set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_DIR/conda/bin/python3"
MAIN_SCRIPT="$PROJECT_DIR/src.py"

cd "$PROJECT_DIR"

# ---------- 显示环境 ----------
if [ -z "${DISPLAY:-}" ]; then
    export DISPLAY=":0"
fi
if [ -z "${XAUTHORITY:-}" ]; then
    if [ -r "/run/user/1000/gdm/Xauthority" ]; then
        export XAUTHORITY="/run/user/1000/gdm/Xauthority"
    elif [ -r "$HOME/.Xauthority" ]; then
        export XAUTHORITY="$HOME/.Xauthority"
    fi
fi

# ---------- 等待图形界面就绪（最多 60 秒） ----------
DISPLAY_NUM="${DISPLAY#*:}"
DISPLAY_NUM="${DISPLAY_NUM%%.*}"
for _ in $(seq 1 60); do
    if [ -e "/tmp/.X11-unix/X${DISPLAY_NUM}" ] && \
       { [ -z "${XAUTHORITY:-}" ] || [ -r "$XAUTHORITY" ]; }; then
        break
    fi
    sleep 1
done

# ---------- 等待 USB 摄像头枚举（最多 120 秒） ----------
for _ in $(seq 1 60); do
    if compgen -G "/dev/video*" >/dev/null; then
        break
    fi
    sleep 2
done

# ---------- 等待串口设备出现（不超时） ----------
# 串口未连接时保持等待；连接上后自动启动主程序。
# 主程序运行中串口断开会自动退出，systemd 重新拉起本脚本并再次等待，
# 所以重新插上串口后程序会自动从头开始运行。
echo "[autostart] 等待串口设备（ttyACM*/ttyUSB*）..."
while ! compgen -G "/dev/ttyACM*" >/dev/null && \
      ! compgen -G "/dev/ttyUSB*" >/dev/null; do
    sleep 2
done
echo "[autostart] 检测到串口设备，启动主程序"

# ---------- 启动主程序 ----------
exec "$PYTHON" "$MAIN_SCRIPT"
