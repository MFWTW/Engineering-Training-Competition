#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qr_display.py —— 外接屏幕大字显示二维码扫码结果

独立运行（src.py 会自动拉起本程序，也可以手动启动）：
    python3 qr_display.py

常用参数：
    --file /tmp/xxx      指定扫码结果状态文件（默认 /tmp/qr_display_result.txt，
                         也可用环境变量 QR_DISPLAY_FILE 指定）
    --monitor 1          指定显示器序号（0 起）；默认自动选外接屏
                         （HDMI/DP/VGA/DVI 优先，多屏时其次选第 2 个屏）
    --text 156+123       直接显示固定内容，不监视文件（调试用）
    --font-scale 0.16    大字大小占屏幕高度的比例
    --replace            若已有旧显示实例，先请其退出再接管（默认不替换）

工作方式：scan_QRcode_andlist.py 每次扫到二维码会把原始内容写入状态文件；
本程序检测到文件变化后，在外接屏幕上用超大字体显示该内容。
如果窗口没有出现在外接屏上，先指定 DISPLAY，例如：
    DISPLAY=:10 python3 qr_display.py --monitor 1

本进程会忽略 SIGHUP/SIGPIPE，并可在独立会话中运行；由 src.py 拉起时，
即使主程序关闭/终端关闭，显示窗口也会继续保留最后结果。
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import tkinter as tk
import tkinter.font as tkfont

try:
    import fcntl
except ImportError:  # 非 Linux 环境（如 Windows）跳过单实例锁
    fcntl = None

DEFAULT_STATE_FILE = "/tmp/qr_display_result.txt"
LOCK_FILE = "/tmp/qr_display.lock"
WAIT_TEXT = ""

_lock_fd = None


def _read_lock_pid():
    """读取锁文件里的旧进程 PID；文件不存在或内容非法返回 None。"""
    try:
        return int(Path(LOCK_FILE).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _pid_exists(pid):
    """pid 是否还存在（不一定是 qr_display 进程）。"""
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # 存在但无权限探测，按存活处理
    return True


def _pid_is_qr_display(pid):
    """确认 pid 确实是 qr_display.py，避免误杀其它程序。
    返回 True/False；无法读取 /proc 时返回 None（不确定）。"""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return b"qr_display.py" in f.read()
    except OSError:
        return None


def _acquire_single_instance(replace=False):
    """单实例锁：已有 qr_display 在运行则返回 False（避免多个全屏窗口叠加）。

    replace=True 且旧进程确认为 qr_display.py 时，先发 SIGTERM 请旧实例退出，
    再重新抢锁接管显示，避免旧进程占锁导致新进程一直起不来。
    """
    global _lock_fd
    if fcntl is None:
        return True

    attempts = 2 if replace else 1
    for _ in range(attempts):
        try:
            _lock_fd = open(LOCK_FILE, "a+", encoding="utf-8")
            fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _lock_fd.seek(0)
            _lock_fd.truncate()
            _lock_fd.write(str(os.getpid()) + "\n")
            _lock_fd.flush()
            return True
        except OSError:
            if _lock_fd is not None:
                try:
                    _lock_fd.close()
                except OSError:
                    pass
                _lock_fd = None
            pid = _read_lock_pid()
            holder = str(pid) if pid is not None else "?"
            if replace:
                if not _pid_exists(pid):
                    continue  # 锁持有人已消失，锁应即将释放，重试一次
                if _pid_is_qr_display(pid) is not True:
                    print(
                        f"[QR显示] 已有进程占用锁 (pid={holder})，但不是本程序或无法确认，"
                        "不自动替换；请手动关闭后重试",
                        file=sys.stderr,
                    )
                    return False
                print(f"[QR显示] 替换旧显示实例 (pid={pid}) ...", file=sys.stderr)
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError as exc:
                    print(f"[QR显示] 通知旧实例退出失败: {exc}", file=sys.stderr)
                time.sleep(0.5)
                continue
            print(
                f"[QR显示] 已有实例在运行 (pid={holder})，本实例退出；如需替换请加 --replace",
                file=sys.stderr,
            )
            return False

    print("[QR显示] 替换旧实例失败（旧进程未退出），请手动关闭后重试", file=sys.stderr)
    return False


def _parse_xrandr():
    """用 xrandr 枚举显示器，返回 [{name, x, y, w, h}, ...]；失败返回 []。"""
    try:
        proc = subprocess.run(
            ["xrandr", "--query"], capture_output=True, text=True, timeout=3
        )
    except (OSError, subprocess.SubprocessError):
        return []

    monitors = []
    pattern = re.compile(
        r"^(\S+)\s+connected\s+(?:primary\s+)?(\d+)x(\d+)\+(\d+)\+(\d+)"
    )
    for line in proc.stdout.splitlines():
        match = pattern.match(line.strip())
        if match:
            name, width, height, x, y = match.groups()
            monitors.append({
                "name": name,
                "w": int(width),
                "h": int(height),
                "x": int(x),
                "y": int(y),
            })
    return monitors


def pick_monitor(monitors, index=None):
    """选择要显示的屏幕：指定序号 > 外接屏（HDMI/DP/VGA/DVI） > 第 2 个屏 > 唯一屏。"""
    if not monitors:
        return None
    if index is not None:
        return monitors[max(0, min(index, len(monitors) - 1))]
    if len(monitors) == 1:
        return monitors[0]
    for monitor in monitors:
        if re.match(r"(hdmi|dp|displayport|vga|dvi)", monitor["name"], re.I):
            return monitor
    return monitors[1]


class QRDisplay:
    """黑色全屏窗口，居中显示扫码结果大字。"""

    def __init__(self, text=WAIT_TEXT, monitor_index=None, font_scale=0.16):
        self.font_scale = font_scale
        self.root = tk.Tk()
        self.root.title("QR Display")
        self.root.configure(bg="black")

        monitors = _parse_xrandr()
        self.monitor = pick_monitor(monitors, monitor_index)
        if self.monitor is not None:
            m = self.monitor
            self.width = m["w"]
            self.height = m["h"]
            self.root.geometry(f"{m['w']}x{m['h']}+{m['x']}+{m['y']}")
            print(f"[QR显示] 使用显示器: {m['name']} ({m['w']}x{m['h']}@{m['x']},{m['y']})")
        else:
            self.width = self.root.winfo_screenwidth()
            self.height = self.root.winfo_screenheight()
            self.root.attributes("-fullscreen", True)
            print(f"[QR显示] 无法枚举显示器，使用默认全屏 ({self.width}x{self.height})")

        # 无边框 + 置顶，作为专用显示窗口
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        self.label = tk.Label(
            self.root,
            text="",
            fg="white",
            bg="black",
            justify="center",
            anchor="center",
            wraplength=max(100, int(self.width * 0.92)),
        )
        self.label.pack(fill="both", expand=True)
        self.set_text(text)

        self.root.bind("<Escape>", lambda _e: self.root.destroy())
        self.root.bind("<Key-q>", lambda _e: self.root.destroy())
        self.root.bind("<Key-Q>", lambda _e: self.root.destroy())

    def set_text(self, text):
        """更新显示内容，字号按屏幕宽度自动缩小以放下完整内容。"""
        text = (text or "").strip()
        shown = text if text else WAIT_TEXT
        self.label.configure(text=shown)

        size = max(20, int(self.height * self.font_scale))
        font = tkfont.Font(family="DejaVu Sans", size=size, weight="bold")
        max_width = self.width * 0.92
        while font.measure(shown) > max_width and size > 18:
            size -= 2
            font.configure(size=size)
        self.label.configure(font=font)

    def mainloop(self):
        self.root.mainloop()


def _watch_file(app, path, poll_ms=200):
    """监视状态文件，内容或时间戳变化时刷新屏幕。"""
    last_mtime = None

    def tick():
        nonlocal last_mtime
        try:
            stat = os.stat(path)
            if stat.st_mtime_ns != last_mtime:
                last_mtime = stat.st_mtime_ns
                text = Path(path).read_text(encoding="utf-8").strip()
                app.set_text(text)
        except FileNotFoundError:
            if last_mtime is not None:
                last_mtime = None
                app.set_text(WAIT_TEXT)
        except OSError as exc:
            print(f"[QR显示] 读取状态文件失败: {exc}")
        app.root.after(poll_ms, tick)

    app.root.after(poll_ms, tick)


def main(argv=None):
    # 忽略挂断/管道信号：终端或主程序关闭时显示进程继续存活
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)

    parser = argparse.ArgumentParser(description="外接屏幕大字显示二维码扫码结果")
    parser.add_argument(
        "--file",
        default=os.environ.get("QR_DISPLAY_FILE", DEFAULT_STATE_FILE),
        help="扫码结果状态文件（默认 %(default)s）",
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=None,
        help="显示器序号（0 起），默认自动选外接屏",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="直接显示固定内容，不监视文件（调试用）",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=0.16,
        help="大字高度占屏幕比例（默认 %(default)s）",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="已有显示实例时先请其退出再接管（默认不替换）",
    )
    args = parser.parse_args(argv)

    if not _acquire_single_instance(replace=args.replace):
        return 2

    try:
        app = QRDisplay(args.text or WAIT_TEXT, args.monitor, args.font_scale)
    except tk.TclError as exc:
        print(f"[QR显示] 无法打开显示窗口: {exc}")
        print("请确认设置了正确的 DISPLAY，例如: DISPLAY=:10 python3 qr_display.py")
        print("可用 xrandr --query 查看可用显示器；跨会话显示时还需设置 XAUTHORITY")
        return 1

    if args.text is None:
        print(f"[QR显示] 监视扫码结果文件: {args.file}")
        print("[QR显示] 扫码后外接屏自动显示识别数字（按 Esc / Q 或 Ctrl+C 退出）")
        _watch_file(app, args.file)
    else:
        print(f"[QR显示] 固定显示: {args.text}")

    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
