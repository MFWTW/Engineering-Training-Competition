#!/usr/bin/env python3
"""从终端日志提取底盘/夹爪数据并绘制曲线。

用法:
    python3 plot_log.py [日志文件.txt] [输出.png] [--last-run]

支持的行格式（src.py 的打印）:
    [抓取] ... 底盘移动量=(-80,-120)mm 夹爪伸长量=97mm | 下位机回传=(-100,-120)mm ...
    [抓取] ... 底盘目标位置=(-80,-120)mm 夹爪伸长量=97mm | 下位机回传=(-100,-120)mm ...
    [RX] chassis=(-179,-240)mm v=...
"""

import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


GRAB_RE = re.compile(
    r"(?:底盘移动量|底盘目标位置)=\((?P<tx>[+-]?\d+),(?P<ty>[+-]?\d+)\)mm "
    r"夹爪伸长量=(?P<g>\d+)mm \| "
    r"下位机回传=\((?P<fx>[+-]?\d+),(?P<fy>[+-]?\d+)\)mm"
)
RX_RE = re.compile(r"\[RX\] chassis=\((?P<fx>[+-]?\d+),(?P<fy>[+-]?\d+)\)mm")
RUN_START_RE = re.compile(
    r"^===== 程序启动 (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ====="
)
RUN_END_RE = re.compile(r"^===== 程序退出 (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def find_last_run(path):
    """返回 (start_line, end_line, 启动时间)；end_line 为退出标记行，可为 None。"""
    starts, ends = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            m = RUN_START_RE.search(line)
            if m:
                starts.append((i, m.group(1)))
            if RUN_END_RE.search(line):
                ends.append(i)
    if not starts:
        return None
    start_line, ts = starts[-1]
    end_line = next((e for e in ends if e > start_line), None)
    return start_line, end_line, ts


def parse_log(path, start_line=0, end_line=None):
    """返回 [{idx, tx?, ty?, g?, fx, fy}, ...]，idx 为该段内的日志行号"""
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            if end_line is not None and i >= end_line:
                break
            m = GRAB_RE.search(line)
            if m:
                rows.append({
                    "idx": i - start_line,
                    "tx": int(m["tx"]),
                    "ty": int(m["ty"]),
                    "g": int(m["g"]),
                    "fx": int(m["fx"]),
                    "fy": int(m["fy"]),
                })
                continue
            m = RX_RE.search(line)
            if m:
                rows.append({
                    "idx": i - start_line,
                    "fx": int(m["fx"]),
                    "fy": int(m["fy"]),
                })
    return rows


def plot_log(path, out_png, last_run=False):
    if last_run:
        bounds = find_last_run(path)
        if bounds is None:
            print("未找到启动/退出标记，按整个文件绘制")
            start_line, end_line, ts = 0, None, None
        else:
            start_line, end_line, ts = bounds
    else:
        start_line, end_line, ts = 0, None, None

    rows = parse_log(path, start_line=start_line, end_line=end_line)
    if not rows:
        print("日志中没有找到可解析的数据行")
        return 1

    idx = [r["idx"] for r in rows]
    fx = [r["fx"] for r in rows]
    fy = [r["fy"] for r in rows]
    tx = [r.get("tx") for r in rows]
    ty = [r.get("ty") for r in rows]
    g = [r.get("g") for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    if last_run and ts:
        fig.suptitle(
            f"Absolute Chassis Target vs Feedback  (last run {ts}, x = log line)",
            fontsize=13,
        )
    else:
        fig.suptitle(
            "Absolute Chassis Target vs Feedback  (x = log line)", fontsize=13
        )

    # 1) 底盘 X：目标 vs 回传
    ax = axes[0]
    ax.plot(idx, fx, "-o", ms=3, color="#1f77b4", label="feedback X (abs mm)")
    txi = [(i, v) for i, v in zip(idx, tx) if v is not None]
    if txi:
        ax.plot([p[0] for p in txi], [p[1] for p in txi], "--s", ms=3,
                color="#ff7f0e", label="target X (abs mm)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("X (mm)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    # 2) 底盘 Y
    ax = axes[1]
    ax.plot(idx, fy, "-o", ms=3, color="#2ca02c", label="feedback Y (abs mm)")
    tyi = [(i, v) for i, v in zip(idx, ty) if v is not None]
    if tyi:
        ax.plot([p[0] for p in tyi], [p[1] for p in tyi], "--s", ms=3,
                color="#d62728", label="target Y (abs mm)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Y (mm)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    # 3) 夹爪伸长量（目标）
    ax = axes[2]
    gi = [(i, v) for i, v in zip(idx, g) if v is not None]
    if gi:
        ax.plot([p[0] for p in gi], [p[1] for p in gi], "-o", ms=3,
                color="#9467bd", label="gripper extension")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("gripper (mm)")
    ax.set_xlabel("log line")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"已保存: {out_png}  （解析 {len(rows)} 行）")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从终端日志提取底盘/夹爪数据并绘制曲线"
    )
    parser.add_argument("log", nargs="?", default="log.txt",
                        help="日志文件（默认 log.txt）")
    parser.add_argument("out", nargs="?", default="log_plot.png",
                        help="输出图片（默认 log_plot.png）")
    parser.add_argument("--last-run", action="store_true",
                        help="只绘制最近一次运行（按启动/退出时间戳分隔）")
    args = parser.parse_args()
    raise SystemExit(plot_log(args.log, args.out, last_run=args.last_run))
