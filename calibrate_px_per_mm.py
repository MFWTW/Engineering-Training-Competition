#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
px_per_mm 标定脚本（标尺法）
═════════════════════════════

用途：
    测量 config.yaml 里 chassis.px_per_mm 的实际数值。
    px_per_mm 表示“下位机报 1mm 移动，在检测画面里对应多少像素”。

原理：
    在物块实际工作距离处放一把尺子（或已知直径的物块），
    鼠标点两个端点，已知实际距离为 distance_mm，
    则 px_per_mm = 两点像素距离 / 实际距离(mm)。

用法：
    python3 calibrate_px_per_mm.py
    python3 calibrate_px_per_mm.py --distance-mm 100
    python3 calibrate_px_per_mm.py --no-save

按键：
    鼠标左键    点两点（第二点落下后自动记为一组）
    R          清除当前未完成的第 1 点
    U          撤销最后一组
    S          保存平均值到 config.yaml 并退出
    Q          退出（不保存）
"""

import argparse
import math
from pathlib import Path

import cv2

from common_camera import (
    open_camera,
    DETECTION_CAMERA_SOURCE,
    DETECTION_FRAME_WIDTH,
    DETECTION_FRAME_HEIGHT,
)
from felling_color import CONFIG, save_config


CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
WINDOW_NAME = "px_per_mm_calib"


class RulerCalibrator:
    """维护鼠标点选的样本：两点一组，自动计算 px_per_mm。"""

    def __init__(self, distance_mm: float):
        self.distance_mm = distance_mm
        self.current_pt = None          # 已点第 1 点，等待第 2 点
        self.samples = []               # 每组: p1/p2/dist_px/angle_deg/px_per_mm

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.current_pt is None:
            self.current_pt = (x, y)
            return

        p1 = self.current_pt
        p2 = (x, y)
        self.current_pt = None

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist_px = math.hypot(dx, dy)
        if dist_px < 1.0:
            print("[提示] 两点几乎重合，已忽略，请重新点选")
            return

        angle = abs(math.degrees(math.atan2(dy, dx)))
        if angle > 90.0:
            angle = 180.0 - angle

        sample = {
            "p1": p1,
            "p2": p2,
            "dist_px": dist_px,
            "angle_deg": angle,
            "px_per_mm": dist_px / self.distance_mm,
        }
        self.samples.append(sample)

        print(
            f"[样本 {len(self.samples)}] 像素距离={dist_px:.1f}px, "
            f"实际={self.distance_mm:.1f}mm, px_per_mm={sample['px_per_mm']:.4f}, "
            f"角度={angle:.1f}°"
        )
        if 10.0 < angle < 80.0:
            print(
                "[警告] 两点既不是水平也不是垂直。"
                "左右移动应让两点尽量水平（0°/180°），前后移动应尽量垂直（90°）。"
            )

    @property
    def mean(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s["px_per_mm"] for s in self.samples) / len(self.samples)


def draw_state(frame, cal: RulerCalibrator):
    """在画面上叠加标定状态，便于对齐标尺/物块。"""
    disp = frame.copy()

    for i, s in enumerate(cal.samples):
        p1, p2 = s["p1"], s["p2"]
        cv2.line(disp, p1, p2, (0, 255, 0), 2)
        cv2.circle(disp, p1, 5, (0, 255, 0), -1)
        cv2.circle(disp, p2, 5, (0, 255, 0), -1)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 10)
        cv2.putText(
            disp,
            f"#{i + 1} {s['px_per_mm']:.3f}px/mm",
            mid,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    if cal.current_pt is not None:
        cv2.circle(disp, cal.current_pt, 6, (0, 255, 255), -1)
        cv2.putText(
            disp,
            "Point 1 (click point 2)",
            (cal.current_pt[0] + 10, cal.current_pt[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

    mean = cal.mean
    lines = [
        f"Real distance: {cal.distance_mm:.1f} mm",
        f"Samples: {len(cal.samples)} | Avg: {mean:.4f} px/mm",
        "L-click 2 pts | R reset | U undo | S save | Q quit",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            disp,
            text,
            (10, 25 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            4,
        )
        cv2.putText(
            disp,
            text,
            (10, 25 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

    return disp


def parse_args():
    parser = argparse.ArgumentParser(
        description="标尺法测量 config.yaml 的 chassis.px_per_mm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--distance-mm",
        type=float,
        default=100.0,
        help="鼠标两个端点之间的实际距离（mm），比如尺子上两格 100mm",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(DETECTION_CAMERA_SOURCE),
        help="摄像头编号或 /dev/videoX 路径",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DETECTION_FRAME_WIDTH,
        help="采集宽度，需与主程序一致",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DETECTION_FRAME_HEIGHT,
        help="采集高度，需与主程序一致",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="只打印结果，不写回 config.yaml",
    )
    return parser.parse_args()


def save_result(px_per_mm: float, no_save: bool):
    if no_save:
        print(f"\n[未保存] config.yaml 中请手动填写:")
        print(f"  chassis:")
        print(f"    px_per_mm: {px_per_mm:.6f}")
        return

    cfg = CONFIG
    cfg.setdefault("chassis", {})["px_per_mm"] = round(px_per_mm, 6)
    save_config(cfg)
    print(f"\n[已保存] {CONFIG_PATH}")
    print(f"  chassis.px_per_mm = {px_per_mm:.6f}")


def main():
    args = parse_args()
    if args.distance_mm <= 0:
        print("错误: --distance-mm 必须大于 0")
        return

    print("=" * 60)
    print("px_per_mm 标定")
    print("=" * 60)
    print(
        "操作步骤：\n"
        "  1. 把尺子（或已知直径的物块）放在物块实际工作距离处；\n"
        "  2. 测左右移动比例：让两点尽量水平，点左端和右端；\n"
        "  3. 若前后移动更重要，把尺子沿前后方向放，让两点尽量垂直；\n"
        "  4. 多测几组取平均，按 S 保存，Q 退出。"
    )
    print(f"\n当前相机: source={args.source}, {args.width}x{args.height}")
    print(f"两点实际距离: {args.distance_mm:.1f} mm\n")

    cap = open_camera(args.source, args.width, args.height)
    if cap is None or not cap.isOpened():
        print("无法打开摄像头，请检查 source 参数")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_w, actual_h) != (args.width, args.height):
        print(
            f"[警告] 实际分辨率 {actual_w}x{actual_h} 与请求的 "
            f"{args.width}x{args.height} 不一致，请按实际分辨率重新标定"
        )

    cal = RulerCalibrator(args.distance_mm)
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, cal.on_mouse)
    except cv2.error as e:
        print("\n[错误] 无法创建显示窗口:")
        print(f"  {e}")
        print(
            "  可能是在无图形界面/SSH 下运行。\n"
            "  请在带桌面的本机终端运行，或先配置 X11 转发后再试。"
        )
        cap.release()
        cv2.destroyAllWindows()
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            cv2.imshow(WINDOW_NAME, draw_state(frame, cal))
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                cal.current_pt = None
                print("[操作] 已清除当前第 1 点")
            elif key == ord("u"):
                if cal.samples:
                    removed = cal.samples.pop()
                    print(
                        f"[操作] 已撤销样本 #{len(cal.samples) + 1} "
                        f"(px_per_mm={removed['px_per_mm']:.4f})"
                    )
                else:
                    print("[操作] 没有可撤销的样本")
            elif key == ord("s"):
                if not cal.samples:
                    print("[提示] 还没有样本，请先点两组端点")
                    continue
                print(f"\n共 {len(cal.samples)} 组，平均 px_per_mm = {cal.mean:.6f}")
                save_result(cal.mean, args.no_save)
                break
            elif key == ord("q"):
                print("\n已退出，未保存")
                break
    except KeyboardInterrupt:
        print("\n用户手动终止")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
