#!/usr/bin/env python3
"""
HSV 颜色阈值实时调试工具
─────────────────────────────
- 调用物块检测 USB 摄像头实时取流
- Trackbar 调整各颜色 HSV 上下界
- 实时显示：原始+ROI / 掩码 / 掩码叠加
- 按 's' 保存当前颜色阈值到 config.yaml
- 按 'r' 开/关 ROI 框显示
- 按 'q' 退出
"""

import copy

import cv2
import numpy as np

from common_camera import (
    open_camera,
    DETECTION_CAMERA_SOURCE,
    DETECTION_FRAME_WIDTH,
    DETECTION_FRAME_HEIGHT,
)
from felling_color import (
    CONFIG,
    COLOR_KEYS,
    color_thresholds,
    MORPH_PARAMS,
    save_config as save_config_yaml,
)


# ══════════════════════════════════════════════════════
# 颜色列表与初始阈值（从 config.yaml 加载，与检测代码共用一份配置）
# ══════════════════════════════════════════════════════

COLOR_NAMES = list(COLOR_KEYS)
# 画面显示用英文标签（OpenCV 默认字体不支持中文，中文会乱码）
COLOR_LABEL_EN = {
    'red': 'RED',
    'green': 'GREEN',
    'blue': 'BLUE',
    'light_blue': 'LIGHT_BLUE',
    'black': 'BLACK',
    'yellow': 'YELLOW',
}
COLOR_LABELS = [' ' + COLOR_LABEL_EN.get(n, n) for n in COLOR_NAMES]

# 深拷贝一份给 trackbar 运行时修改
thresholds = copy.deepcopy(color_thresholds)

# ROI 区域 (x, y, w, h)：直接跟随检测实际使用的 detection_area，
# 避免调试画面里的绿色框和真正裁剪区域不一致；为 null 时不画框。
_DETECTION_AREA = CONFIG['detection'].get('detection_area')
ROI = tuple(_DETECTION_AREA) if _DETECTION_AREA else None

# 形态学核（与检测代码一致）
KERNEL = np.ones((int(CONFIG['detection']['kernel_size']),) * 2, np.uint8)


# ══════════════════════════════════════════════════════
# Trackbar 辅助
# ══════════════════════════════════════════════════════

def nothing(_):
    pass


def sync_trackbars(win, color):
    """把滑条位置同步到当前颜色的阈值，不重建窗口，避免闪烁"""
    cv2.setWindowTitle(win, f"HSV Tuner - {COLOR_LABEL_EN[color]}")
    t = thresholds[color]

    if color == 'red':
        cv2.setTrackbarPos('H_low',  win, t['lower1'][0])
        cv2.setTrackbarPos('H_high', win, t['upper1'][0])
        cv2.setTrackbarPos('S_low',  win, t['lower1'][1])
        cv2.setTrackbarPos('S_high', win, t['upper1'][1])
        cv2.setTrackbarPos('V_low',  win, t['lower1'][2])
        cv2.setTrackbarPos('V_high', win, t['upper1'][2])
        cv2.setTrackbarPos('H_low2',  win, t['lower2'][0])
        cv2.setTrackbarPos('H_high2', win, t['upper2'][0])
        cv2.setTrackbarPos('S_low2',  win, t['lower2'][1])
        cv2.setTrackbarPos('S_high2', win, t['upper2'][1])
        cv2.setTrackbarPos('V_low2',  win, t['lower2'][2])
        cv2.setTrackbarPos('V_high2', win, t['upper2'][2])
    else:
        cv2.setTrackbarPos('H_low',  win, t['lower'][0])
        cv2.setTrackbarPos('H_high', win, t['upper'][0])
        cv2.setTrackbarPos('S_low',  win, t['lower'][1])
        cv2.setTrackbarPos('S_high', win, t['upper'][1])
        cv2.setTrackbarPos('V_low',  win, t['lower'][2])
        cv2.setTrackbarPos('V_high', win, t['upper'][2])


def read_trackbars(win, color):
    """从 trackbar 读取值并写入 thresholds"""
    t = thresholds[color]
    low = [
        cv2.getTrackbarPos('H_low',  win),
        cv2.getTrackbarPos('S_low',  win),
        cv2.getTrackbarPos('V_low',  win),
    ]
    high = [
        cv2.getTrackbarPos('H_high', win),
        cv2.getTrackbarPos('S_high', win),
        cv2.getTrackbarPos('V_high', win),
    ]
    if color == 'red':
        t['lower1'] = low
        t['upper1'] = high
        t['lower2'] = [cv2.getTrackbarPos('H_low2', win),
                       cv2.getTrackbarPos('S_low2', win),
                       cv2.getTrackbarPos('V_low2', win)]
        t['upper2'] = [cv2.getTrackbarPos('H_high2', win),
                       cv2.getTrackbarPos('S_high2', win),
                       cv2.getTrackbarPos('V_high2', win)]
    else:
        t['lower'] = low
        t['upper'] = high


def apply_mask(hsv, color):
    """对 HSV 图像应用当前阈值"""
    t = thresholds[color]
    if color == 'red':
        m1 = cv2.inRange(hsv, np.array(t['lower1']), np.array(t['upper1']))
        m2 = cv2.inRange(hsv, np.array(t['lower2']), np.array(t['upper2']))
        return cv2.bitwise_or(m1, m2)
    return cv2.inRange(hsv, np.array(t['lower']), np.array(t['upper']))


# ══════════════════════════════════════════════════════
# 保存配置到 YAML
# ══════════════════════════════════════════════════════

def save_to_yaml():
    """把 trackbar 调好的阈值写回 config.yaml（检测代码下次运行即生效）"""
    for name in COLOR_NAMES:
        CONFIG['colors'][name].update(thresholds[name])
    save_config_yaml(CONFIG)
    print("\n" + "─" * 62)
    print("  ✅ 已保存颜色阈值到 config.yaml")
    for name in COLOR_NAMES:
        t = thresholds[name]
        if name == 'red':
            print(f"    red: lower1={t['lower1']} upper1={t['upper1']} "
                  f"lower2={t['lower2']} upper2={t['upper2']}")
        else:
            print(f"    {name}: lower={t['lower']} upper={t['upper']}")
    print("─" * 62 + "\n")


# ══════════════════════════════════════════════════════
# 主循环
# ══════════════════════════════════════════════════════

def main():
    # ── 初始化物块检测 USB 摄像头 ──
    cap = open_camera(
        DETECTION_CAMERA_SOURCE,
        width=DETECTION_FRAME_WIDTH,
        height=DETECTION_FRAME_HEIGHT,
    )
    if cap is None:
        print("❌ 无法打开物块检测 USB 摄像头，退出。")
        return
    print("✅ 物块检测 USB 摄像头已就绪")

    # ── 创建窗口 ──
    TB_WIN = "HSV Trackbars"
    cv2.namedWindow(TB_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TB_WIN, 540, 760)

    # 滑条只创建一次；切颜色时只更新位置，不销毁重建窗口
    cv2.createTrackbar('Color', TB_WIN, 0, 5, nothing)
    cv2.createTrackbar('H_low',  TB_WIN, 0, 180, nothing)
    cv2.createTrackbar('H_high', TB_WIN, 0, 180, nothing)
    cv2.createTrackbar('S_low',  TB_WIN, 0, 255, nothing)
    cv2.createTrackbar('S_high', TB_WIN, 0, 255, nothing)
    cv2.createTrackbar('V_low',  TB_WIN, 0, 255, nothing)
    cv2.createTrackbar('V_high', TB_WIN, 0, 255, nothing)
    cv2.createTrackbar('---  Range2 (red)  ---', TB_WIN, 0, 1, nothing)
    cv2.createTrackbar('H_low2',  TB_WIN, 0, 180, nothing)
    cv2.createTrackbar('H_high2', TB_WIN, 0, 180, nothing)
    cv2.createTrackbar('S_low2',  TB_WIN, 0, 255, nothing)
    cv2.createTrackbar('S_high2', TB_WIN, 0, 255, nothing)
    cv2.createTrackbar('V_low2',  TB_WIN, 0, 255, nothing)
    cv2.createTrackbar('V_high2', TB_WIN, 0, 255, nothing)

    # 初始化为红色
    sync_trackbars(TB_WIN, 'red')

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Masked Result", cv2.WINDOW_NORMAL)

    print("\n┌──────────────────────────────────────────┐")
    print("│  🎮 操作说明                             │")
    print("│  拖动 Trackbar 调整 HSV 上下界           │")
    print("│  Color slider 切换要调试的颜色            │")
    print("│  按 's' 保存当前颜色阈值到 YAML          │")
    print("│  按 'r' 开/关 ROI 框显示                 │")
    print("│  按 'q' 退出                             │")
    print("└──────────────────────────────────────────┘\n")

    prev_color = 0
    show_roi = True

    while True:
        # ── 获取一帧 ──
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # ── 检测颜色切换 ──
        cur_color_idx = cv2.getTrackbarPos('Color', TB_WIN)
        if cur_color_idx != prev_color:
            prev_color = cur_color_idx
            sync_trackbars(TB_WIN, COLOR_NAMES[cur_color_idx])

        color_name = COLOR_NAMES[cur_color_idx]

        # ── 读取 trackbar ──
        read_trackbars(TB_WIN, color_name)

        # ── 全图检测 ──
        roi = frame
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = apply_mask(hsv_roi, color_name)
        e_iter, d_iter = MORPH_PARAMS[color_name]
        mask = cv2.erode(mask, KERNEL, iterations=e_iter)
        mask = cv2.dilate(mask, KERNEL, iterations=d_iter)

        masked = cv2.bitwise_and(roi, roi, mask=mask)

        # ── 显示 ──
        disp = frame.copy()
        if show_roi and ROI is not None:
            rx, ry, rw, rh = [int(v) for v in ROI]
            hh, ww = disp.shape[:2]
            rx = max(0, min(rx, ww - 1))
            ry = max(0, min(ry, hh - 1))
            rw = max(0, min(rw, ww - rx))
            rh = max(0, min(rh, hh - ry))
            cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv2.putText(disp, "ROI", (rx + 5, ry + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(disp, COLOR_LABELS[cur_color_idx], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ── 叠加显示当前颜色的 HSV 阈值数字 ──
        t = thresholds[color_name]
        lines = []
        if color_name == 'red':
            lines.append(
                f"H1 {t['lower1'][0]}-{t['upper1'][0]}  "
                f"S1 {t['lower1'][1]}-{t['upper1'][1]}  "
                f"V1 {t['lower1'][2]}-{t['upper1'][2]}"
            )
            lines.append(
                f"H2 {t['lower2'][0]}-{t['upper2'][0]}  "
                f"S2 {t['lower2'][1]}-{t['upper2'][1]}  "
                f"V2 {t['lower2'][2]}-{t['upper2'][2]}"
            )
        else:
            lines.append(
                f"H {t['lower'][0]}-{t['upper'][0]}  "
                f"S {t['lower'][1]}-{t['upper'][1]}  "
                f"V {t['lower'][2]}-{t['upper'][2]}"
            )
        for i, line in enumerate(lines):
            cv2.putText(disp, line, (10, 62 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        cv2.imshow("Original", disp)

        cv2.imshow("Mask", mask)
        cv2.imshow("Masked Result", masked)

        # ── 按键 ──
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_to_yaml()
        elif key == ord('r'):
            show_roi = not show_roi

    # ── 清理 ──
    cap.release()
    cv2.destroyAllWindows()
    print("👋 已退出")


if __name__ == "__main__":
    main()
