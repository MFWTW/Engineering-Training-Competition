#!/usr/bin/env python3
"""
HSV 颜色阈值实时调试工具
─────────────────────────────
- 调用海康工业相机实时取流（无海康时自动回退到 USB 摄像头）
- Trackbar 调整各颜色 HSV 上下界
- 实时显示：原始+ROI / 掩码 / 掩码叠加
- 按 's' 打印当前所有颜色的 color_thresholds 配置
- 按 'r' 开/关 ROI 框显示
- 按 'q' 退出
"""

import cv2
import numpy as np
import sys

from hikrobot_camera import enum_devices, create_camera_handle, start_grabbing, read_frame


# ══════════════════════════════════════════════════════
# 颜色列表与初始阈值（与 felling_color.py 保持同步）
# ══════════════════════════════════════════════════════

COLOR_NAMES = ['red', 'green', 'blue', 'light_blue', 'black', 'yellow']
COLOR_LABELS = [' 红色', ' 绿色', ' 蓝色', ' 浅蓝', ' 黑色', ' 黄色']

INITIAL = {
    'red': {'lower1': [0, 43, 46], 'upper1': [10, 255, 255],
            'lower2': [156, 43, 46], 'upper2': [180, 255, 255]},
    'green':      {'lower': [35, 43, 46],  'upper': [77, 255, 255]},
    'blue':       {'lower': [100, 43, 46], 'upper': [124, 255, 150]},
    'light_blue': {'lower': [85, 40, 150], 'upper': [100, 200, 255]},
    'black':      {'lower': [0, 0, 0],     'upper': [180, 255, 46]},
    'yellow':     {'lower': [26, 43, 46],  'upper': [34, 255, 255]},
}

# 深拷贝一份给 trackbar 运行时修改
import copy
thresholds = copy.deepcopy(INITIAL)

# ROI 区域 (x, y, w, h)
ROI = (120, 80, 400, 217)


# ══════════════════════════════════════════════════════
# Trackbar 辅助
# ══════════════════════════════════════════════════════

def nothing(_):
    pass


def make_trackbars(win, color):
    """为指定颜色创建 HSV 上下界 trackbar"""
    cv2.setWindowTitle(win, f"HSV Tuner ── {COLOR_LABELS[COLOR_NAMES.index(color)]}")
    t = thresholds[color]

    if color == 'red':
        # 范围1
        cv2.createTrackbar('H_low1',  win, t['lower1'][0], 180, nothing)
        cv2.createTrackbar('H_high1', win, t['upper1'][0], 180, nothing)
        cv2.createTrackbar('S_low1',  win, t['lower1'][1], 255, nothing)
        cv2.createTrackbar('S_high1', win, t['upper1'][1], 255, nothing)
        cv2.createTrackbar('V_low1',  win, t['lower1'][2], 255, nothing)
        cv2.createTrackbar('V_high1', win, t['upper1'][2], 255, nothing)
        # 范围2
        cv2.createTrackbar('---  Range2  ---', win, 0, 1, nothing)
        cv2.createTrackbar('H_low2',  win, t['lower2'][0], 180, nothing)
        cv2.createTrackbar('H_high2', win, t['upper2'][0], 180, nothing)
        cv2.createTrackbar('S_low2',  win, t['lower2'][1], 255, nothing)
        cv2.createTrackbar('S_high2', win, t['upper2'][1], 255, nothing)
        cv2.createTrackbar('V_low2',  win, t['lower2'][2], 255, nothing)
        cv2.createTrackbar('V_high2', win, t['upper2'][2], 255, nothing)
    else:
        cv2.createTrackbar('H_low',  win, t['lower'][0], 180, nothing)
        cv2.createTrackbar('H_high', win, t['upper'][0], 180, nothing)
        cv2.createTrackbar('S_low',  win, t['lower'][1], 255, nothing)
        cv2.createTrackbar('S_high', win, t['upper'][1], 255, nothing)
        cv2.createTrackbar('V_low',  win, t['lower'][2], 255, nothing)
        cv2.createTrackbar('V_high', win, t['upper'][2], 255, nothing)


def read_trackbars(win, color):
    """从 trackbar 读取值并写入 thresholds"""
    t = thresholds[color]
    if color == 'red':
        t['lower1'] = [cv2.getTrackbarPos('H_low1', win),
                       cv2.getTrackbarPos('S_low1', win),
                       cv2.getTrackbarPos('V_low1', win)]
        t['upper1'] = [cv2.getTrackbarPos('H_high1', win),
                       cv2.getTrackbarPos('S_high1', win),
                       cv2.getTrackbarPos('V_high1', win)]
        t['lower2'] = [cv2.getTrackbarPos('H_low2', win),
                       cv2.getTrackbarPos('S_low2', win),
                       cv2.getTrackbarPos('V_low2', win)]
        t['upper2'] = [cv2.getTrackbarPos('H_high2', win),
                       cv2.getTrackbarPos('S_high2', win),
                       cv2.getTrackbarPos('V_high2', win)]
    else:
        t['lower'] = [cv2.getTrackbarPos('H_low', win),
                      cv2.getTrackbarPos('S_low', win),
                      cv2.getTrackbarPos('V_low', win)]
        t['upper'] = [cv2.getTrackbarPos('H_high', win),
                      cv2.getTrackbarPos('S_high', win),
                      cv2.getTrackbarPos('V_high', win)]


def apply_mask(hsv, color):
    """对 HSV 图像应用当前阈值"""
    t = thresholds[color]
    if color == 'red':
        m1 = cv2.inRange(hsv, np.array(t['lower1']), np.array(t['upper1']))
        m2 = cv2.inRange(hsv, np.array(t['lower2']), np.array(t['upper2']))
        return cv2.bitwise_or(m1, m2)
    return cv2.inRange(hsv, np.array(t['lower']), np.array(t['upper']))


# ══════════════════════════════════════════════════════
# 打印配置
# ══════════════════════════════════════════════════════

def dump_config():
    print("\n" + "─" * 62)
    print("  📋  当前 color_thresholds 配置（复制到 felling_color.py）")
    print("─" * 62)
    print("color_thresholds = {")
    for name in COLOR_NAMES:
        t = thresholds[name]
        if name == 'red':
            print(f"    'red': {{'lower1': {t['lower1']}, 'upper1': {t['upper1']},")
            print(f"            'lower2': {t['lower2']}, 'upper2': {t['upper2']}}},")
        else:
            print(f"    '{name}': {{'lower': {t['lower']}, 'upper': {t['upper']}}},")
    print("}")
    print("─" * 62 + "\n")


# ══════════════════════════════════════════════════════
# 主循环
# ══════════════════════════════════════════════════════

def main():
    # ── 初始化相机 ──
    print("🔍 枚举设备...")
    dev_list = enum_devices()
    use_hik = False
    cam = None
    cap = None

    if dev_list is not None:
        cam = create_camera_handle(dev_list, 0,)
        if cam is not None and start_grabbing(cam):
            use_hik = True
            print("✅ 海康相机已就绪")
        else:
            print("⚠️ 海康相机打开失败，尝试 USB 摄像头...")
    else:
        print("⚠️ 未发现海康设备，尝试 USB 摄像头...")

    if not use_hik:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 无法打开任何摄像头，退出。")
            return
        print("✅ USB 摄像头已就绪")

    # ── 创建窗口 ──
    TB_WIN = "HSV Trackbars"
    cv2.namedWindow(TB_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TB_WIN, 540, 400)

    # 颜色选择 slider
    cv2.createTrackbar('Color 0红 1绿 2蓝 3浅蓝 4黑 5黄', TB_WIN, 0, 5, nothing)

    # 初始化为红色
    make_trackbars(TB_WIN, 'red')

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Masked Result", cv2.WINDOW_NORMAL)

    print("\n┌──────────────────────────────────────────┐")
    print("│  🎮 操作说明                             │")
    print("│  拖动 Trackbar 调整 HSV 上下界           │")
    print("│  Color slider 切换要调试的颜色            │")
    print("│  按 's' 打印当前全部颜色阈值             │")
    print("│  按 'r' 开/关 ROI 框显示                 │")
    print("│  按 'q' 退出                             │")
    print("└──────────────────────────────────────────┘\n")

    prev_color = 0
    show_roi = True

    while True:
        # ── 获取一帧 ──
        if use_hik:
            frame = read_frame(cam)
        else:
            ret, frame = cap.read()
            if not ret:
                break

        if frame is None:
            continue

        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # ── 检测颜色切换 ──
        cur_color_idx = cv2.getTrackbarPos(
            'Color 0红 1绿 2蓝 3浅蓝 4黑 5黄', TB_WIN)
        if cur_color_idx != prev_color:
            prev_color = cur_color_idx
            cv2.destroyWindow(TB_WIN)
            cv2.namedWindow(TB_WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(TB_WIN, 540, 400)
            cv2.createTrackbar('Color 0红 1绿 2蓝 3浅蓝 4黑 5黄',
                               TB_WIN, cur_color_idx, 5, nothing)
            make_trackbars(TB_WIN, COLOR_NAMES[cur_color_idx])

        color_name = COLOR_NAMES[cur_color_idx]

        # ── 读取 trackbar ──
        read_trackbars(TB_WIN, color_name)

        # ── 全图检测 ──
        roi = frame
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = apply_mask(hsv_roi, color_name)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        masked = cv2.bitwise_and(roi, roi, mask=mask)

        # ── 显示 ──
        disp = frame.copy()
        cv2.putText(disp, COLOR_LABELS[cur_color_idx], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Original", disp)

        cv2.imshow("Mask", mask)
        cv2.imshow("Masked Result", masked)

        # ── 按键 ──
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            dump_config()
        elif key == ord('r'):
            show_roi = not show_roi

    # ── 清理 ──
    if use_hik:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
    else:
        cap.release()
    cv2.destroyAllWindows()
    print("👋 已退出")


if __name__ == "__main__":
    main()
