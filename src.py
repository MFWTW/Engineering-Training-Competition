import cv2
import numpy as np
import threading
import time
import queue
from collections import deque
from pathlib import Path
from common_camera import (
    open_camera,
    QR_CAMERA_SOURCE,
    DETECTION_CAMERA_SOURCE,
    DETECTION_FRAME_WIDTH,
    DETECTION_FRAME_HEIGHT,
)
from preprocessing import *
import scan_QRcode_andlist
from felling_color import (
    CONFIG, CODE_TO_KEY,
    block_preprocessing, get_detector, reset_detector,
)
from gimbal import SerialComm, VisionToGimbal
from kalman_tracker import KALMAN_CFG, KalmanBlockTracker
from one_euro_filter import OneEuroTracker2D
from intercept_planner import InterceptPlanner
import transformer
from placement import PlacementRecognizer

# ==================== 可调参数统一入口 ====================
# 所有可调变量均从 config.yaml 读取（对应分段见文件内注释），
# 修改配置后重启程序生效。CONFIG 由 felling_color.load_config() 加载。
def _cfg(*keys, default=None):
    """按路径读取 CONFIG 中的值，路径不存在时返回 default。"""
    node = CONFIG
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


# 位置滤波算法选择（config.yaml → filter.type）：
#   none     = 不滤波，直接用当前帧检测中心（速度/加速度视为 0）
#   kalman   = 6 维卡尔曼（位置+速度+加速度）
#   one_euro = 一欧元低通（平滑位置 + 速度估计，慢速去抖、快速跟手）
# 未配置 filter.type 时，按旧的 kalman.enabled 决定默认行为，保证兼容。
FILTER_TYPE = str(_cfg("filter", "type", default=None) or (
    "kalman" if KALMAN_CFG.get("enabled", True) else "none"
)).strip().lower()
KALMAN_ENABLED = FILTER_TYPE == "kalman"

# 一欧元低通参数（config.yaml → one_euro）
ONE_EURO_MIN_CUTOFF = float(_cfg("one_euro", "min_cutoff", default=1.2))
ONE_EURO_BETA = float(_cfg("one_euro", "beta", default=0.7))
ONE_EURO_D_CUTOFF = float(_cfg("one_euro", "d_cutoff", default=1.0))
ONE_EURO_VELOCITY_WINDOW = int(_cfg("one_euro", "velocity_window", default=12))
ONE_EURO_DT_MIN = float(_cfg("one_euro", "dt_min_s", default=0.005))
ONE_EURO_DT_MAX = float(_cfg("one_euro", "dt_max_s", default=0.2))

# ==================== 目标切换与对准（config.yaml → control） ====================
# 下位机回传 finish_capture == 1 后自动切换下一个目标（主要方式）；
# "manual" 模式下仍可按 n/空格 手动切换；
# 非 manual 模式在 auto_switch_timeout 秒内未收到 finish_capture 时超时兜底切换。
CONTROL_MODE = str(_cfg("control", "mode", default="manual"))
AUTO_SWITCH_TIMEOUT = float(_cfg("control", "auto_switch_timeout", default=10.0))
# true=扫描二维码后跳过抓取，直接前往放置区（用于物块已就位、单独调试放置）
SKIP_GRAB = bool(_cfg("control", "skip_grab", default=False))
# true=扫码后只调试抓取：抓完当前轮即退出，跳过放置阶段
GRAB_ONLY = bool(_cfg("control", "grab_only", default=False))
# 抓取/放置阶段各自独立的 x 轴（左右）对准容差（px）：
# |目标x - 图像中心x| ≤ 对应容差才请求抓取/放置。
# 新键 grab_center_tolerance_px / place_center_tolerance_px 优先；
# 未配置时兼容旧的 center_tolerance_px（两个阶段都使用该值）。
_legacy_center_tol = _cfg("control", "center_tolerance_px", default=None)
_grab_center_tol = _cfg("control", "grab_center_tolerance_px", default=None)
_place_center_tol = _cfg("control", "place_center_tolerance_px", default=None)
GRAB_CENTER_TOLERANCE_PX = float(
    _grab_center_tol if _grab_center_tol is not None
    else (_legacy_center_tol if _legacy_center_tol is not None else 20)
)
PLACE_CENTER_TOLERANCE_PX = float(
    _place_center_tol if _place_center_tol is not None
    else (_legacy_center_tol if _legacy_center_tol is not None else 5)
)

# ==================== 指令发送节流/死区/心跳（config.yaml → tracking） ====================
# 等待抓取时，若下位机未回传 capture_ack=1，每隔该秒数重发一次 capture=1
CAPTURE_RESEND_INTERVAL = float(_cfg("tracking", "capture_resend_interval", default=1.0))
# 普通跟踪/对准指令（capture=0）的最小发送间隔（秒）；
# capture=1、阶段切换、区域移动、重发等事件包不受此限制，仍立即发送
TRACKING_SEND_INTERVAL = float(_cfg("tracking", "send_interval", default=0.5))
# 底盘比例增益：目标偏移先乘以该系数再下发，越靠近移动量越小，
# 避免 1~2cm 附近每次修正都刚好过冲、在 0 上下来回摆。
CHASSIS_P_GAIN = float(_cfg("tracking", "chassis_p_gain", default=0.5))
# 底盘指令变化死区（mm）：目标偏移相对上次已发送值变化小于该值时不重发，
# 避免下位机增量执行过程中被 100→90 这类微小变化反复打断。
CHASSIS_SEND_DEADBAND_MM = float(_cfg("tracking", "chassis_send_deadband_mm", default=1.0))
# 夹爪指令死区（mm）
GRIPPER_DEADBAND_MM = float(_cfg("tracking", "gripper_deadband_mm", default=5))
# 夹爪比例增益：夹爪指令本质是“上次指令 + 本次测距”的增量式控制，
# 直接全量累加会因执行滞后/测量抖动而积分冲顶、来回震荡；
# 该系数只把测距差值按比例修正（0.5=每帧修正一半），配合限幅后稳定收敛。
GRIPPER_P_GAIN = float(_cfg("tracking", "gripper_gain", default=0.5))
# 夹爪指令每个发送周期的变化量上限（mm，按 send_interval 标定），
# 防止 0→84mm 这种一次顶满的积分冲程。
GRIPPER_RAMP_STEP_MM = float(_cfg("tracking", "gripper_ramp_step_mm", default=5.0))
# 平滑跟踪：底盘指令每个发送周期的变化量上限（mm，按 send_interval 标定）。
# 让指令从小步连续爬升/衰减，而不是 0→30mm→0 这样跳变，避免“动一下停一下”。
CHASSIS_RAMP_STEP_MM = float(_cfg("tracking", "chassis_ramp_step_mm", default=4.0))
# 普通跟踪包的心跳间隔（秒）：即使指令变化很小，也至少按该间隔重发一次；
# 应大于 TRACKING_SEND_INTERVAL，否则死区不生效；设 None 禁用。
_send_heartbeat = _cfg("tracking", "send_heartbeat", default=5.0)
CHASSIS_SEND_HEARTBEAT = None if _send_heartbeat is None else float(_send_heartbeat)

# ==================== 显示窗口（config.yaml → display） ====================
# 宽度或高度超过时按同一比例缩小显示，避免画面超出屏幕；只影响显示，
# 不影响检测分辨率与坐标换算
DISPLAY_MAX_WIDTH = float(_cfg("display", "max_width", default=800))
DISPLAY_MAX_HEIGHT = float(_cfg("display", "max_height", default=540))
# 画面左下角叠加显示串口收发信息（只显示英文/数字，避免 OpenCV 中文乱码）
SERIAL_OVERLAY_ENABLED = bool(_cfg("display", "serial_overlay", "enabled", default=True))
SERIAL_OVERLAY_MAX = int(_cfg("display", "serial_overlay", "max_lines", default=4))

# 画面显示用英文颜色名（OpenCV 默认字体不支持中文，中文会显示乱码）
COLOR_LABEL_EN = {
    "red": "RED",
    "green": "GREEN",
    "blue": "BLUE",
    "light_blue": "LIGHT_BLUE",
    "black": "BLACK",
    "yellow": "YELLOW",
}

# 画圆/标签用 BGR 颜色（OpenCV 为 BGR 顺序），按颜色名对应 config.yaml 的 code
COLOR_BGR = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "light_blue": (255, 255, 0),
    "black": (0, 0, 0),
    "yellow": (0, 255, 255),
}

# ==================== 串口协议动作码（config.yaml → protocol） ====================
# action 字段：0=启动/空闲，1=抓取模式，2=放置模式
# （与下位机协议约定，一般不要改）
IDLE_ACTION = int(_cfg("protocol", "idle_action", default=0))
GRAB_ACTION = int(_cfg("protocol", "grab_action", default=1))
PLACE_ACTION = int(_cfg("protocol", "place_action", default=2))

# ==================== 指令保护（config.yaml → safety） ====================
# 底盘/夹爪指令合理范围（mm），防止坐标换算异常时把离谱值发给下位机
MAX_CHASSIS_CMD_MM = float(_cfg("safety", "max_chassis_cmd_mm", default=2000))
MAX_GRIPPER_MM = float(_cfg("safety", "max_gripper_mm", default=400))
# 普通跟踪包单次允许下发的底盘移动量上限（mm）。限制为小步后，
# 车会逐次逼近而不是一次发全量偏移大幅来回甩。
MAX_CHASSIS_STEP_MM = float(_cfg("safety", "max_chassis_step_mm", default=50))

# ==================== 放置阶段夹爪策略（config.yaml → placement） ====================
# gripper_fixed: min=固定最短(0mm伸长)只调底盘；max=固定最长(84mm伸长)只调底盘；
#                dynamic=与抓取一样动态调夹爪；旧配置 gripper_fixed_max 仍兼容
# gripper_fixed_mm: 自定义固定伸长量（mm），非空时优先生效，例如 40=固定伸长40mm只调底盘
PLACE_GRIPPER_FIXED = str(
    _cfg("placement", "gripper_fixed", default=None) or (
        "max" if bool(_cfg("placement", "gripper_fixed_max", default=True)) else "dynamic"
    )
).strip().lower()
PLACE_GRIPPER_FIXED_MAX = PLACE_GRIPPER_FIXED == "max"
PLACE_GRIPPER_FIXED_MIN = PLACE_GRIPPER_FIXED == "min"
_place_fixed_mm_cfg = _cfg("placement", "gripper_fixed_mm", default=None)
PLACE_GRIPPER_FIXED_CUSTOM = _place_fixed_mm_cfg is not None
if PLACE_GRIPPER_FIXED_CUSTOM:
    PLACE_GRIPPER_EXTEND_MM = int(round(float(_place_fixed_mm_cfg)))
    PLACE_GRIPPER_EXTEND_MM = min(
        max(PLACE_GRIPPER_EXTEND_MM, 0), transformer.MAX_GRIPPER_EXTEND_MM
    )
    PLACE_GRIPPER_EXTEND_CM = PLACE_GRIPPER_EXTEND_MM / 10.0
    PLACE_GRIPPER_MM = PLACE_GRIPPER_EXTEND_MM
else:
    PLACE_GRIPPER_EXTEND_CM = (
        0.0 if PLACE_GRIPPER_FIXED_MIN else transformer.MAX_GRIPPER_EXTEND_CM
    )
    PLACE_GRIPPER_MM = (
        0 if PLACE_GRIPPER_FIXED_MIN else transformer.MAX_GRIPPER_EXTEND_MM
    )

# ==================== 放置顺序记录（config.yaml → placement） ====================
# true=把每次实际放置的圆环数字按时间顺序追加写入文本文件，
# 同时每轮每遍完成时写一条“期望顺序 vs 实际顺序”汇总。
RECORD_PLACEMENT_ORDER = bool(_cfg("placement", "record_placement_order", default=True))
PLACEMENT_ORDER_LOG = str(_cfg(
    "placement", "placement_order_log",
    default=str(Path(__file__).resolve().parent / "placement_order.log"),
))

# 放置阶段位置稳定条件（config.yaml → placement）
PLACE_STABLE_THRESHOLD = int(_cfg("placement", "place_stable_threshold", default=30))
PLACE_STABLE_MAX_MOVE = float(_cfg("placement", "place_stable_max_pixel_move", default=20.0))

# ==================== 托盘阶段（config.yaml → placement） ====================
# 放置完成后进入托盘阶段：复用抓取阶段的视觉跟踪逻辑；
# 抓取顺序由 tray_phase_order 决定：actual=按实际放置顺序，reverse=倒序。
TRAY_PHASE_ENABLED = bool(_cfg("placement", "tray_phase_enabled", default=True))
TRAY_PHASE_ORDER = str(_cfg("placement", "tray_phase_order", default="reverse")).strip().lower()
TRAY_PHASE_ACTION = int(_cfg("placement", "tray_phase_action", default=GRAB_ACTION))
TRAY_PHASE_CAPTURE = bool(_cfg("placement", "tray_phase_capture", default=True))
TRAY_PHASE_SKIP_LAST_OF_ROUND = bool(
    _cfg("placement", "tray_phase_skip_last_of_round", default=True)
)
# 托盘阶段夹爪策略（config.yaml → placement.tray_gripper_fixed）：
# dynamic/null=动态调夹爪；min/max/数字mm=固定夹爪长度，只靠底盘对准。
# 相机装在夹爪上，机构响应慢时动态夹爪的“测量→指令”闭环会震荡，
# 固定夹爪后底盘有回传反馈能收敛，彻底绕开这个环。
_tray_gripper_fixed_cfg = _cfg("placement", "tray_gripper_fixed", default=None)
TRAY_GRIPPER_FIXED = (
    str(_tray_gripper_fixed_cfg).strip().lower()
    if _tray_gripper_fixed_cfg is not None else "dynamic"
)
TRAY_GRIPPER_FIXED_MIN = TRAY_GRIPPER_FIXED == "min"
TRAY_GRIPPER_FIXED_MAX = TRAY_GRIPPER_FIXED == "max"
_tray_fixed_mm_cfg = None
try:
    _tray_fixed_mm_cfg = float(_tray_gripper_fixed_cfg)
except (TypeError, ValueError):
    pass
TRAY_GRIPPER_FIXED_CUSTOM = (
    _tray_fixed_mm_cfg is not None
    and not TRAY_GRIPPER_FIXED_MIN
    and not TRAY_GRIPPER_FIXED_MAX
)
if TRAY_GRIPPER_FIXED_CUSTOM:
    TRAY_GRIPPER_EXTEND_MM = int(round(_tray_fixed_mm_cfg))
    TRAY_GRIPPER_EXTEND_MM = min(
        max(TRAY_GRIPPER_EXTEND_MM, 0), transformer.MAX_GRIPPER_EXTEND_MM
    )
    TRAY_GRIPPER_EXTEND_CM = TRAY_GRIPPER_EXTEND_MM / 10.0
    TRAY_GRIPPER_MM = TRAY_GRIPPER_EXTEND_MM
else:
    TRAY_GRIPPER_EXTEND_CM = (
        0.0 if TRAY_GRIPPER_FIXED_MIN else transformer.MAX_GRIPPER_EXTEND_CM
    )
    TRAY_GRIPPER_MM = (
        0 if TRAY_GRIPPER_FIXED_MIN else transformer.MAX_GRIPPER_EXTEND_MM
    )

# ==================== 日志打印节流（config.yaml → logging） ====================
# 指令打印节流：数值变化或超过该间隔才打印一次，避免每帧刷屏
COMMAND_PRINT_INTERVAL = float(_cfg("logging", "command_print_interval", default=0.5))
# “坐标无效 / 命令全0”警告打印最小间隔（秒）
WARN_INTERVAL_S = float(_cfg("logging", "warn_interval_s", default=1.0))

# ==================== 拦截规划器（config.yaml → planner） ====================
PLANNER_CAR_MAX_SPEED = float(_cfg("planner", "car_max_speed_px_per_s", default=200.0))
PLANNER_CAR_ACCEL = float(_cfg("planner", "car_accel_px_per_s2", default=100.0))
PLANNER_CAR_DECEL = float(_cfg("planner", "car_decel_px_per_s2", default=150.0))
PLANNER_TIME_RESOLUTION = float(_cfg("planner", "time_resolution_s", default=0.02))
PLANNER_MAX_PREDICT_TIME = float(_cfg("planner", "max_predict_time_s", default=5.0))
PLANNER_TOLERANCE = float(_cfg("planner", "tolerance_s", default=0.1))

# ==================== ROI 配置 ====================
# ROI 区域在 config.yaml 的 detection.detection_area 中配置（[x, y, w, h]，None=全图），
# 由 felling_color.BlockDetector 自动读取，主循环中会做边界修正并绘制。

# ==================== 全局变量 ====================
C_1 = None
C_2 = None

# 底盘回传数据（串口接收线程更新，主循环读取）
chassis_x: int = 0
chassis_y: int = 0
chassis_vx: int = 0
chassis_vy: int = 0
# 下位机回传单位：chassis_x/y 为 mm，chassis_vx/vy 为 mm/s；
# 拦截规划在图像像素空间进行，用该倍率把 mm→px、mm/s→px/s；
# 数值在 config.yaml 的 chassis.px_per_mm 中配置（按实际标定填写）
PX_PER_MM = float(CONFIG.get("chassis", {}).get("px_per_mm", 1.0))

# 共享串口对象（发送线程 + 主循环都需要用）
_serial_comm = None

# 串口收发信息显示缓冲（TX / RX 各保留最近 N 条，全部用英文显示）
_tx_log = deque(maxlen=SERIAL_OVERLAY_MAX)
_rx_log = deque(maxlen=SERIAL_OVERLAY_MAX)
_last_rx_key = None
_tx_offline_logged = False


def log_serial_tx(text):
    """记录一条串口发送信息（供画面叠加显示）。"""
    _tx_log.append(f"{time.strftime('%H:%M:%S')} TX {text}")


def log_serial_rx(text):
    """记录一条串口接收信息（供画面叠加显示）。"""
    _rx_log.append(f"{time.strftime('%H:%M:%S')} RX {text}")


def log_serial_rx_if_changed(data):
    """底盘回传值有变化时才记录，避免每帧都刷同一行。"""
    global _last_rx_key
    key = (
        data.chassis_x, data.chassis_y,
        data.chassis_vx, data.chassis_vy,
        data.capture_ack, data.finish_capture, data.arrived,
    )
    if key == _last_rx_key:
        return
    _last_rx_key = key
    log_serial_rx(
        f"x={data.chassis_x} y={data.chassis_y} "
        f"vx={data.chassis_vx} vy={data.chassis_vy} "
        f"ack={data.capture_ack} fin={data.finish_capture} arr={data.arrived}"
    )


def draw_serial_overlay(frame):
    """在画面左下角叠加显示最近的串口 TX / RX 信息（英文，避免乱码）。"""
    if not SERIAL_OVERLAY_ENABLED:
        return
    h, w = frame.shape[:2]
    tx_lines = list(_tx_log)
    rx_lines = list(_rx_log)
    if not tx_lines and not rx_lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1
    line_h = 15
    pad = 6
    total = len(tx_lines) + len(rx_lines)
    box_h = total * line_h + pad * 2
    x0 = 6
    x1 = w - 6
    y1 = h - 6
    y0 = y1 - box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    frame[...] = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    y = y0 + pad + line_h - 4
    for line in tx_lines:
        cv2.putText(frame, line, (x0 + 4, y), font, scale,
                    (0, 255, 255), thickness, cv2.LINE_AA)
        y += line_h
    for line in rx_lines:
        cv2.putText(frame, line, (x0 + 4, y), font, scale,
                    (0, 255, 0), thickness, cv2.LINE_AA)
        y += line_h


def enqueue_latest(data_pack, vg):
    """只保留最新一包：队列满时丢弃旧包再放入新包，避免断线积压后重发过期坐标。"""
    try:
        if data_pack.full():
            try:
                data_pack.get_nowait()
            except queue.Empty:
                pass
        data_pack.put_nowait(vg)
    except queue.Full:
        pass


def clamp_roi(roi, frame_shape):
    """修正 ROI 在图像范围内，避免超出边界。"""
    if roi is None:
        return None
    x, y, w, h = roi
    H, W = frame_shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return [x, y, w, h]


def show_detection(frame):
    """缩放后显示检测画面，防止窗口过大；仅显示用，不修改原始帧。"""
    h, w = frame.shape[:2]
    scale = min(
        1.0,
        DISPLAY_MAX_WIDTH / float(w),
        DISPLAY_MAX_HEIGHT / float(h),
    )
    if scale < 1.0:
        frame = cv2.resize(
            frame, (int(round(w * scale)), int(round(h * scale)))
        )
    draw_serial_overlay(frame)
    cv2.imshow("detection", frame)


_last_command_print = {"t": 0.0, "sig": None}
_last_invalid_warn = {"t": 0.0}
_last_zero_warn = {"t": 0.0}


def sanitize_protocol_mm(cmd, last_valid):
    """底盘/夹爪指令越界时返回上一帧有效指令，避免误发巨大移动量。"""
    chassis_x_mm, chassis_y_mm, gripper_mm = cmd
    if (
        abs(chassis_x_mm) > MAX_CHASSIS_CMD_MM
        or abs(chassis_y_mm) > MAX_CHASSIS_CMD_MM
        or not (0 <= gripper_mm <= MAX_GRIPPER_MM)
    ):
        print(
            f"[保护] 指令越界 ({chassis_x_mm:+d},{chassis_y_mm:+d})mm "
            f"夹爪={gripper_mm}mm，已沿用上一帧有效指令 {last_valid}"
        )
        return last_valid
    return (chassis_x_mm, chassis_y_mm, gripper_mm)


def warn_invalid_coord(tag, u, v, image_w, image_h):
    """坐标换算失败时节流打印，便于定位为什么一直发 0。"""
    now = time.time()
    if now - _last_invalid_warn["t"] < WARN_INTERVAL_S:
        return
    _last_invalid_warn["t"] = now
    print(
        f"[坐标无效] {tag}: 目标({u},{v}) @ {image_w}x{image_h}，"
        "无法换算成相机坐标，沿用上一帧有效指令"
    )


def warn_zero_command(tag, camera_coord, u, v):
    """像素坐标有效但转换结果全为 0 时节流打印，便于定位标定问题。"""
    now = time.time()
    if now - _last_zero_warn["t"] < WARN_INTERVAL_S:
        return
    _last_zero_warn["t"] = now
    print(
        f"[命令全0] {tag}: 目标({u},{v}) 相机坐标={camera_coord}，"
        "转换后底盘/夹爪全为 0，请检查相机标定与安装角度"
    )


def log_command(tag, target, action, capture,
                chassis_x_mm, chassis_y_mm, gripper_mm,
                fb_x=0, fb_y=0, fb_vx=0, fb_vy=0,
                camera_coord=None, world_coord=None):
    """打印当前下发给下位机的底盘/夹爪指令，以及下位机回传的底盘数据。"""
    now = time.time()
    sig = (
        tag, target, action, capture,
        chassis_x_mm, chassis_y_mm, gripper_mm,
        camera_coord, world_coord,
    )
    if (
        not capture
        and sig == _last_command_print["sig"]
        and now - _last_command_print["t"] < COMMAND_PRINT_INTERVAL
    ):
        return

    _last_command_print["t"] = now
    _last_command_print["sig"] = sig
    print(
        f"[{tag}] target={target} action={action} capture={int(capture)} "
        f"底盘移动量=({chassis_x_mm:+d},{chassis_y_mm:+d})mm "
        f"夹爪伸长量={gripper_mm}mm | "
        f"下位机回传=({fb_x},{fb_y})mm v=({fb_vx},{fb_vy})mm/s"
    )
    if camera_coord is not None:
        print(
            f"  相机坐标={camera_coord}cm, "
            f"车中心坐标={world_coord}cm"
        )


def append_placement_record(text):
    """把一行放置记录追加到 PLACEMENT_ORDER_LOG（失败只告警，不影响主流程）。"""
    if not RECORD_PLACEMENT_ORDER:
        return
    try:
        with open(PLACEMENT_ORDER_LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")
    except OSError as exc:
        print(f"[放置记录] 写入失败: {PLACEMENT_ORDER_LOG}: {exc}")


def log_tray_order(round_no, cycle_no, placed_order, order_mode="actual"):
    """按配置的顺序模式生成“托盘阶段”抓取顺序并记录（数字即托盘号）。"""
    if not placed_order:
        return
    if order_mode in ("reverse", "reversed", "倒序"):
        seq = list(reversed(placed_order))
        label = "倒序"
    else:
        seq = list(placed_order)
        label = "实际顺序"
    line = (
        f"第{round_no}轮 第{cycle_no}遍 托盘阶段：{label}抓取="
        f"{','.join(map(str, seq))}，对应托盘={','.join(map(str, seq))}"
    )
    print(f"[托盘顺序] {line}")
    append_placement_record(line)


# ==================== 发送线程 ====================
def Sending2Gimbal(data_pack, serial_comm):
    """后台常驻：阻塞等待队列数据，收到就打包发送"""
    global _tx_offline_logged
    print("[发送线程启动]，等待数据")
    last_capture = None

    while True:
        vg = data_pack.get()
        if vg is None:
            print("[发送线程] 收到停止信号，退出")
            break

        try:
            packed = vg.pack()
            # 跟踪包已由主循环按 TRACKING_SEND_INTERVAL 节流，只打印 capture 变化，避免刷屏
            if vg.capture_ != last_capture:
                print(f"  capture={vg.capture_} hex: {packed.hex(' ')}")
                last_capture = vg.capture_
        except Exception as e:
            print(f"  pack 失败: {e}")
            continue

        if serial_comm:
            try:
                serial_comm.send(vg)
                log_serial_tx(
                    f"t={vg.target_} a={vg.action_} c={vg.capture_} "
                    f"x={vg.chassis_x_mm} y={vg.chassis_y_mm} g={vg.gripper_mm}"
                )
            except Exception as e:
                print(f"  串口发送失败: {e}")
                log_serial_tx(f"SEND-ERR {e}")
        else:
            if not _tx_offline_logged:
                log_serial_tx("OFFLINE")
                _tx_offline_logged = True
            print("  [离线] 未发送")

    print("[发送线程] 已退出")


# ==================== 主程序 ====================
def main():
    global C_1, C_2, chassis_x, chassis_y, chassis_vx, chassis_vy, _serial_comm

    # 两台 USB 免驱摄像头：cap 用于二维码扫描，detection_cap 用于物块检测/放置
    cap = open_camera(QR_CAMERA_SOURCE)
    if cap is None:
        return

    detection_cap = open_camera(
        DETECTION_CAMERA_SOURCE,
        width=DETECTION_FRAME_WIDTH,
        height=DETECTION_FRAME_HEIGHT,
    )
    if detection_cap is None:
        print("未检测到物块检测 USB 摄像头，退出")
        if cap:
            cap.release()
        return

    q = queue.Queue(maxsize=1)

    # 创建串口 + 启动底盘接收线程
    try:
        _serial_comm = SerialComm()
        _serial_comm.start_chassis_recv()
        print("串口已打开，底盘接收线程已启动")
    except Exception as e:
        print(f"串口打开失败: {e}")
        _serial_comm = None

    sending_thread = threading.Thread(target=Sending2Gimbal, args=(q, _serial_comm), daemon=True)
    sending_thread.start()

    # 项目启动：先向下位机发送 action=0（启动/空闲）信号
    vg = VisionToGimbal(target=0, action=IDLE_ACTION)
    enqueue_latest(q, vg)
    print("已发送启动信号 action=0，等待识别二维码...")

    # ---- 状态 ----
    last_sessions = None
    detector = get_detector()
    detector.reset()
    # 检测/稳定性参数统一从 config.yaml 读取（felling_color.BlockDetector 默认值）

    # 卡尔曼滤波追踪器（替代神经网络预测），参数从 config.yaml 的 kalman 段读取
    kf = KalmanBlockTracker()

    # 一欧元低通追踪器（filter.type=one_euro 时使用），参数从 config.yaml 的 one_euro 段读取
    one_euro_tracker = OneEuroTracker2D(
        min_cutoff=ONE_EURO_MIN_CUTOFF,
        beta=ONE_EURO_BETA,
        d_cutoff=ONE_EURO_D_CUTOFF,
        velocity_window=ONE_EURO_VELOCITY_WINDOW,
        dt_min=ONE_EURO_DT_MIN,
        dt_max=ONE_EURO_DT_MAX,
    )

    # 拦截规划器（考虑小车动力学，博弈 T 求解）
    planner = InterceptPlanner(
        car_max_speed=PLANNER_CAR_MAX_SPEED,
        car_accel=PLANNER_CAR_ACCEL,
        car_decel=PLANNER_CAR_DECEL,
        time_resolution=PLANNER_TIME_RESOLUTION,
        max_predict_time=PLANNER_MAX_PREDICT_TIME,
        tolerance=PLANNER_TOLERANCE,
    )

    roi_clamped = False  # ROI 只修正一次

    detection_sent = False
    waiting_for_next = False    # 已请求抓取，等待下位机 finish_capture 或手动切换
    all_done = False            # 所有目标已完成，退出主循环
    rounds = []                 # [{"grab": [...], "place": [...]}, ...]
    current_round = 0
    round_cycles_done = 0       # 当前轮已完成几遍“抓取→放置”（0 开始，每轮 repeat 次）
    phase = IDLE_ACTION         # 当前动作：0=启动/空闲，1=抓取，2=放置
    phase_after_arrival = IDLE_ACTION
    waiting_for_arrive = False  # 正在等待下位机到达指定区域
    place_waiting_arrived = False  # 放置完一个槽位后，等待下位机到达下一位置（新的 arrived=1）
    prev_arrived = 0
    recognition_started = False  # 是否已打印当前阶段“开始识别”提示
    placed_digits = set()       # 本轮已放置的圆环数字
    placed_order = []           # 本轮实际放置的圆环数字（按放置先后顺序）
    last_placed_digit = None
    tray_phase_active = False   # 是否处于托盘阶段（按配置顺序抓取托盘上的物块）
    tray_targets = []           # 托盘阶段剩余待发送 target（按 tray_phase_order 配置生成）
    tray_plan = []              # 托盘阶段剩余 [(托盘号, 物块颜色), ...]，进入阶段时一次算好
    tray_pending_digit = None   # 托盘阶段当前正在抓取的 target（已到达托盘，等待抓取完成）
    place_stable_count = 0      # 放置阶段位置稳定计数（连续同圆环且位移小）
    place_last_center = None    # 放置阶段上一帧选中圆环中心
    place_last_digit = None     # 放置阶段上一帧选中圆环数字
    placement_recognizer = None
    last_place_dbg = 0.0          # 放置发送调试日志节流
    target_colors = []          # 当前轮抓取颜色序列（rounds[current_round]["grab"] 的别名）
    target_index = 0
    last_detection_time = None
    sent_time = None  # 记录发送时间
    capture_ack_received = False     # 下位机是否已确认收到抓取请求
    capture_last_sent_time = None    # 最近一次发送/重发 capture=1 的时间
    last_capture_vg = None           # 最近一次抓取请求包，用于重发
    last_tracking_send = 0.0         # 最近一次普通跟踪包（capture=0）发送时间
    last_sent_tracking_mm = None     # 最近一次普通跟踪包发送的 (x, y, gripper) mm
    last_smooth_cmd_mm = (0, 0, 0)   # 平滑后的底盘/夹爪指令（ramp 状态）
    last_smooth_time = time.time()   # 最近一次平滑更新时间
    last_kf_time = None              # 上一帧 KF 更新时间（用于按实际帧间隔更新 dt）
    # 最近一次有效的底盘/夹爪指令；pixel_to_camera 失效时沿用，避免误发 0
    last_valid_mm = (0, 0, 0)

    # 上一个检测到的物块信息（等待阶段重绘用）
    last_detected_center = None
    last_detected_radius = 0
    last_detected_color = (0, 255, 0)
    last_detected_label = ""

    # 下位机 finish_capture 上升沿检测（0→1 只触发一次）
    prev_finish_capture = 0

    def switch_to_next_target(reason="指令"):
        """切换到下一个目标：统一重置跟踪状态，避免各处分叉重复。"""
        nonlocal waiting_for_next, detection_sent, sent_time, last_detected_center, target_index
        nonlocal capture_ack_received, capture_last_sent_time, last_capture_vg
        nonlocal last_tracking_send, last_sent_tracking_mm
        nonlocal last_smooth_cmd_mm, last_smooth_time
        nonlocal last_valid_mm, last_kf_time
        waiting_for_next = False
        detector.reset()
        kf.reset()
        one_euro_tracker.reset()
        detection_sent = False
        sent_time = None
        capture_ack_received = False
        capture_last_sent_time = None
        last_capture_vg = None
        last_tracking_send = 0.0
        last_sent_tracking_mm = None
        last_smooth_cmd_mm = (0, 0, 0)
        last_smooth_time = time.time()
        last_kf_time = None
        last_detected_center = None
        last_valid_mm = (0, 0, 0)
        if target_index + 1 < len(target_colors):
            target_index += 1
            print(f"[{reason}] 切换到目标 {target_index + 1}/{len(target_colors)}")
        else:
            print(f"[{reason}] 已是本轮最后一个抓取目标")

    def reset_action_state():
        """切换到新阶段/新位置时重置抓取或放置的发送状态。"""
        nonlocal detection_sent, waiting_for_next, sent_time
        nonlocal capture_ack_received, capture_last_sent_time, last_capture_vg
        nonlocal last_tracking_send, last_sent_tracking_mm
        nonlocal last_smooth_cmd_mm, last_smooth_time
        nonlocal last_valid_mm, last_detected_center, last_detection_time, last_kf_time
        nonlocal recognition_started
        nonlocal place_stable_count, place_last_center, place_last_digit
        detection_sent = False
        waiting_for_next = False
        sent_time = None
        capture_ack_received = False
        capture_last_sent_time = None
        last_capture_vg = None
        last_tracking_send = 0.0
        last_sent_tracking_mm = None
        last_smooth_cmd_mm = (0, 0, 0)
        last_smooth_time = time.time()
        last_kf_time = None
        last_valid_mm = (0, 0, 0)
        last_detected_center = None
        last_detection_time = None
        recognition_started = False
        place_stable_count = 0
        place_last_center = None
        place_last_digit = None
        detector.reset()
        kf.reset()
        one_euro_tracker.reset()

    def advance_after_placement_cycle():
        """一轮（一遍）放置完成后：未重复完则回抓取区，重复完则进下一轮或结束。

        返回 "done" 表示所有轮次完成，否则已发出前往抓取区的指令。
        """
        nonlocal round_cycles_done, current_round, target_colors, target_index
        nonlocal placed_digits, placed_order, place_waiting_arrived
        nonlocal waiting_for_arrive, phase_after_arrival, prev_arrived
        nonlocal placed_block_slots

        repeat = rounds[current_round].get("repeat", 1)
        if round_cycles_done + 1 < repeat:
            round_cycles_done += 1
            append_placement_record(
                f"第{current_round + 1}轮 第{round_cycles_done}遍 完成 "
                f"期望顺序={','.join(rounds[current_round]['place'])} "
                f"实际顺序={','.join(map(str, placed_order))}"
            )
            target_colors = rounds[current_round]["grab"]
            target_index = 0
            placed_digits = set()
            placed_order = []
            # 新一遍回抓取区时物块重新立放（14.3cm）：
            # 清掉上一遍放置留下的躺倒标记，让第二次抓取与第一次用同一套跟踪策略；
            # 托盘阶段的躺倒高度由 tray_phase_active 单独判断。
            placed_block_slots.clear()
            place_waiting_arrived = False
            print(f"第 {current_round + 1} 轮第 {round_cycles_done} 次"
                  f"放置完成，回抓取区进行第 {round_cycles_done + 1} 次抓取")
            vg = VisionToGimbal(target=0, action=GRAB_ACTION)
            enqueue_latest(q, vg)
            waiting_for_arrive = True
            phase_after_arrival = GRAB_ACTION
            prev_arrived = 0
            reset_action_state()
            return "grab"

        if current_round + 1 < len(rounds):
            append_placement_record(
                f"第{current_round + 1}轮 第{round_cycles_done + 1}遍 完成 "
                f"期望顺序={','.join(rounds[current_round]['place'])} "
                f"实际顺序={','.join(map(str, placed_order))}"
            )
            current_round += 1
            round_cycles_done = 0
            target_colors = rounds[current_round]["grab"]
            target_index = 0
            placed_digits = set()
            placed_order = []
            placed_block_slots.clear()
            place_waiting_arrived = False
            print(f"第 {current_round} 轮放置完成（共 {repeat} 次），"
                  f"前往第 {current_round + 1} 轮抓取区")
            vg = VisionToGimbal(target=0, action=GRAB_ACTION)
            enqueue_latest(q, vg)
            waiting_for_arrive = True
            phase_after_arrival = GRAB_ACTION
            prev_arrived = 0
            reset_action_state()
            return "grab"

        append_placement_record(
            f"第{current_round + 1}轮 第{round_cycles_done + 1}遍 完成 "
            f"期望顺序={','.join(rounds[current_round]['place'])} "
            f"实际顺序={','.join(map(str, placed_order))} 全部任务完成"
        )
        print("所有轮次完成，退出")
        return "done"

    def smooth_tracking_cmd(cmd_mm, now):
        """
        把目标移动量平滑成连续小步：
        比例缩放 → 单包限幅 → 按 ramp 限制指令每帧变化量；
        夹爪同样先乘比例增益再按 ramp 限幅，避免增量式指令积分冲顶。
        底盘指令不会从 0 突然跳到几十 mm，也不会在目标附近骤停。
        """
        nonlocal last_smooth_cmd_mm, last_smooth_time
        x, y, gripper = cmd_mm
        dx = int(max(-MAX_CHASSIS_STEP_MM,
                     min(MAX_CHASSIS_STEP_MM, round(x * CHASSIS_P_GAIN))))
        dy = int(max(-MAX_CHASSIS_STEP_MM,
                     min(MAX_CHASSIS_STEP_MM, round(y * CHASSIS_P_GAIN))))

        # ramp 按发送周期标定，这里按实际帧间隔缩放
        dt = min(max(now - last_smooth_time, 0.0), 0.5)
        if TRACKING_SEND_INTERVAL > 0:
            ramp = CHASSIS_RAMP_STEP_MM * (dt / TRACKING_SEND_INTERVAL)
            gripper_ramp = GRIPPER_RAMP_STEP_MM * (dt / TRACKING_SEND_INTERVAL)
        else:
            ramp = CHASSIS_RAMP_STEP_MM
            gripper_ramp = GRIPPER_RAMP_STEP_MM
        ramp = max(ramp, 0.5)
        gripper_ramp = max(gripper_ramp, 0.5)

        lx, ly, lz = last_smooth_cmd_mm
        nx = lx + max(-ramp, min(ramp, dx - lx))
        ny = ly + max(-ramp, min(ramp, dy - ly))
        # 夹爪：先比例增益再限幅（只允许向目标方向缓慢修正）
        gz_delta = int(round((gripper - lz) * GRIPPER_P_GAIN))
        gz = lz + max(-gripper_ramp, min(gripper_ramp, gz_delta))
        gz = int(round(min(max(gz, 0), MAX_GRIPPER_MM)))

        last_smooth_cmd_mm = (int(round(nx)), int(round(ny)), gz)
        last_smooth_time = now
        return last_smooth_cmd_mm

    def tracking_send_allowed(capture, cmd_mm):
        """
        普通 capture=0 包按“最小间隔 + 变化死区 + 心跳”节流；
        capture=1 立即发送。
        """
        nonlocal last_tracking_send, last_sent_tracking_mm
        now = time.time()
        if capture:
            last_tracking_send = now
            last_sent_tracking_mm = cmd_mm
            return True

        if now - last_tracking_send < TRACKING_SEND_INTERVAL:
            return False

        if last_sent_tracking_mm is None:
            last_tracking_send = now
            last_sent_tracking_mm = cmd_mm
            return True

        # 底盘变化量（目标偏移变化超过发送死区才需要再修正）
        chassis_changed = (
            abs(cmd_mm[0] - last_sent_tracking_mm[0]) >= CHASSIS_SEND_DEADBAND_MM
            or abs(cmd_mm[1] - last_sent_tracking_mm[1]) >= CHASSIS_SEND_DEADBAND_MM
        )
        gripper_changed = (
            abs(cmd_mm[2] - last_sent_tracking_mm[2]) >= GRIPPER_DEADBAND_MM
        )
        # 心跳：变化很小也定期重发，防止下位机断链
        heartbeat = (
            CHASSIS_SEND_HEARTBEAT is not None
            and now - last_tracking_send >= CHASSIS_SEND_HEARTBEAT
        )
        if not chassis_changed and not gripper_changed and not heartbeat:
            return False

        last_tracking_send = now
        last_sent_tracking_mm = cmd_mm
        return True

    try:
        while True:
            if cap is not None:
                ret, frame = cap.read()
            else:
                ret, frame = False, None

            if not ret and not scan_QRcode_andlist.session:
                print("无法读取画面")
                break

            # ============ QR 扫描阶段 ============
            if len(scan_QRcode_andlist.session) == 0:
                Ostu_image = ostu_threshold(frame)
                scan_QRcode_andlist.scan_qrcode(Ostu_image, frame)
                sessions = scan_QRcode_andlist.session

                if sessions and sessions != last_sessions:
                    last_sessions = sessions
                    print(f"QR识别结果: {sessions}")

                    # 4 组数解析为两轮任务：[抓取序列, 放置序列] × 2
                    groups = scan_QRcode_andlist.groups
                    rounds = []
                    for i in range(0, len(groups) - 1, 2):
                        grab = [c for c in groups[i] if c.isdigit()]
                        place = [c for c in groups[i + 1] if c.isdigit()]
                        if grab and place:
                            # 每轮“抓取→放置”要重复执行 2 遍（第 1 轮=前两组×2，
                            # 第 2 轮=后两组×2），显示仍只显示二维码扫到的 4 组
                            rounds.append({
                                "grab": grab,
                                "place": place,
                                "repeat": 2,
                            })
                    if not rounds:
                        print("[QR] 无法解析抓取/放置序列，退出")
                        break

                    current_round = 0
                    round_cycles_done = 0
                    target_colors = rounds[0]["grab"]
                    target_index = 0
                    phase = GRAB_ACTION
                    phase_after_arrival = GRAB_ACTION
                    waiting_for_arrive = False
                    place_waiting_arrived = False
                    placed_digits = set()
                    placed_order = []
                    last_placed_digit = None
                    tray_phase_active = False
                    tray_targets = []
                    tray_plan = []
                    tray_pending_digit = None
                    place_stable_count = 0
                    place_last_center = None
                    place_last_digit = None
                    grabbed_slots = set()
                    placed_block_slots = set()   # 已放置过的槽位：再次夹取时物块已躺倒
                    last_grabbed_slot = None
                    slot_of_color = {c: i + 1 for i, c in enumerate(target_colors)}
                    slot_of_place_digit = {}
                    placement_recognizer = None

                    # 协议 B：QR 内容只在上位机使用，下位机只收一个 task 开始信号
                    if SKIP_GRAB and not GRAB_ONLY:
                        # 物块已就位：跳过抓取，直接前往放置区单独调放置
                        phase = PLACE_ACTION
                        vg = VisionToGimbal(target=0, action=PLACE_ACTION)
                        enqueue_latest(q, vg)
                        waiting_for_arrive = True
                        phase_after_arrival = PLACE_ACTION
                        prev_arrived = 0
                        print(f"已跳过抓取，放置编号: {rounds[0]['place']}（每轮重复 2 次）")
                        print("已发送放置区移动指令，等待下位机 arrived=1 后开始识别放置位置")
                    else:
                        vg = VisionToGimbal(target=0, action=GRAB_ACTION)
                        enqueue_latest(q, vg)
                        # 等下位机到达抓取区回 arrived=1 后再开始识别物料
                        waiting_for_arrive = True
                        phase_after_arrival = GRAB_ACTION
                        prev_arrived = 0

                        print(f"第 1 轮抓取顺序: {target_colors}，"
                              f"放置编号: {rounds[0]['place']}（每轮重复 2 次）")
                        print("已发送抓取区移动指令，等待下位机 arrived=1 后开始识别物料")

                if last_sessions is not None:
                    if cap:
                        cap.release()
                        cap = None
                    cv2.destroyAllWindows()
                    print("识别到QR，关闭二维码USB摄像头")
                    detector.reset()
                    kf.reset()
                    one_euro_tracker.reset()
                    detection_sent = False
                    last_detection_time = None
                    last_detected_center = None

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # ============ 物块检测阶段 ============
            else:
                # ---- 时间戳同步：先记底盘最新时间，再抓帧 ----
                chassis_data = _serial_comm.get_chassis_data() if _serial_comm else None
                if chassis_data:
                    chassis_x = chassis_data.chassis_x
                    chassis_y = chassis_data.chassis_y
                    chassis_vx = chassis_data.chassis_vx
                    chassis_vy = chassis_data.chassis_vy
                    chassis_t = chassis_data.timestamp  # 底盘数据时间戳

                # 抓取图像帧，记录时间
                ret, detection_frame = detection_cap.read()
                if not ret:
                    print("无法读取物块检测USB摄像头画面")
                    break
                frame_t = time.time()
                frame = detection_frame

                # 若图像帧落后于底盘数据，说明底盘更新了，读取最新值
                chassis_data2 = _serial_comm.get_chassis_data() if _serial_comm else None
                if chassis_data2 is not None:
                    log_serial_rx_if_changed(chassis_data2)
                if chassis_data2 and chassis_data2.timestamp > frame_t:
                    # 用离 frame_t 更近的底盘数据
                    if chassis_data is None or (
                        abs(chassis_data2.timestamp - frame_t) < abs(chassis_t - frame_t)
                    ):
                        chassis_x = chassis_data2.chassis_x
                        chassis_y = chassis_data2.chassis_y
                        chassis_vx = chassis_data2.chassis_vx
                        chassis_vy = chassis_data2.chassis_vy

                # 摄像头装在夹爪/云台上，会随夹爪一起伸缩；
                # 换算“物块相对车中心”时需加上当前夹爪伸长量（取最近一次已下发指令）
                current_gripper_cm = (
                    (last_sent_tracking_mm[2] / 10.0)
                    if last_sent_tracking_mm is not None
                    else 0.0
                )

                # ---- 动作确认/完成信号 + 到达信号 ----
                finish_rising = False
                arrived_rising = False
                if chassis_data2:
                    if chassis_data2.arrived == 1 and not prev_arrived:
                        arrived_rising = True
                        print("[下位机] 收到 arrived=1（上升沿）")
                    prev_arrived = 1 if chassis_data2.arrived else 0

                    if waiting_for_next:
                        if chassis_data2.capture_ack and not capture_ack_received:
                            print("[下位机] 已确认动作请求，等待 finish_capture")
                        capture_ack_received = capture_ack_received or bool(chassis_data2.capture_ack)
                        # finish_capture 0→1 上升沿只触发一次
                        if chassis_data2.finish_capture == 1 and not prev_finish_capture:
                            finish_rising = True
                        prev_finish_capture = 1 if chassis_data2.finish_capture else 0
                    else:
                        # 未等待时同步当前电平，避免旧电平在下次造成误触发
                        prev_finish_capture = 1 if chassis_data2.finish_capture else 0

                # 下位机已到达指定区域（抓取区/放置区）
                if waiting_for_arrive and arrived_rising:
                    phase = phase_after_arrival
                    waiting_for_arrive = False
                    reset_action_state()
                    target_index = 0
                    if phase == PLACE_ACTION:
                        place_waiting_arrived = False
                        placed_digits = set()
                        placed_order = []
                        last_placed_digit = None
                        slot_of_place_digit = {
                            int(d): i + 1
                            for i, d in enumerate(rounds[current_round]["place"])
                        }
                        if placement_recognizer is None:
                            placement_recognizer = PlacementRecognizer()
                        print("已到达放置区，开始识别物料（圆环数字）")
                    else:
                        grabbed_slots = set()
                        last_grabbed_slot = None
                        if tray_phase_active and tray_pending_digit is not None:
                            print(f"[托盘] 已到达托盘 {tray_pending_digit}，开始识别物块")
                        else:
                            slot_of_color = {c: i + 1 for i, c in enumerate(target_colors)}
                            print(f"已到达抓取区，开始识别物料"
                                  f"（第 {current_round + 1} 轮第 {round_cycles_done + 1} 次"
                                  f"抓取顺序: {target_colors}）")
                    continue

                # 动作完成（抓取完成 / 放置完成）
                if finish_rising:
                    # 动作完成：立即补发 capture=0，通知下位机本次抓取/放置已结束
                    if tray_phase_active and tray_pending_digit is not None:
                        release_target = tray_pending_digit
                        release_action = TRAY_PHASE_ACTION
                    elif phase == GRAB_ACTION:
                        release_target = last_grabbed_slot
                        release_action = GRAB_ACTION
                    else:
                        release_target = (
                            slot_of_place_digit.get(last_placed_digit, 0)
                            if last_placed_digit is not None else 0
                        )
                        release_action = PLACE_ACTION
                    enqueue_latest(q, VisionToGimbal(
                        target=release_target if release_target is not None else 0,
                        action=release_action,
                        capture=False,
                    ))
                    print(f"[动作完成] 已补发 capture=0 "
                          f"(target={release_target}, action={release_action})")

                    if tray_phase_active and tray_pending_digit is not None:
                        tray_pending_digit = None
                        grabbed_slots = set()
                        last_grabbed_slot = None
                        reset_action_state()
                        print("托盘抓取完成，已补发 capture=0，"
                              "等待下位机前往下一托盘（arrived=1）")
                        continue

                    if phase == GRAB_ACTION:
                        grabbed_new = last_grabbed_slot is not None
                        if last_grabbed_slot is not None:
                            grabbed_slots.add(last_grabbed_slot)
                            last_grabbed_slot = None
                            print(f"已抓取放入槽位: {sorted(grabbed_slots)}")

                        if len(grabbed_slots) >= len(target_colors):
                            if GRAB_ONLY:
                                print(f"第 {current_round + 1} 轮第 {round_cycles_done + 1} 次"
                                      f"抓取完成（抓取调试模式），跳过放置并退出")
                                all_done = True
                                break
                            print(f"第 {current_round + 1} 轮第 {round_cycles_done + 1} 次"
                                  f"抓取完成，前往放置区...")
                            vg = VisionToGimbal(target=0, action=PLACE_ACTION)
                            enqueue_latest(q, vg)
                            waiting_for_arrive = True
                            phase_after_arrival = PLACE_ACTION
                            prev_arrived = 0
                            reset_action_state()
                            continue
                        else:
                            reset_action_state()
                            if grabbed_new:
                                target_index += 1
                                next_code = target_colors[target_index]
                                next_label = CONFIG['colors'].get(
                                    CODE_TO_KEY.get(next_code), {}
                                ).get('label', next_code)
                                print(f"继续识别下一个目标: {next_label} "
                                      f"({target_index + 1}/{len(target_colors)})")
                            else:
                                print("收到完成信号但未记录新抓取，继续识别当前目标")
                            continue
                    else:
                        if last_placed_digit is not None:
                            done_digit = last_placed_digit
                            placed_digits.add(done_digit)
                            placed_order.append(done_digit)
                            placed_block_slots.add(slot_of_place_digit[done_digit])
                            last_placed_digit = None
                            print(f"已放置数字: {sorted(placed_digits)}，"
                                  f"实际顺序: {placed_order}")
                            append_placement_record(
                                f"第{current_round + 1}轮 第{round_cycles_done + 1}遍 "
                                f"第{len(placed_order)}个 放置数字{done_digit}"
                                f"（已放置顺序: {','.join(map(str, placed_order))}）"
                            )

                        if len(placed_digits) >= len(rounds[current_round]["grab"]):
                            repeat = rounds[current_round].get("repeat", 1)
                            is_last_pass = (round_cycles_done + 1 >= repeat)
                            want_tray_phase = (
                                TRAY_PHASE_ENABLED
                                and bool(placed_order)
                                and not (TRAY_PHASE_SKIP_LAST_OF_ROUND and is_last_pass)
                            )
                            if want_tray_phase:
                                tray_phase_active = True
                                tray_order_label = (
                                    "倒序"
                                    if TRAY_PHASE_ORDER in ("reverse", "reversed", "倒序")
                                    else "实际顺序"
                                )
                                tray_targets = (
                                    list(reversed(placed_order))
                                    if TRAY_PHASE_ORDER in ("reverse", "reversed", "倒序")
                                    else list(placed_order)
                                )
                                # 进入阶段时一次算好每个托盘对应的物块颜色，
                                # 避免后面 slot_of_color 被单托盘映射覆盖后找不到颜色
                                tray_plan = []
                                for _tray_digit in tray_targets:
                                    _place_slot = slot_of_place_digit.get(_tray_digit)
                                    _tray_color = None
                                    if _place_slot is not None:
                                        _tray_color = next(
                                            (c for c, s in slot_of_color.items()
                                             if s == _place_slot),
                                            None,
                                        )
                                    if _tray_color is None:
                                        print(f"[托盘] 托盘{_tray_digit} "
                                              f"找不到对应物块颜色，跳过")
                                        continue
                                    tray_plan.append((_tray_digit, _tray_color))
                                plan_digits = ",".join(
                                    str(d) for d, _ in tray_plan
                                )
                                plan_colors = ",".join(
                                    str(c) for _, c in tray_plan
                                )
                                print(f"[托盘计划] 第{current_round + 1}轮 "
                                      f"第{round_cycles_done + 1}遍 {tray_order_label}抓取="
                                      f"{plan_digits}，对应颜色={plan_colors}")
                                append_placement_record(
                                    f"第{current_round + 1}轮 "
                                    f"第{round_cycles_done + 1}遍 托盘计划："
                                    f"{tray_order_label}抓取={plan_digits}，"
                                    f"对应颜色={plan_colors}"
                                )
                                log_tray_order(
                                    current_round + 1,
                                    round_cycles_done + 1,
                                    placed_order,
                                    TRAY_PHASE_ORDER,
                                )
                                # 先置 1：等 arrived 掉到 0 再重新上升，
                                # 避免把放置区旧的 arrived=1 误当成托盘阶段触发
                                prev_arrived = 1
                                reset_action_state()
                                tray_pending_digit = None
                                print(f"第 {current_round + 1} 轮第 {round_cycles_done + 1} 次"
                                      f"放置完成，进入托盘阶段（{tray_order_label}抓取）: "
                                      f"{tray_targets}，"
                                      f"等待下位机 arrived")
                                continue
                            result = advance_after_placement_cycle()
                            if result == "done":
                                all_done = True
                                break
                            continue
                        else:
                            # 还有位置没放完，下位机自己移动到下一个位置
                            reset_action_state()
                            place_waiting_arrived = True
                            print("本槽放置完成，等待下位机到达下一个放置位置（arrived=1）")
                            continue

                # ==================== 托盘阶段（按配置顺序抓取托盘上的物块） ====================
                # 复用抓取阶段的视觉跟踪逻辑：按托盘对应物块颜色识别、
                # 动态调夹爪，位置稳定且对准后才发 capture=1。
                if tray_phase_active:
                    if arrived_rising:
                        if tray_pending_digit is not None:
                            # 上一托盘还没抓完：这个 arrived 上升沿只是干扰，
                            # 必须直接跳过本帧，否则会掉进下面“所有托盘已抓完”
                            # 的分支，把托盘阶段提前结束（表现为目标颜色突然跳变）。
                            continue
                        elif tray_plan:
                            tray_digit, tray_color = tray_plan.pop(0)
                            tray_pending_digit = tray_digit
                            target_colors = [tray_color]
                            target_index = 0
                            slot_of_color = {tray_color: tray_digit}
                            grabbed_slots = set()
                            last_grabbed_slot = None
                            phase = GRAB_ACTION
                            prev_arrived = 1
                            reset_action_state()
                            remaining = [
                                f"{d}:{c}" for d, c in tray_plan
                            ]
                            print(f"[托盘] 当前抓取对象: 托盘{tray_digit}，"
                                  f"物块颜色={tray_color}，剩余计划: {remaining}")
                            append_placement_record(
                                f"第{current_round + 1}轮 "
                                f"第{round_cycles_done + 1}遍 托盘阶段 "
                                f"当前抓取对象: 托盘{tray_digit} 颜色={tray_color}"
                            )
                            continue

                        # 所有托盘已抓完，收到 arrived 说明最后一个托盘抓取已结束
                        tray_phase_active = False
                        tray_targets = []
                        tray_plan = []
                        tray_pending_digit = None
                        print("托盘阶段完成，前往下一遍/下一轮")
                        result = advance_after_placement_cycle()
                        # advance_after_placement_cycle 已把 prev_arrived 置 0：
                        # 这个 arrived=1 就是下位机到达下一区域（抓取区）的信号，
                        # 不能再吞掉，否则会永远等不到 0→1 上升沿而卡住。
                        if result == "done":
                            all_done = True
                            break
                        continue

                    if tray_pending_digit is None:
                        if frame is not None:
                            cv2.putText(frame, "Tray phase: waiting arrived ...",
                                        (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 255, 255), 2)
                            show_detection(frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        continue

                    # 已到达当前托盘：继续走下方 GRAB_ACTION 视觉跟踪/抓取逻辑

                if frame is not None:
                    h_img, w_img = frame.shape[:2]

                    # 修正 ROI（仅首次），后续直接绘制
                    if not roi_clamped and detector.detection_area is not None:
                        detector.detection_area = clamp_roi(detector.detection_area, frame.shape)
                        roi_clamped = True
                    if detector.detection_area is not None:
                        rx, ry, rw, rh = detector.detection_area
                        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
                        cv2.putText(frame, "ROI", (rx + 5, ry + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # 显示 QR 和状态
                    if scan_QRcode_andlist.groups:
                        qr_text = "+".join(scan_QRcode_andlist.groups)
                        cv2.putText(frame, qr_text, (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # 底盘坐标 + 实际速度（右上角，单位为下位机原始单位 mm / mm/s）
                    chassis_speed = np.hypot(chassis_vx, chassis_vy)
                    cv2.putText(frame, f"chassis:({chassis_x},{chassis_y})mm V={chassis_speed:.0f}mm/s",
                                (w_img - 300, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (200, 200, 200), 1)

                    if phase == GRAB_ACTION:
                        current_code = target_colors[target_index]
                        current_label = COLOR_LABEL_EN.get(
                            CODE_TO_KEY.get(current_code), current_code
                        )
                        t = f"Grab {target_index + 1}/{len(target_colors)}: {current_label}"
                        color = (0, 200, 255) if not waiting_for_next else (0, 255, 100)
                        cv2.putText(frame, t, (50, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # ==================== 放置阶段 ====================
                    if phase == PLACE_ACTION:
                        if waiting_for_arrive:
                            area_name = (
                                "placement" if phase_after_arrival == PLACE_ACTION else "grab"
                            )
                            cv2.putText(frame, f"Waiting for arrival at {area_name} area...",
                                        (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 255, 255), 2)
                            show_detection(frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                            continue

                        if waiting_for_next:
                            if not capture_ack_received and last_capture_vg is not None:
                                elapsed_since_resend = (
                                    time.time() - capture_last_sent_time
                                    if capture_last_sent_time else 0
                                )
                                if elapsed_since_resend >= CAPTURE_RESEND_INTERVAL:
                                    enqueue_latest(q, last_capture_vg)
                                    capture_last_sent_time = time.time()
                                    print(f"[重发] 放置请求未确认，重发数字{last_placed_digit}")

                            cv2.putText(frame, f"Placing digit {last_placed_digit}...",
                                        (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 255, 100), 2)
                            show_detection(frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                            continue

                        # 放置完一个槽位后：先等 finish_capture（上面的 waiting_for_next），
                        # 再等一次新的 arrived 0→1，才允许识别下一个放置位置
                        if place_waiting_arrived:
                            if not arrived_rising:
                                cv2.putText(frame, "Waiting for arrived=1 ...",
                                            (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                            (0, 255, 255), 2)
                                show_detection(frame)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                                continue
                            place_waiting_arrived = False
                            print("已到达下一个放置位置，继续识别")

                        # 下位机 arrived=1 时才允许识别/放置，否则不识别
                        # （下位机移动中 arrived=0，自然停在这里等它到位）
                        if chassis_data2 is None or chassis_data2.arrived != 1:
                            cv2.putText(frame, "Waiting for arrived=1 ...",
                                        (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 255, 255), 2)
                            show_detection(frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                            continue

                        if placement_recognizer is None:
                            placement_recognizer = PlacementRecognizer()

                        expected_order = ",".join(rounds[current_round]["place"])
                        got_order = ",".join(map(str, placed_order)) if placed_order else "-"
                        if placed_order:
                            tray_order = (
                                ",".join(map(str, reversed(placed_order)))
                                if TRAY_PHASE_ORDER in ("reverse", "reversed", "倒序")
                                else ",".join(map(str, placed_order))
                            )
                        else:
                            tray_order = "-"
                        cv2.putText(frame,
                                    f"Exp: {expected_order}  Got: {got_order}  Tray: {tray_order}",
                                    (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 255), 2)

                        all_rings = placement_recognizer.recognize_all(frame)

                        # 所有检测到的圆环都画到主画面上，方便看识别情况
                        for ring in all_rings:
                            rx, ry = int(ring["center"][0]), int(ring["center"][1])
                            rr = int(ring["radius"])
                            cv2.circle(frame, (rx, ry), rr, (0, 255, 0), 2)
                            cv2.circle(frame, (rx, ry), 3, (0, 255, 0), -1)
                            label = str(ring["digit"]) if ring["digit"] is not None else "?"
                            conf_txt = (f"{ring['confidence']:.2f}"
                                        if ring["digit"] is not None else "")
                            cv2.putText(frame, f"{label} {conf_txt}",
                                        (rx - 30, ry - rr - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 0), 1)

                        candidates = [
                            r for r in all_rings
                            if r["digit"] is not None
                            and r["digit"] not in placed_digits
                            and r["digit"] in slot_of_place_digit
                        ]

                        if not candidates:
                            now = time.time()
                            if all_rings and now - last_place_dbg >= 1.0:
                                last_place_dbg = now
                                digits_seen = sorted(
                                    {r["digit"] for r in all_rings
                                     if r["digit"] is not None}
                                )
                                print(
                                    f"[放置] 识别到数字 {digits_seen}，"
                                    f"槽位映射={slot_of_place_digit}，"
                                    f"已放={sorted(placed_digits)}，无候选，未发送"
                                )
                            cv2.putText(frame, "Looking for ring digit...",
                                        (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 0, 255), 2)
                            show_detection(frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                            continue

                        # 多个圆环同时可见时，优先选择离图像中心最近（当前下位机所在位置）的圆环
                        target = min(
                            candidates,
                            key=lambda r: (
                                (r["center"][0] - w_img / 2.0) ** 2
                                + (r["center"][1] - h_img / 2.0) ** 2
                            ),
                        )
                        digit = target["digit"]
                        slot_index = slot_of_place_digit[digit]
                        ring_cx, ring_cy = target["center"]
                        ring_cx, ring_cy = int(ring_cx), int(ring_cy)

                        cv2.circle(frame, (ring_cx, ring_cy), int(target["radius"]),
                                   (255, 0, 255), 2)
                        cv2.circle(frame, (ring_cx, ring_cy), 4, (255, 0, 255), -1)
                        cv2.putText(frame, f"digit={digit} conf={target['confidence']:.2f}",
                                    (ring_cx - 60,
                                     int(ring_cy - target['radius'] - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                        # 位置连续稳定 N 帧才允许放置（参数在 config.yaml → placement）
                        place_stable_threshold = PLACE_STABLE_THRESHOLD
                        place_move_max = PLACE_STABLE_MAX_MOVE
                        if place_last_center is None or place_last_digit != digit:
                            place_stable_count = 1
                        else:
                            move_px = np.hypot(
                                ring_cx - place_last_center[0],
                                ring_cy - place_last_center[1],
                            )
                            place_stable_count = (
                                place_stable_count + 1 if move_px < place_move_max else 1
                            )
                        place_last_center = (float(ring_cx), float(ring_cy))
                        place_last_digit = digit
                        position_stable = (
                            place_stable_count >= place_stable_threshold
                        )

                        # 只比较 x 轴（左右）偏差；y 轴对应前后距离，
                        # 由底盘 y / 夹爪伸长量按 mm 死区闭环，不作为图像对准判据
                        cur_offset = abs(ring_cx - w_img // 2)
                        capture = (
                            position_stable
                            and cur_offset <= PLACE_CENTER_TOLERANCE_PX
                        )
                        cv2.putText(
                            frame,
                            f"place stable: {place_stable_count}/{place_stable_threshold}",
                            (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 100), 1,
                        )

                        coord = None
                        if transformer.CAMERA_FOCAL_PX_X is not None:
                            if PLACE_GRIPPER_FIXED_CUSTOM:
                                place_gripper_extend_cm = PLACE_GRIPPER_EXTEND_CM
                                place_fixed_gripper_cm = (
                                    transformer.min_jar_dis[1] + place_gripper_extend_cm
                                )
                            elif PLACE_GRIPPER_FIXED_MAX:
                                place_gripper_extend_cm = PLACE_GRIPPER_EXTEND_CM
                                place_fixed_gripper_cm = transformer.max_jar_dis[1]
                            elif PLACE_GRIPPER_FIXED_MIN:
                                place_gripper_extend_cm = 0.0
                                place_fixed_gripper_cm = transformer.min_jar_dis[1]
                            else:
                                place_gripper_extend_cm = current_gripper_cm
                                place_fixed_gripper_cm = None
                            coord = transformer.pixel_to_camera(
                                ring_cx, ring_cy,
                                image_width=w_img,
                                image_height=h_img,
                                block_height_cm=0.0,   # 放置区对准的是地面圆环
                                gripper_extension_cm=place_gripper_extend_cm,
                            )

                        if coord is not None:
                            cmd_mm = transformer.command_to_protocol_mm(
                                coord,
                                gripper_extension_cm=place_gripper_extend_cm,
                                fixed_gripper_cm=place_fixed_gripper_cm,
                            )
                            if cmd_mm == (0, 0, 0):
                                warn_zero_command("放置", coord, ring_cx, ring_cy)
                            chassis_x_mm, chassis_y_mm, gripper_mm = cmd_mm
                            if (
                                PLACE_GRIPPER_FIXED_CUSTOM
                                or PLACE_GRIPPER_FIXED_MAX
                                or PLACE_GRIPPER_FIXED_MIN
                            ):
                                # 放置时夹爪固定（min/max/自定义 mm）：只调底盘
                                gripper_mm = PLACE_GRIPPER_MM
                            chassis_x_mm, chassis_y_mm, gripper_mm = sanitize_protocol_mm(
                                (chassis_x_mm, chassis_y_mm, gripper_mm), last_valid_mm
                            )
                            desired_mm = (chassis_x_mm, chassis_y_mm, gripper_mm)
                            chassis_x_mm, chassis_y_mm, gripper_mm = smooth_tracking_cmd(
                                desired_mm, time.time()
                            )
                            last_valid_mm = (chassis_x_mm, chassis_y_mm, gripper_mm)
                        else:
                            chassis_x_mm, chassis_y_mm, gripper_mm = last_valid_mm
                            desired_mm = last_valid_mm
                            warn_invalid_coord(
                                "放置", ring_cx, ring_cy, w_img, h_img
                            )

                        vg = VisionToGimbal(
                            target=slot_index,
                            action=PLACE_ACTION,
                            capture=capture,
                            chassis_x_mm=chassis_x_mm,
                            chassis_y_mm=chassis_y_mm,
                            gripper_mm=gripper_mm,
                        )
                        sent = tracking_send_allowed(capture, desired_mm)
                        if sent:
                            enqueue_latest(q, vg)
                            log_command(
                                "放置", slot_index, PLACE_ACTION, capture,
                                chassis_x_mm, chassis_y_mm, gripper_mm,
                                chassis_x, chassis_y, chassis_vx, chassis_vy,
                                camera_coord=coord,
                                world_coord=transformer.camera_to_world(coord),
                            )
                        else:
                            now = time.time()
                            if now - last_place_dbg >= 1.0:
                                last_place_dbg = now
                                print(
                                    f"[放置] 识别到了但未发送: capture={capture} "
                                    f"cmd={desired_mm} 上次发送={last_sent_tracking_mm} "
                                    f"距上次发送={now - last_tracking_send:.2f}s "
                                    f"x偏移={cur_offset}px"
                                )

                        if capture and last_placed_digit is None:
                            last_placed_digit = digit
                            detection_sent = True
                            waiting_for_next = True
                            sent_time = time.time()
                            capture_ack_received = False
                            capture_last_sent_time = time.time()
                            last_capture_vg = vg
                            print(f"圆环数字 {digit} 已对准(x偏移={cur_offset}px)，放置槽位 {slot_index} 的物块")

                        show_detection(frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        continue

                    # ---- 等待切换（根据模式） ----
                    if waiting_for_next:
                        hint = "Waiting for finish_capture"

                        # ---- 未收到确认时按间隔重发抓取请求 ----
                        if not capture_ack_received and last_capture_vg is not None:
                            elapsed_since_resend = (
                                time.time() - capture_last_sent_time
                                if capture_last_sent_time else 0
                            )
                            if elapsed_since_resend >= CAPTURE_RESEND_INTERVAL:
                                enqueue_latest(q, last_capture_vg)
                                capture_last_sent_time = time.time()
                                print(f"[重发] capture=1 未收到确认，"
                                      f"重发槽位{last_grabbed_slot}抓取请求")

                        # 重绘上一个已检测到的物块（避免圆消失）
                        if last_detected_center is not None:
                            cv2.circle(frame, last_detected_center,
                                       last_detected_radius, last_detected_color, 2)
                            cv2.circle(frame, last_detected_center, 3,
                                       last_detected_color, -1)
                            cv2.putText(frame, last_detected_label,
                                        (last_detected_center[0] - 20,
                                         last_detected_center[1] - last_detected_radius - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, last_detected_color, 2)

                        cv2.putText(frame, hint,
                                    (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 255), 2)

                        show_detection(frame)
                        key = cv2.waitKey(1) & 0xFF

                        if key == ord('q'):
                            break

                        # 等待中不执行检测，直接到循环末尾显示
                        continue

                    # ---- 正常检测模式 ----
                    if not recognition_started:
                        recognition_started = True
                        if tray_phase_active and tray_pending_digit is not None:
                            print(f"[托盘] 开始识别托盘{tray_pending_digit}物块"
                                  f"（目标颜色={target_colors[target_index]}）")
                        else:
                            print(
                                "[识别] 开始抓取区识别"
                                f"（等待 arrived: {'是' if waiting_for_arrive else '否'}）"
                            )
                    current_time = cv2.getTickCount()
                    # 按 QR 顺序只检测当前目标颜色，避免每帧构建 6 种颜色掩膜
                    current_target_code = target_colors[target_index]
                    if current_target_code in CODE_TO_KEY:
                        data, current_center, current_color = block_preprocessing(
                            frame, target=current_target_code
                        )
                    else:
                        # 无效颜色代码（不在 1~6）：本帧不检测，避免白白构建全部掩膜
                        data = current_center = current_color = None

                    # 识别驱动：只接受本轮抓取序列中的颜色，且该槽位还没抓过
                    if data and current_color:
                        if current_color != current_target_code:
                            data = None
                            current_center = None
                            current_color = None
                        slot_index = slot_of_color.get(current_color)
                        if slot_index is None or slot_index in grabbed_slots:
                            data = None
                            current_center = None
                            current_color = None

                    if data and current_color:
                        last_detection_time = current_time
                        # 更新两个独立稳定计数（位置/颜色），
                        # 实际控制逻辑见下方 color_stable / position_stable
                        detector.update_stability(current_center, current_color)

                        # ---- 位置滤波：一欧元 / 卡尔曼 / 原始 ----
                        if FILTER_TYPE == "one_euro":
                            fx, fy, fvx, fvy = one_euro_tracker.update(
                                current_center[0], current_center[1], frame_t
                            )
                            fax = fay = 0.0
                        elif FILTER_TYPE == "kalman":
                            # 按实际帧间隔更新 dt：固定 1/30 在帧率不足/波动时会
                            # 造成滤波滞后、速度估计偏差，表现为跟踪“卡”。
                            dt = 1.0 / 30.0
                            if last_kf_time is not None:
                                dt = min(max(frame_t - last_kf_time, 0.005), 0.2)
                            kf.set_dt(dt)
                            kf.predict()
                            kf.update(current_center[0], current_center[1])
                            last_kf_time = frame_t
                            fx, fy, fvx, fvy, fax, fay = kf.get_state()
                        else:
                            # 不用卡尔曼：直接以当前帧检测中心为准，不估计速度/加速度
                            fx, fy = float(current_center[0]), float(current_center[1])
                            fvx = fvy = fax = fay = 0.0

                        # 可视化
                        color_key = CODE_TO_KEY.get(current_color)
                        draw_color = COLOR_BGR.get(color_key, (255, 255, 255))
                        radius = detector.last_radius
                        viz = KALMAN_CFG["visualize"]
                        viz_on = viz["enabled"]

                        # 原始测量（虚线细圈）
                        if viz_on and viz["draw_raw"]:
                            cv2.circle(frame, current_center, radius, draw_color, 1)
                            cv2.circle(frame, current_center, 2, (150, 150, 150), -1)

                        # 滤波后（实线粗圈）
                        filtered_center = (int(fx), int(fy))
                        if viz_on and viz["draw_filtered"]:
                            cv2.circle(frame, filtered_center, radius, draw_color, 2)
                            cv2.circle(frame, filtered_center, 4, draw_color, -1)

                        label = COLOR_LABEL_EN.get(
                            CODE_TO_KEY.get(current_color), "?"
                        )
                        if viz_on:
                            cv2.putText(frame, label,
                                        (filtered_center[0] - 20, filtered_center[1] - radius - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)
                            # 速度标注
                            if viz["draw_speed"]:
                                speed = np.sqrt(fvx**2 + fvy**2)
                                cv2.putText(frame, f"V={speed:.0f}px/s",
                                            (filtered_center[0] + radius + 5, filtered_center[1]),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 180), 1)

                        # 保存最近一次检测结果（等待阶段重绘用）
                        last_detected_center = filtered_center
                        last_detected_radius = radius
                        last_detected_color = draw_color
                        last_detected_label = label

                        if viz_on:
                            # ==== 滤波/预测轨迹 + 路径可视化 ====
                            h, w = frame.shape[:2]
                            cx, cy = w // 2, h // 2

                            # ── 图像中心十字 ──
                            cv2.drawMarker(frame, (cx, cy), (255, 255, 255),
                                           cv2.MARKER_CROSS, 12, 1)

                            # ── 滤波历史轨迹（滤波后位置，橙黄色线）──
                            if viz["draw_history"] and FILTER_TYPE in ("kalman", "one_euro"):
                                trail_len = int(viz["history_trail_len"])
                                trail_src = (
                                    kf.history if FILTER_TYPE == "kalman"
                                    else one_euro_tracker.history
                                )
                                trail = [(int(p[0]), int(p[1])) for p in trail_src][-trail_len:]
                                if len(trail) >= 2:
                                    for p1, p2 in zip(trail[:-1], trail[1:]):
                                        cv2.line(frame, p1, p2, (0, 200, 255), 1)

                            # ── 预测轨迹（青色，步数/时长由配置控制）──
                            if viz["draw_trajectory"] or viz["draw_intercept"]:
                                if FILTER_TYPE == "kalman":
                                    future = kf.predict_future()
                                elif FILTER_TYPE == "one_euro":
                                    future = one_euro_tracker.predict_future(
                                        T=KALMAN_CFG["predict"]["horizon_s"],
                                        steps=KALMAN_CFG["predict"]["steps"],
                                    )
                                else:
                                    future = []
                            if viz["draw_trajectory"]:
                                prev = filtered_center
                                for i, (px, py) in enumerate(future):
                                    pt = (int(px), int(py))
                                    alpha = 1.0 - i * 0.12
                                    color = (0, int(255 * alpha), int(255 * alpha))
                                    if i == 0:
                                        cv2.arrowedLine(frame, prev, pt, color, 1, tipLength=0.3)
                                    else:
                                        cv2.line(frame, prev, pt, color, 1)
                                    cv2.circle(frame, pt, 2, color, -1)
                                    prev = pt

                            # ── 拦截点（预测轨迹上距中心最近的点）──
                            if viz["draw_intercept"]:
                                all_pts = [filtered_center] + [(int(p[0]), int(p[1])) for p in future]
                                best_d = float("inf")
                                best_pt = filtered_center
                                for pt in all_pts:
                                    d = np.sqrt((pt[0] - cx)**2 + (pt[1] - cy)**2)
                                    if d < best_d:
                                        best_d = d
                                        best_pt = pt
                                if best_pt != (cx, cy):
                                    cv2.arrowedLine(frame, (cx, cy), best_pt,
                                                    (0, 0, 255), 2, tipLength=0.15)
                                    cv2.circle(frame, best_pt, 5, (0, 0, 255), -1)
                                cv2.putText(frame, f"intercept:{best_d:.0f}px",
                                            (cx + 20, cy - 8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                            # 显示检测状态
                            track_label = {
                                "kalman": "KF",
                                "one_euro": "OneEuro",
                                "none": "Raw",
                            }.get(FILTER_TYPE, "Track")
                            status = (
                                f"{track_label} tracking "
                                f"(stable: {detector.stable_count}/{detector.stability_settings['threshold']} "
                                f"color: {detector.color_stable_count}/"
                                f"{detector.stability_settings['color_stable_threshold']})"
                            )
                            cv2.putText(frame, status, (50, 160),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        # ---- 向云台发送跟踪/抓取指令 ----
                        # 颜色稳定后即可开始跟踪移动（capture=0）；
                        # 位置稳定（threshold）且对准后才允许抓取（capture=1）。
                        color_stable = (
                            detector.color_stable_count
                            >= detector.stability_settings['color_stable_threshold']
                        )
                        position_stable = (
                            detector.stable_count
                            >= detector.stability_settings['threshold']
                        )
                        if not detection_sent and color_stable:
                            # 颜色稳定后用博弈求解拦截点
                            car_v = np.hypot(chassis_vx, chassis_vy) * PX_PER_MM
                            intercept = planner.solve(
                                block_x=fx, block_y=fy,
                                block_vx=fvx, block_vy=fvy,
                                block_ax=fax, block_ay=fay,
                                car_x=chassis_x * PX_PER_MM,
                                car_y=chassis_y * PX_PER_MM,
                                car_v=car_v,
                            )

                            if intercept and intercept["feasible"]:
                                target_x, target_y = int(intercept["x"]), int(intercept["y"])
                                T_solve = intercept["T"]
                            else:
                                # 规划无解 → 用当前位置
                                target_x, target_y = filtered_center
                                T_solve = 0.0

                            # 只比较 x 轴（左右）偏差；y 轴对应前后距离，
                            # 由底盘 y / 夹爪伸长量按 mm 死区闭环，不作为图像对准判据
                            cur_offset = abs(fx - cx)

                            # 目标中心必须在检测 ROI 内（含滤波滞后），
                            # 防止物块已经跑出区域仍被切边检测/旧滤波位置触发抓取
                            roi = detector.detection_area
                            target_in_roi = (
                                roi is None
                                or (roi[0] <= fx <= roi[0] + roi[2]
                                    and roi[1] <= fy <= roi[1] + roi[3])
                            )

                            # 位置未稳定（或尚未对准）→ capture=0；
                            # 位置稳定且对准 → capture=1，请求下位机抓取
                            would_capture = (
                                position_stable
                                and cur_offset <= GRAB_CENTER_TOLERANCE_PX
                            )
                            capture = would_capture and target_in_roi
                            if would_capture and not target_in_roi:
                                print(
                                    f"[ROI] 目标已出检测区 ({fx:.0f},{fy:.0f})，"
                                    f"不在 {roi} 内，暂不抓取"
                                )

                            # 托盘阶段固定夹爪时：夹爪长度固定，只靠底盘把物块
                            # 带到固定距离，避免“相机随夹爪伸缩 → 测量滞后 → 闭环震荡”。
                            _tray_fixed_gripper = (
                                tray_phase_active
                                and (TRAY_GRIPPER_FIXED_CUSTOM
                                     or TRAY_GRIPPER_FIXED_MIN
                                     or TRAY_GRIPPER_FIXED_MAX)
                            )
                            if _tray_fixed_gripper:
                                if TRAY_GRIPPER_FIXED_CUSTOM:
                                    track_gripper_cm = TRAY_GRIPPER_EXTEND_CM
                                    track_fixed_gripper_cm = (
                                        transformer.min_jar_dis[1]
                                        + TRAY_GRIPPER_EXTEND_CM
                                    )
                                elif TRAY_GRIPPER_FIXED_MAX:
                                    track_gripper_cm = transformer.MAX_GRIPPER_EXTEND_CM
                                    track_fixed_gripper_cm = transformer.max_jar_dis[1]
                                else:
                                    track_gripper_cm = 0.0
                                    track_fixed_gripper_cm = transformer.min_jar_dis[1]
                            else:
                                track_gripper_cm = current_gripper_cm
                                track_fixed_gripper_cm = None

                            if transformer.CAMERA_FOCAL_PX_X is not None:
                                block_height = (
                                    transformer.BLOCK_HEIGHT_PLACED_CM
                                    if (tray_phase_active
                                        or slot_index in placed_block_slots)
                                    else transformer.BLOCK_HEIGHT_CM
                                )
                                block_camera_coord = transformer.pixel_to_camera(
                                    target_x,
                                    target_y,
                                    image_width=frame.shape[1],
                                    image_height=frame.shape[0],
                                    block_height_cm=block_height,
                                    gripper_extension_cm=track_gripper_cm,
                                )
                            else:
                                block_camera_coord = None

                            if block_camera_coord is not None:
                                cmd_mm = transformer.command_to_protocol_mm(
                                    block_camera_coord,
                                    gripper_extension_cm=track_gripper_cm,
                                    fixed_gripper_cm=track_fixed_gripper_cm,
                                )
                                if cmd_mm == (0, 0, 0):
                                    warn_zero_command(
                                        "抓取", block_camera_coord, target_x, target_y
                                    )
                                chassis_x_mm, chassis_y_mm, gripper_mm = cmd_mm
                                if track_fixed_gripper_cm is not None:
                                    gripper_mm = TRAY_GRIPPER_MM
                                chassis_x_mm, chassis_y_mm, gripper_mm = sanitize_protocol_mm(
                                    (chassis_x_mm, chassis_y_mm, gripper_mm), last_valid_mm
                                )
                                desired_mm = (chassis_x_mm, chassis_y_mm, gripper_mm)
                                chassis_x_mm, chassis_y_mm, gripper_mm = smooth_tracking_cmd(
                                    desired_mm, time.time()
                                )
                                last_valid_mm = (chassis_x_mm, chassis_y_mm, gripper_mm)
                            else:
                                # 坐标无效时沿用上一帧有效指令，避免下位机误以为停止
                                chassis_x_mm, chassis_y_mm, gripper_mm = last_valid_mm
                                desired_mm = last_valid_mm
                                warn_invalid_coord(
                                    "抓取", target_x, target_y,
                                    frame.shape[1], frame.shape[0],
                                )

                            vg = VisionToGimbal(
                                target=slot_index,
                                action=GRAB_ACTION,
                                capture=capture,
                                chassis_x_mm=chassis_x_mm,
                                chassis_y_mm=chassis_y_mm,
                                gripper_mm=gripper_mm,
                            )
                            if tracking_send_allowed(capture, desired_mm):
                                enqueue_latest(q, vg)
                                log_command(
                                    "抓取", slot_index, GRAB_ACTION, capture,
                                    chassis_x_mm, chassis_y_mm, gripper_mm,
                                    chassis_x, chassis_y, chassis_vx, chassis_vy,
                                    camera_coord=block_camera_coord,
                                    world_coord=(
                                        transformer.camera_to_world(block_camera_coord)
                                        if block_camera_coord is not None else None
                                    ),
                                )

                                if capture:
                                    print(f"物块已对准 (x偏移={cur_offset:.0f}px)，请求抓取 "
                                          f"颜色{current_color} → 槽位{slot_index}")
                                    C_1 = detector.final_color
                                    last_grabbed_slot = slot_index
                                    detection_sent = True
                                    sent_time = time.time()
                                    capture_ack_received = False
                                    capture_last_sent_time = time.time()
                                    last_capture_vg = vg
                                    waiting_for_next = True
                                    print("已请求抓取，等待下位机 finish_capture=1")

                            if position_stable and not capture:
                                cv2.putText(frame,
                                    f"Aligning... off={cur_offset:.0f}px T={T_solve:.2f}s "
                                    f"-> ({target_x},{target_y}) capture=0",
                                    (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 200, 255), 1)

                    else:
                        # 本帧未识别到目标颜色：颜色计数清零，
                        # 重新出现后必须再连续攒够 color_stable_threshold 帧
                        detector.on_miss()

                        # 未检测到物块，显示提示
                        target_label = COLOR_LABEL_EN.get(
                            CODE_TO_KEY.get(current_target_code), current_target_code
                        )
                        cv2.putText(frame, f"Looking for {target_label}...",
                                    (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        if last_detection_time is not None:
                            elapsed = (current_time - last_detection_time) / cv2.getTickFrequency() * 1000
                            if elapsed > detector.timeout_settings['timeout_ms']:
                                pass

                if frame is not None:
                    show_detection(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("用户手动终止")
    finally:
        enqueue_latest(q, None)
        sending_thread.join(timeout=2)

        if cap and cap.isOpened():
            cap.release()
        if detection_cap and detection_cap.isOpened():
            detection_cap.release()
        cv2.destroyAllWindows()

        if _serial_comm:
            try:
                _serial_comm.stop_chassis_recv()
                _serial_comm.close()
            except Exception:
                pass

        print("资源已释放")


if __name__ == "__main__":
    main()
