import cv2
import numpy as np
import os
import subprocess
import sys
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
    DETECTION_CAMERA_FPS,
)
from preprocessing import *
import scan_QRcode_andlist
from felling_color import (
    CONFIG, CODE_TO_KEY,
    block_preprocessing, get_detector, reset_detector,
)
from gimbal import SerialComm, VisionToGimbal
from kalman_tracker import KALMAN_CFG, KalmanBlockTracker, KalmanWorldTracker
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
KALMAN_WORLD_ENABLED = FILTER_TYPE == "kalman_world"
# 世界系卡尔曼是否把底盘速度回传作为控制输入（config.yaml → kalman_world）
_kw_cfg = _cfg("kalman_world", default=None) or {}
KALMAN_WORLD_USE_CHASSIS_VEL = bool(_kw_cfg.get("use_chassis_velocity", True))
# 测量用夹爪位置低通系数（config.yaml → kalman_world.gripper_meas_filter）：
# 夹爪指令跳变时相机原点做低通，越大越跟手（滞后小）、越小越平滑
KALMAN_WORLD_GRIPPER_MEAS_FILTER = min(
    1.0, max(0.0, float(_kw_cfg.get("gripper_meas_filter", 0.3)))
)

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
# 抓取/放置阶段各自独立的 y 轴（上下）对准容差（px）。
# 未配置时沿用对应 x 轴容差，保持兼容。
GRAB_CENTER_TOLERANCE_Y_PX = float(
    _cfg("control", "grab_center_tolerance_y_px", default=GRAB_CENTER_TOLERANCE_PX)
)
PLACE_CENTER_TOLERANCE_Y_PX = float(
    _cfg("control", "place_center_tolerance_y_px", default=PLACE_CENTER_TOLERANCE_PX)
)
# 抓取区“开始识别时第一目标已在场”跳过逻辑：
# 到达抓取区开始识别时（只针对本轮第一个目标），若开始识别后的前几帧内
# 就检测到第一目标颜色，说明复位期间它已经/正在转走，来不及抓；
# 本圈不跟踪不抓取，等它转一圈重新出现后再按正常流程抓取。
GRAB_SKIP_FIRST_ENABLED = bool(
    _cfg("control", "grab_skip_first_enabled", default=True)
)
GRAB_SKIP_FIRST_EVAL_FRAMES = int(
    _cfg("control", "grab_skip_first_eval_frames", default=5)
)

# ==================== 指令发送节流/死区/心跳（config.yaml → tracking） ====================
# 等待抓取时，若下位机未回传 capture_ack=1，每隔该秒数重发一次 capture=1
CAPTURE_RESEND_INTERVAL = float(_cfg("tracking", "capture_resend_interval", default=1.0))
# 普通跟踪/对准指令（capture=0）的最小发送间隔（秒）；
# capture=1、阶段切换、区域移动、重发等事件包不受此限制，仍立即发送
TRACKING_SEND_INTERVAL = float(_cfg("tracking", "send_interval", default=0.5))
# 底盘指令变化死区（mm）：目标偏移相对上次已发送值变化小于该值时不重发，
# 避免下位机闭环执行过程中被 100→90 这类微小变化反复打断。
CHASSIS_SEND_DEADBAND_MM = float(_cfg("tracking", "chassis_send_deadband_mm", default=1.0))
# 夹爪指令死区（mm）：夹爪目标为绝对伸长量，变化小于该值时不重发
GRIPPER_DEADBAND_MM = float(_cfg("tracking", "gripper_deadband_mm", default=1))
# 夹爪已改为“绝对目标直发”（与底盘绝对目标一致）：
# 目标直接取视觉换算的绝对伸长量，不再做增量比例增益/ramp 累加。
# 平滑跟踪：底盘指令每个发送周期的变化量上限（mm，按 send_interval 标定）。
# 让指令从小步连续爬升/衰减，而不是 0→30mm→0 这样跳变，避免“动一下停一下”。
CHASSIS_RAMP_STEP_MM = float(_cfg("tracking", "chassis_ramp_step_mm", default=4.0))
# 普通跟踪包的心跳间隔（秒）：即使指令变化很小，也至少按该间隔重发一次；
# 应大于 TRACKING_SEND_INTERVAL，否则死区不生效；设 None 禁用。
_send_heartbeat = _cfg("tracking", "send_heartbeat", default=5.0)
CHASSIS_SEND_HEARTBEAT = None if _send_heartbeat is None else float(_send_heartbeat)

# 底盘前瞻目标速度上限（mm/s），防止异常速度把目标推太远
CHASSIS_LOOKAHEAD_MAX_SPEED_MM_S = float(
    _cfg("tracking", "chassis_lookahead_max_speed_mm_s", default=50.0)
)
# 底盘前瞻时间（秒）：绝对目标模式下，把物块未来 T 秒后的位置作为底盘目标，
# 补偿视觉/串口/机械延迟（Pure Pursuit / Lookahead）；0=关闭
CHASSIS_LOOKAHEAD_S = float(
    _cfg("tracking", "chassis_lookahead_ms", default=0.0)
) / 1000.0
CHASSIS_LOOKAHEAD_S = max(0.0, min(CHASSIS_LOOKAHEAD_S, 0.5))

# ==================== 串口生命周期（config.yaml → serial） ====================
# 运行中串口连续断开超过该秒数后自动退出程序（systemd 重新拉起并等待，
# 重新插上串口后自动从头开始运行）；0 = 一旦检测到断开立即退出。
SERIAL_DISCONNECT_EXIT_DELAY_S = max(
    0.0, float(_cfg("serial", "disconnect_exit_delay_s", default=3.0))
)

# ==================== 显示窗口（config.yaml → display） ====================
# 宽度或高度超过时按同一比例缩小显示，避免画面超出屏幕；只影响显示，
# 不影响检测分辨率与坐标换算
DISPLAY_MAX_WIDTH = float(_cfg("display", "max_width", default=800))
DISPLAY_MAX_HEIGHT = float(_cfg("display", "max_height", default=540))
# 画面左下角叠加显示串口收发信息（只显示英文/数字，避免 OpenCV 中文乱码）
SERIAL_OVERLAY_ENABLED = bool(_cfg("display", "serial_overlay", "enabled", default=True))
SERIAL_OVERLAY_MAX = int(_cfg("display", "serial_overlay", "max_lines", default=4))

# 外接屏大字显示扫码结果（qr_display.py，config.yaml → display.qr_display）
# display 留空 = 继承本程序当前的 DISPLAY（一般即用户正在看的桌面，最稳妥）；
# 只有确实要显示到另一个 X 会话（如物理外接屏）时才填具体值，并配合 xauthority。
QR_DISPLAY_ENABLED = bool(_cfg("display", "qr_display", "enabled", default=True))
QR_DISPLAY_DISPLAY = str(
    _cfg("display", "qr_display", "display", default="") or ""
).strip()
QR_DISPLAY_XAUTHORITY = str(
    _cfg("display", "qr_display", "xauthority", default="") or ""
).strip()
QR_DISPLAY_MONITOR = _cfg("display", "qr_display", "monitor", default=None)
QR_DISPLAY_STATE_FILE = str(
    _cfg("display", "qr_display", "state_file",
         default="/tmp/qr_display_result.txt")
    or "/tmp/qr_display_result.txt"
)
QR_DISPLAY_LOG_FILE = str(
    _cfg("display", "qr_display", "log_file",
         default="/tmp/qr_display.log")
    or "/tmp/qr_display.log"
)
# true=启动时若检测到旧的显示进程（如上次运行遗留），先请其退出再接管，
# 避免旧进程占着单实例锁导致新的显示窗口一直起不来
QR_DISPLAY_REPLACE = bool(_cfg("display", "qr_display", "replace", default=True))
if not os.environ.get("QR_DISPLAY_FILE"):
    os.environ["QR_DISPLAY_FILE"] = QR_DISPLAY_STATE_FILE

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
# 放置阶段圆心 EMA 平滑系数（0~1）：越大越跟手，越小越抗识别抖动。
# 识别偶发跳边时只被衰减，不会像“丢弃帧”那样把车辆完全卡住。
PLACE_CENTER_SMOOTH_ALPHA = float(
    _cfg("placement", "place_center_smooth_alpha", default=0.5)
)
PLACE_CENTER_SMOOTH_ALPHA = min(1.0, max(0.05, PLACE_CENTER_SMOOTH_ALPHA))
# 放置阶段视觉误差增益（0~1）：目标位置 = 回传位置 + 视觉误差 × 增益。
# 降增益让底盘只响应持续存在的偏差，避免单帧识别噪声被全量放大成大幅指令。
PLACE_VISUAL_GAIN = float(_cfg("placement", "place_visual_gain", default=0.5))
PLACE_VISUAL_GAIN = min(1.0, max(0.05, PLACE_VISUAL_GAIN))
# 放置阶段底盘指令变化死区（mm），比全局死区大，过滤小幅识别抖动
PLACE_CHASSIS_DEADBAND_MM = float(
    _cfg("placement", "place_chassis_deadband_mm", default=5.0)
)
# true=到达放置区后先拿最近可见圆环（如数字 2）调整车的位置
# （只发跟踪 capture=0，不发 capture=1）；调整完成后，
# 若它不是当前应放数字，再发送应放数字（如 1），由下位机前往对应圆环放置。
PLACE_PREALIGN_ENABLED = bool(
    _cfg("placement", "place_prealign_enabled", default=True)
)
# ==================== 托盘阶段（config.yaml → placement） ====================
# 放置完成后进入托盘阶段：复用抓取阶段的视觉跟踪逻辑；
# 抓取顺序由 tray_phase_order 决定：actual=按实际放置顺序；放置顺序已固定为
# 二维码放置序列（如 132），因此 actual 等价于按抓取颜色顺序（如 345）抓回。
# reverse=按实际放置顺序倒序。
TRAY_PHASE_ENABLED = bool(_cfg("placement", "tray_phase_enabled", default=True))
TRAY_PHASE_ORDER = str(_cfg("placement", "tray_phase_order", default="actual")).strip().lower()
TRAY_PHASE_IS_REVERSE = TRAY_PHASE_ORDER in ("reverse", "reversed", "倒序")
TRAY_PHASE_ACTION = int(_cfg("placement", "tray_phase_action", default=GRAB_ACTION))
TRAY_PHASE_CAPTURE = bool(_cfg("placement", "tray_phase_capture", default=True))
TRAY_PHASE_SKIP_LAST_OF_ROUND = bool(
    _cfg("placement", "tray_phase_skip_last_of_round", default=True)
)
# 托盘阶段 arrived 处理（config.yaml → placement.tray_phase_arrived_mode）：
# edge（默认）=逐托盘等待 arrived 0→1 上升沿；进入托盘阶段时 prev_arrived 置 0，
#   放置完成后的 arrived=1 会被当作第一个托盘（同一位置）的到达信号。
# none/immediate/不等待 =完全不依赖 arrived：进入托盘阶段立即开始第一个托盘，
#   每抓完一个立即开始下一个（下位机 arrived 持续为 1、不产生上升沿时使用）。
TRAY_PHASE_ARRIVED_MODE = str(
    _cfg("placement", "tray_phase_arrived_mode", default="edge")
).strip().lower()
TRAY_PHASE_WAIT_ARRIVED = TRAY_PHASE_ARRIVED_MODE not in (
    "none", "immediate", "direct", "直通", "不等待", "不等"
)
# 放置完最后一个物块、下位机回传 finish_capture=1 后，延时该秒数再发送
# 托盘阶段的“前往第一个托盘”移动指令（number=托盘号），给下位机留出
# 完成动作/稳定时间；0=不延时（原行为，立即发送）。
TRAY_PHASE_ENTRY_DELAY_S = float(
    _cfg("placement", "tray_phase_entry_delay_s", default=2.0)
)
# 托盘阶段夹爪策略（config.yaml → placement.tray_gripper_fixed）：
# 倒序托盘阶段保持该策略（dynamic/null=动态调夹爪；min/max/数字mm=固定夹爪长度，
# 只靠底盘对准）；非倒序（actual）托盘阶段复用第一次抓取方式。
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

# 抓取阶段夹爪策略（config.yaml → tracking.grab_gripper_fixed）：
# dynamic/null=动态调夹爪；数字mm=固定夹爪伸长量，只靠底盘对准（与托盘阶段同理）。
_grab_fixed_cfg = _cfg("tracking", "grab_gripper_fixed", default=None)
GRAB_GRIPPER_FIXED_CUSTOM = False
GRAB_GRIPPER_EXTEND_MM = 0
GRAB_GRIPPER_EXTEND_CM = 0.0
GRAB_GRIPPER_MM = 0
try:
    _grab_fixed_mm = float(_grab_fixed_cfg)
except (TypeError, ValueError):
    _grab_fixed_mm = None
if _grab_fixed_mm is not None:
    GRAB_GRIPPER_FIXED_CUSTOM = True
    GRAB_GRIPPER_EXTEND_MM = int(round(_grab_fixed_mm))
    GRAB_GRIPPER_EXTEND_MM = min(
        max(GRAB_GRIPPER_EXTEND_MM, 0), transformer.MAX_GRIPPER_EXTEND_MM
    )
    GRAB_GRIPPER_EXTEND_CM = GRAB_GRIPPER_EXTEND_MM / 10.0
    GRAB_GRIPPER_MM = GRAB_GRIPPER_EXTEND_MM

# 固定夹爪抓取时，先等夹爪反馈到位再动底盘（倒序托盘阶段除外）：
# 避免夹爪还在伸长时底盘就带着相机移动，造成测量/对准偏差。
GRIPPER_SETTLE_TOLERANCE_MM = float(
    _cfg("tracking", "grab_gripper_settle_tolerance_mm", default=3.0)
)
GRIPPER_SETTLE_TIMEOUT_S = float(
    _cfg("tracking", "grab_gripper_settle_timeout_s", default=5.0)
)

# ==================== 日志打印节流（config.yaml → logging） ====================
# 指令打印节流：数值变化或超过该间隔才打印一次，避免每帧刷屏
COMMAND_PRINT_INTERVAL = float(_cfg("logging", "command_print_interval", default=0.5))
# “坐标无效 / 命令全0”警告打印最小间隔（秒）
WARN_INTERVAL_S = float(_cfg("logging", "warn_interval_s", default=1.0))

# ==================== 运行日志文件（config.yaml → logging.log_file） ====================
# 每次启动都追加写入该文件：控制台输出会同时写入 log.txt，
# 并在启动/退出时各写一行时间戳分隔，便于区分多次运行。
LOG_FILE = Path(_cfg("logging", "log_file", default="log.txt"))
if not LOG_FILE.is_absolute():
    LOG_FILE = Path(__file__).resolve().parent / LOG_FILE

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
# detection_area_after_first：放置/托盘阶段使用的另一个 ROI；
# None=不切换（各阶段都用 detection_area）。
DETECTION_AREA_FIRST = _cfg("detection", "detection_area", default=None)
DETECTION_AREA_AFTER_FIRST = _cfg(
    "detection", "detection_area_after_first", default=None
)

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
_last_rx_print = {"t": 0.0}
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
        data.gripper_mm,
        data.capture_ack, data.finish_capture, data.arrived,
    )
    if key == _last_rx_key:
        return
    _last_rx_key = key
    log_serial_rx(
        f"x={data.chassis_x} y={data.chassis_y} "
        f"vx={data.chassis_vx} vy={data.chassis_vy} "
        f"g={data.gripper_mm} "
        f"ack={data.capture_ack} fin={data.finish_capture} arr={data.arrived}"
    )
    # 终端打印接收数据（变化时打印，节流 0.5s 防刷屏）
    now = time.time()
    if now - _last_rx_print["t"] >= 0.5:
        _last_rx_print["t"] = now
        print(
            f"[RX] chassis=({data.chassis_x},{data.chassis_y})mm "
            f"v=({data.chassis_vx},{data.chassis_vy})mm/s "
            f"gripper={data.gripper_mm}mm "
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


def show_scan(frame):
    """缩放后显示二维码扫描画面，防止窗口过大；仅显示用，不修改原始帧。"""
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
    cv2.imshow("qr_scan", frame)


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


def log_command(tag, target, number, action, capture,
                chassis_x_mm, chassis_y_mm, gripper_mm,
                fb_x=0, fb_y=0, fb_vx=0, fb_vy=0, fb_gripper_mm=0,
                camera_coord=None, world_coord=None,
                image_center=None):
    """打印当前下发给下位机的底盘/夹爪指令，以及下位机回传的底盘数据。"""
    now = time.time()
    sig = (
        tag, target, number, action, capture,
        chassis_x_mm, chassis_y_mm, gripper_mm,
        camera_coord, world_coord, image_center,
    )
    if (
        not capture
        and sig == _last_command_print["sig"]
        and now - _last_command_print["t"] < COMMAND_PRINT_INTERVAL
    ):
        return

    _last_command_print["t"] = now
    _last_command_print["sig"] = sig
    chassis_label = "底盘目标位置"
    print(
        f"[{tag}] target={target} number={number} "
        f"action={action} capture={int(capture)} "
        f"{chassis_label}=({chassis_x_mm:+d},{chassis_y_mm:+d})mm "
        f"夹爪伸长量={gripper_mm}mm | "
        f"下位机回传=({fb_x},{fb_y})mm v=({fb_vx},{fb_vy})mm/s "
        f"gripper反馈={fb_gripper_mm}mm"
    )
    if camera_coord is not None:
        print(
            f"  相机坐标={camera_coord}cm, "
            f"车中心坐标={world_coord}cm"
        )
    if image_center is not None:
        obj_x, obj_y, center_x, center_y = image_center
        dx = obj_x - center_x
        dy = obj_y - center_y
        dist = np.hypot(dx, dy)
        print(
            f"  图像: 物块=({obj_x:.0f},{obj_y:.0f}) "
            f"图像中心=({center_x},{center_y}) "
            f"偏移=({dx:+.0f},{dy:+.0f})px 距离={dist:.0f}px"
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


def log_tray_order(round_no, cycle_no, tray_targets, label):
    """记录托盘阶段抓取顺序（数字即托盘号）。"""
    if not tray_targets:
        return
    line = (
        f"第{round_no}轮 第{cycle_no}遍 托盘阶段：{label}抓取="
        f"{','.join(map(str, tray_targets))}，"
        f"对应托盘={','.join(map(str, tray_targets))}"
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
                sent_ok = serial_comm.send(vg)
            except Exception as e:
                print(f"  串口发送失败: {e}")
                log_serial_tx(f"SEND-ERR {e}")
                sent_ok = False
            if sent_ok:
                log_serial_tx(
                    f"t={vg.target_} n={vg.number_} a={vg.action_} c={vg.capture_} "
                    f"x={vg.chassis_x_mm} y={vg.chassis_y_mm} g={vg.gripper_mm}"
                )
                if _tx_offline_logged:
                    _tx_offline_logged = False
                    print("[串口] 通信已恢复，继续发送")
            else:
                if not _tx_offline_logged:
                    _tx_offline_logged = True
                    log_serial_tx("OFFLINE")
                    print("[串口离线] 发送失败，命令已丢弃，等待自动重连")
        else:
            if not _tx_offline_logged:
                log_serial_tx("OFFLINE")
                _tx_offline_logged = True
            print("  [离线] 未发送")

    print("[发送线程] 已退出")


_QR_DISPLAY_PROC = None


def _reset_qr_display_state():
    """删除扫码结果状态文件，让仍在运行的外接屏回到空白状态。"""
    state_file = os.environ.get("QR_DISPLAY_FILE", QR_DISPLAY_STATE_FILE)
    try:
        os.unlink(state_file)
    except OSError:
        pass


def _read_log_tail(path, n=8):
    """读取文件末尾若干行，用于显示进程启动失败时的诊断输出。"""
    try:
        lines = Path(path).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        return lines[-n:]
    except OSError:
        return []


def _start_qr_display(reset_state=False):
    """自动拉起外接屏显示进程；已在运行时不重复启动。

    reset_state=True 时清空状态文件，让外接屏先保持空白
    （只在 src.py 启动时调用；扫码命中时传 False，保留刚写入的结果）。

    显示进程用独立会话（setsid）+ 忽略 SIGHUP + 输出重定向到日志启动，
    关闭本程序/终端不会关闭显示窗口。
    """
    global _QR_DISPLAY_PROC
    if not QR_DISPLAY_ENABLED:
        return
    if _QR_DISPLAY_PROC is not None and _QR_DISPLAY_PROC.poll() is None:
        return

    script = Path(__file__).resolve().parent / "qr_display.py"
    state_file = os.environ.get("QR_DISPLAY_FILE", "/tmp/qr_display_result.txt")
    if reset_state:
        _reset_qr_display_state()
    cmd = [sys.executable, str(script), "--file", state_file]
    if QR_DISPLAY_REPLACE:
        cmd.append("--replace")
    if QR_DISPLAY_MONITOR is not None:
        cmd += ["--monitor", str(int(QR_DISPLAY_MONITOR))]

    env = os.environ.copy()
    if QR_DISPLAY_DISPLAY:
        env["DISPLAY"] = QR_DISPLAY_DISPLAY
    if QR_DISPLAY_XAUTHORITY:
        env["XAUTHORITY"] = QR_DISPLAY_XAUTHORITY

    log_fd = None
    try:
        log_fd = open(QR_DISPLAY_LOG_FILE, "a", encoding="utf-8")
    except OSError as exc:
        print(f"[QR显示] 无法打开显示进程日志 {QR_DISPLAY_LOG_FILE}: {exc}")
        # 日志文件可能被旧进程/其它用户占用，退回用户自己的 /tmp 日志，
        # 避免显示进程连日志都写不了。
        try:
            fallback = Path(f"/tmp/qr_display_{os.getuid()}.log")
            log_fd = open(fallback, "a", encoding="utf-8")
        except OSError as fallback_exc:
            print(f"[QR显示] 备用日志 {fallback} 也无法打开: {fallback_exc}")

    try:
        _QR_DISPLAY_PROC = subprocess.Popen(
            cmd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fd if log_fd is not None else subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        _QR_DISPLAY_PROC = None
        print(f"[QR显示] 外接屏显示进程启动失败: {exc}")
        return
    finally:
        if log_fd is not None:
            log_fd.close()

    print(f"[QR显示] 已启动外接屏显示进程 (pid={_QR_DISPLAY_PROC.pid})")
    if QR_DISPLAY_DISPLAY:
        print(f"[QR显示] 使用 DISPLAY={QR_DISPLAY_DISPLAY}")
    if QR_DISPLAY_XAUTHORITY:
        print(f"[QR显示] 使用 XAUTHORITY={QR_DISPLAY_XAUTHORITY}")
    print(f"[QR显示] 监视扫码结果文件: {state_file}（日志: {QR_DISPLAY_LOG_FILE}）")

    # 等约 1 秒确认显示进程没有“启动即退出”（常见原因：DISPLAY 连不上、
    # 旧的单实例锁未释放、tkinter 异常），退出时把日志尾巴打出来方便排查。
    time.sleep(1.0)
    if _QR_DISPLAY_PROC.poll() is not None:
        print(
            f"[QR显示] 警告：显示进程启动后立即退出 "
            f"(pid={_QR_DISPLAY_PROC.pid}, exit={_QR_DISPLAY_PROC.returncode})"
        )
        tail = _read_log_tail(QR_DISPLAY_LOG_FILE)
        if tail:
            print("[QR显示] 显示进程日志（最后几行）：")
            for line in tail:
                print(f"  {line}")


# ==================== 主程序 ====================
def main():
    global C_1, C_2, chassis_x, chassis_y, chassis_vx, chassis_vy
    global _serial_comm

    # ==================== 第一步：先识别并打开串口 ====================
    # 串口未就绪时持续重试等待（不直接退出）；按 Ctrl+C 中止
    try:
        _serial_comm = SerialComm()
    except Exception as e:
        print(f"串口打开异常: {e}")
        _serial_comm = None

    last_wait_msg = 0.0
    while _serial_comm is None or not _serial_comm.connected_:
        if _serial_comm is None:
            try:
                _serial_comm = SerialComm()
            except Exception as e:
                print(f"串口打开异常: {e}")
                _serial_comm = None
        else:
            _serial_comm.retry_open(silent=True)

        if _serial_comm is not None and _serial_comm.connected_:
            break

        now = time.time()
        if now - last_wait_msg >= 5.0:
            last_wait_msg = now
            port_hint = _serial_comm.port if _serial_comm else "/dev/ttyACM0"
            print(
                "[串口] 尚未连接下位机串口，每 3 秒重试"
                f"（当前目标 {port_hint}；请检查 USB/重新上电，Ctrl+C 退出）..."
            )
        try:
            time.sleep(3.0)
        except KeyboardInterrupt:
            print("用户终止：未打开串口，退出")
            if _serial_comm:
                _serial_comm.close()
            return

    print(f"串口识别并打开成功：{_serial_comm.port} @ {_serial_comm.baudrate}")
    _serial_comm.start_chassis_recv()
    print("底盘接收线程已启动")

    # ==================== 第二步：打开 USB 摄像头 ====================
    # 两台 USB 免驱摄像头：cap 用于二维码扫描，detection_cap 用于物块检测/放置
    cap = open_camera(
        QR_CAMERA_SOURCE,
        width=DETECTION_FRAME_WIDTH,
        height=DETECTION_FRAME_HEIGHT,
    )
    if cap is None:
        _serial_comm.stop_chassis_recv()
        _serial_comm.close()
        return

    detection_cap = open_camera(
        DETECTION_CAMERA_SOURCE,
        width=DETECTION_FRAME_WIDTH,
        height=DETECTION_FRAME_HEIGHT,
        fps=DETECTION_CAMERA_FPS,
    )
    if detection_cap is None:
        print("未检测到物块检测 USB 摄像头，退出")
        if cap:
            cap.release()
        _serial_comm.stop_chassis_recv()
        _serial_comm.close()
        return

    q = queue.Queue(maxsize=1)

    if CHASSIS_LOOKAHEAD_S > 0.0:
        print(
            f"[配置] 底盘前瞻已开启：{CHASSIS_LOOKAHEAD_S * 1000.0:.0f}ms"
        )

    sending_thread = threading.Thread(target=Sending2Gimbal, args=(q, _serial_comm), daemon=True)
    sending_thread.start()

    # 项目启动：先向下位机发送 action=0（启动/空闲）信号
    vg = VisionToGimbal(target=0, action=IDLE_ACTION)
    enqueue_latest(q, vg)
    print("已发送启动信号 action=0，等待识别二维码...")
    _start_qr_display(reset_state=True)

    # ---- 状态 ----
    last_sessions = None
    detector = get_detector()
    detector.reset()
    # 检测/稳定性参数统一从 config.yaml 读取（felling_color.BlockDetector 默认值）

    # 卡尔曼滤波追踪器（替代神经网络预测），参数从 config.yaml 的 kalman 段读取
    kf = KalmanBlockTracker()

    # 世界系卡尔曼追踪器（filter.type=kalman_world 时使用，状态为车中心系 mm）
    kwf = KalmanWorldTracker()

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

    detection_sent = False
    waiting_for_next = False    # 已请求抓取，等待下位机 finish_capture 或手动切换
    all_done = False            # 所有目标已完成，退出主循环
    rounds = []                 # [{"grab": [...], "place": [...]}, ...]
    round_digit_of_color = []   # 每轮 "颜色代码→圆环数字" 映射（供下一轮遮挡反推）
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
    tray_pending_slot = None    # 当前托盘对应物块的放置目标槽位（下位机 target 用）
    tray_move_delay_until = None  # 进入托盘阶段后，延时发送首条移动指令的到期时间
    tray_move_delay_vg = None     # 延时待发的“前往第一个托盘”移动指令
    place_stable_count = 0      # 放置阶段位置稳定计数（连续同圆环且位移小）
    place_last_center = None    # 放置阶段上一帧平滑后的圆环中心
    place_last_digit = None     # 放置阶段上一帧选中圆环数字
    place_smooth_center = None  # 放置阶段圆心 EMA 平滑状态
    placement_recognizer = None
    last_place_dbg = 0.0          # 放置发送调试日志节流
    place_prealign_active = False # 放置区“先用可见圆环调整车的位置”阶段是否激活
    last_worldkf_print = 0.0      # 世界系卡尔曼终端调试打印节流
    last_status_print = 0.0       # 底盘/夹爪状态行打印节流
    serial_offline_since = None   # 串口连续断开起始时间（None=当前在线）
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
    gripper_cm_meas = 0.0            # 测量用夹爪位置（低通，防止指令跳变灌进卡尔曼）
    gripper_settle_target_mm = None  # 当前等待到位的固定夹爪目标（mm），None=不需要等待
    gripper_settle_started = None    # 开始等待夹爪到位的时间戳
    gripper_settle_done = True       # 夹爪是否已到位（或超时）
    last_kf_time = None              # 上一帧 KF 更新时间（用于按实际帧间隔更新 dt）
    # 最近一次有效的底盘/夹爪指令；pixel_to_camera 失效时沿用，避免误发 0
    last_valid_mm = (0, 0, 0)
    # 抓取区“开始识别时第一目标已在场”跳过逻辑状态
    grab_skip_first_checked = False    # 本次抓取区“第一目标已在场”判定是否已完成
    grab_skip_first_active = False     # 已判定第一目标在场，本圈跳过，等下一圈
    grab_skip_first_miss_seen = False  # 跳过期间是否已看到目标真正离开画面
    grab_skip_first_fake_miss = False  # 本帧目标可见但被有意忽略（不是真消失）
    grab_skip_first_eval_left = 0      # 开始识别后的判定窗口剩余帧数

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
        nonlocal grab_skip_first_checked, grab_skip_first_active
        nonlocal grab_skip_first_miss_seen, grab_skip_first_fake_miss
        nonlocal grab_skip_first_eval_left
        waiting_for_next = False
        detector.reset()
        kf.reset()
        kwf.reset()
        one_euro_tracker.reset()
        detection_sent = False
        sent_time = None
        capture_ack_received = False
        capture_last_sent_time = None
        last_capture_vg = None
        last_tracking_send = 0.0
        last_sent_tracking_mm = None
        last_smooth_cmd_mm = (chassis_x, chassis_y, last_smooth_cmd_mm[2])
        last_smooth_time = time.time()
        last_kf_time = None
        last_detected_center = None
        last_valid_mm = (0, 0, 0)
        grab_skip_first_checked = False
        grab_skip_first_active = False
        grab_skip_first_miss_seen = False
        grab_skip_first_fake_miss = False
        grab_skip_first_eval_left = 0
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
        nonlocal place_smooth_center
        nonlocal gripper_settle_target_mm, gripper_settle_started, gripper_settle_done
        nonlocal grab_skip_first_checked, grab_skip_first_active
        nonlocal grab_skip_first_miss_seen, grab_skip_first_fake_miss
        nonlocal grab_skip_first_eval_left
        detection_sent = False
        waiting_for_next = False
        sent_time = None
        capture_ack_received = False
        capture_last_sent_time = None
        last_capture_vg = None
        last_tracking_send = 0.0
        last_sent_tracking_mm = None
        last_smooth_cmd_mm = (chassis_x, chassis_y, last_smooth_cmd_mm[2])
        last_smooth_time = time.time()
        last_kf_time = None
        last_valid_mm = (0, 0, 0)
        last_detected_center = None
        last_detection_time = None
        # 每个新目标都重新等夹爪到位，避免沿用上一个目标已到位的状态
        gripper_settle_target_mm = None
        gripper_settle_started = None
        gripper_settle_done = True
        recognition_started = False
        place_stable_count = 0
        place_last_center = None
        place_smooth_center = None
        place_last_digit = None
        grab_skip_first_checked = False
        grab_skip_first_active = False
        grab_skip_first_miss_seen = False
        grab_skip_first_fake_miss = False
        grab_skip_first_eval_left = 0
        detector.reset()
        kf.reset()
        kwf.reset()
        one_euro_tracker.reset()

    def advance_after_placement_cycle(already_at_next_area=False):
        """一次放置完成后推进流程：
        - 第 1 次放置（托盘阶段已把物块夹回后部槽位）：直接前往下一个放置区；
        - 第 2 次（本轮最后一次）放置：进下一轮或结束。

        already_at_next_area=True 时（托盘阶段结束收到的 arrived=1 已经是
        “到达下一放置区”的信号），不再重发 action=2、不再等新的上升沿，
        直接进入放置识别。

        返回 "done" 表示所有轮次完成，否则流程已推进。
        """
        nonlocal round_cycles_done, current_round, target_colors, target_index
        nonlocal placed_digits, placed_order, place_waiting_arrived
        nonlocal waiting_for_arrive, phase, phase_after_arrival, prev_arrived
        nonlocal placed_block_slots, slot_of_place_digit
        nonlocal last_placed_digit, placement_recognizer
        nonlocal place_prealign_active

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
            # 本轮第 1 次放置完成：物块已由托盘阶段按抓取颜色顺序夹回后部槽位，
            # 不再回抓取区，直接前往下一个放置区再次全部放置。
            placed_block_slots.clear()
            place_waiting_arrived = False
            if already_at_next_area:
                # 临时绕过：下位机在托盘阶段结束后已经自行驶到下一个放置区，
                # 并且刚回 arrived=1。这个 1 直接当作“已到达放置区 B”的信号，
                # 不再重发 action=2（否则下位机可能再走一段且 arrived 不掉 0，
                # 上位机永远等不到新的 0→1 上升沿）。
                print(f"第 {current_round + 1} 轮第 {round_cycles_done} 次放置完成，"
                      f"已到达下一个放置区（物块已夹回后部槽位）")
                waiting_for_arrive = False
                phase = PLACE_ACTION
                phase_after_arrival = PLACE_ACTION
                prev_arrived = 1
                reset_action_state()
                last_placed_digit = None
                place_prealign_active = PLACE_PREALIGN_ENABLED
                slot_of_place_digit = {
                    int(d): i + 1
                    for i, d in enumerate(rounds[current_round]["place"])
                }
                if placement_recognizer is None:
                    placement_recognizer = PlacementRecognizer()
                print(
                    "已到达放置区，开始识别物料（圆环数字）"
                    + (
                        "；先拿最近可见圆环调整车的位置"
                        if place_prealign_active else ""
                    )
                )
            else:
                print(f"第 {current_round + 1} 轮第 {round_cycles_done} 次放置完成，"
                      f"前往下一个放置区（物块已夹回后部槽位）")
                vg = VisionToGimbal(target=0, action=PLACE_ACTION)
                enqueue_latest(q, vg)
                waiting_for_arrive = True
                phase_after_arrival = PLACE_ACTION
                prev_arrived = 0
                reset_action_state()
            return "place"

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
            print(f"第 {current_round} 轮放置完成（共 {repeat} 个放置区），"
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
        把绝对目标指令平滑成连续小步：
        x/y 是绝对目标位置（mm），按 ramp 限制每帧变化量；
        夹爪为绝对伸长量直发，不做增量比例/ramp。
        """
        nonlocal last_smooth_cmd_mm, last_smooth_time
        x, y, gripper = cmd_mm
        # 绝对目标模式：x/y 就是目标位置（mm），只做 ramp 平滑
        dx = int(round(x))
        dy = int(round(y))

        # ramp 按发送周期标定，这里按实际帧间隔缩放
        # 恢复/断帧后不允许一次性大步补回来：dt 最多按一个发送周期算，
        # 配合断帧时重置平滑状态，避免“停一下然后猛冲一下”。
        smooth_dt_cap = (
            TRACKING_SEND_INTERVAL if TRACKING_SEND_INTERVAL > 0 else 0.5
        )
        dt = min(max(now - last_smooth_time, 0.0), smooth_dt_cap)
        if TRACKING_SEND_INTERVAL > 0:
            ramp = CHASSIS_RAMP_STEP_MM * (dt / TRACKING_SEND_INTERVAL)
        else:
            ramp = CHASSIS_RAMP_STEP_MM
        ramp = max(ramp, 0.5)

        lx, ly, lz = last_smooth_cmd_mm
        nx = lx + max(-ramp, min(ramp, dx - lx))
        ny = ly + max(-ramp, min(ramp, dy - ly))
        # 夹爪：绝对伸长量协议（0=最短位置，可伸可缩）。
        # 目标来自视觉换算的物块距离（世界系），直接作为绝对目标下发，
        # 由下位机自己闭环到位，不再做增量比例/ramp 累加。
        gripper = max(0.0, float(gripper))
        gz = int(round(min(gripper, MAX_GRIPPER_MM)))

        last_smooth_cmd_mm = (int(round(nx)), int(round(ny)), gz)
        last_smooth_time = now
        return last_smooth_cmd_mm

    def update_gripper_settle(desired_mm, now, feedback_valid, fb_mm):
        """
        固定夹爪抓取前先等夹爪反馈到位，再允许底盘跟踪。
        返回 True=已到位/不需要等待/超时；False=仍在等夹爪。
        """
        nonlocal gripper_settle_target_mm, gripper_settle_started, gripper_settle_done
        if desired_mm is None:
            gripper_settle_target_mm = None
            gripper_settle_started = None
            gripper_settle_done = True
            return True

        if (
            feedback_valid
            and abs(fb_mm - desired_mm) <= GRIPPER_SETTLE_TOLERANCE_MM
        ):
            gripper_settle_target_mm = desired_mm
            gripper_settle_started = now
            gripper_settle_done = True
            return True

        if gripper_settle_target_mm != desired_mm:
            gripper_settle_target_mm = desired_mm
            gripper_settle_started = now
            gripper_settle_done = False
            print(f"[夹爪] 等待夹爪到位 {desired_mm}mm，先不动底盘...")

        if gripper_settle_done:
            return True
        if not feedback_valid:
            # 没有串口反馈时无法判断，直接允许跟踪，避免离线/异常时卡住
            gripper_settle_done = True
            return True
        if now - gripper_settle_started >= GRIPPER_SETTLE_TIMEOUT_S:
            gripper_settle_done = True
            print(f"[夹爪] 等待到位超时（{GRIPPER_SETTLE_TIMEOUT_S:.1f}s），继续底盘跟踪")
            return True
        return False

    def tracking_send_allowed(capture, cmd_mm, deadband_mm=None):
        """
        普通 capture=0 包按“最小间隔 + 变化死区 + 心跳”节流；
        capture=1 立即发送。
        """
        nonlocal last_tracking_send, last_sent_tracking_mm
        now = time.time()
        if deadband_mm is None:
            deadband_mm = CHASSIS_SEND_DEADBAND_MM
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
            abs(cmd_mm[0] - last_sent_tracking_mm[0]) >= deadband_mm
            or abs(cmd_mm[1] - last_sent_tracking_mm[1]) >= deadband_mm
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

            # ============ 串口断开检测：断开超过阈值即退出程序 ============
            # 拔掉串口后进程正常退出，systemd 会自动重新拉起并等待；
            # 重新插上串口后程序自动从头开始运行。
            if _serial_comm is not None and not _serial_comm.connected_:
                now = time.time()
                if serial_offline_since is None:
                    serial_offline_since = now
                    print("[串口离线] 检测到串口断开，等待自动重连…")
                elif (
                    now - serial_offline_since
                    >= SERIAL_DISCONNECT_EXIT_DELAY_S
                ):
                    print(
                        f"[串口离线] 已连续断开 "
                        f"{now - serial_offline_since:.1f}s，程序退出；"
                        "重新插上串口后会自动启动"
                    )
                    break
            else:
                serial_offline_since = None

            # ============ QR 扫描阶段 ============
            if len(scan_QRcode_andlist.session) == 0:
                Ostu_image = ostu_threshold(frame)
                scan_QRcode_andlist.scan_qrcode(Ostu_image, frame)
                # 显示扫码摄像头实时画面（识别到二维码时画面里会带绿色边框和文字）
                show_scan(frame)
                sessions = scan_QRcode_andlist.session

                if sessions and sessions != last_sessions:
                    last_sessions = sessions
                    print(f"QR识别结果: {sessions}")
                    _start_qr_display()

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

                    # 每轮按“槽位”关系建立 颜色→圆环数字 映射：
                    # 第 n 轮放在圆环数字 d 上的物块颜色 = 槽位与 d 相同的抓取颜色。
                    round_digit_of_color = []
                    for rnd in rounds:
                        r_slot_of_color = {
                            c: i + 1 for i, c in enumerate(rnd["grab"])
                        }
                        r_slot_of_digit = {
                            int(d): i + 1 for i, d in enumerate(rnd["place"])
                        }
                        r_digit_of_color = {}
                        for r_digit, r_slot in r_slot_of_digit.items():
                            for r_color, r_cslot in r_slot_of_color.items():
                                if r_cslot == r_slot:
                                    r_digit_of_color[r_color] = r_digit
                        round_digit_of_color.append(r_digit_of_color)

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
                    place_smooth_center = None
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
                    # 扫到二维码：关闭扫码画面窗口，切换到物块检测画面
                    try:
                        cv2.destroyWindow("qr_scan")
                    except cv2.error:
                        pass
                    print("识别到QR，关闭二维码USB摄像头")
                    detector.reset()
                    kf.reset()
                    kwf.reset()
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

                # 每秒固定打印一次底盘/夹爪当前位置（不管值有没有变化）
                if chassis_data2 is not None and time.time() - last_status_print >= 1.0:
                    last_status_print = time.time()
                    print(
                        f"[状态] 底盘=({chassis_x},{chassis_y})mm "
                        f"gripper={chassis_data2.gripper_mm}mm "
                        f"ack={chassis_data2.capture_ack} fin={chassis_data2.finish_capture} "
                        f"arr={chassis_data2.arrived}"
                    )

                # 摄像头装在夹爪/云台上，会随夹爪一起伸缩；
                # 换算“物块相对车中心”时需加上当前夹爪伸长量。
                # 下位机回传夹爪绝对位置时优先用反馈；未回传时退回最近一次已下发指令。
                fb_gripper_mm = (
                    chassis_data2.gripper_mm
                    if chassis_data2 is not None
                    else (last_sent_tracking_mm[2] if last_sent_tracking_mm is not None else 0)
                )
                current_gripper_cm = fb_gripper_mm / 10.0

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
                        # 完成判定：优先认 0→1 上升沿；
                        # 若上升沿在“未等待”期间已被吞掉（fin 提前变 1 并保持），
                        # 则用“ack 已确认 + fin==1”电平兜底，避免上下位机互相等死。
                        if chassis_data2.finish_capture == 1 and not prev_finish_capture:
                            finish_rising = True
                        elif capture_ack_received and chassis_data2.finish_capture == 1:
                            finish_rising = True
                            print("[下位机] fin 电平兜底触发"
                                  "（ack 已确认且 fin=1，未等到 0→1 上升沿）")
                        prev_finish_capture = 1 if chassis_data2.finish_capture else 0
                    else:
                        # 未等待时同步当前电平，避免旧电平在下次造成误触发
                        if chassis_data2.finish_capture == 1 and not prev_finish_capture:
                            print("[下位机] fin=1 上升沿但当前未在等待，"
                                  "已同步电平（本次不触发完成）")
                        prev_finish_capture = 1 if chassis_data2.finish_capture else 0

                # 下位机已到达指定区域（抓取区/放置区）
                if waiting_for_arrive and arrived_rising:
                    phase = phase_after_arrival
                    waiting_for_arrive = False
                    reset_action_state()
                    target_index = 0
                    if phase == PLACE_ACTION:
                        place_waiting_arrived = False
                        place_prealign_active = PLACE_PREALIGN_ENABLED
                        placed_digits = set()
                        placed_order = []
                        last_placed_digit = None
                        slot_of_place_digit = {
                            int(d): i + 1
                            for i, d in enumerate(rounds[current_round]["place"])
                        }
                        if placement_recognizer is None:
                            placement_recognizer = PlacementRecognizer()
                        print(
                            "已到达放置区，开始识别物料（圆环数字）"
                            + (
                                "；先拿最近可见圆环调整车的位置"
                                if place_prealign_active else ""
                            )
                        )
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
                        release_target = tray_pending_slot
                        release_action = TRAY_PHASE_ACTION
                        release_number = tray_pending_digit
                    elif phase == GRAB_ACTION:
                        release_target = last_grabbed_slot
                        release_action = GRAB_ACTION
                        release_number = 0
                    else:
                        # 放置完成后补发的 capture=0 里带“下一个应放数字”，
                        # 让下位机知道要移动到下一个圆环（例如放完 1 → number=3）。
                        remaining_digits = [
                            int(d) for d in rounds[current_round]["place"]
                            if int(d) not in placed_digits
                            and int(d) != last_placed_digit
                        ]
                        next_place_digit = (
                            remaining_digits[0] if remaining_digits else None
                        )
                        if next_place_digit is not None:
                            release_target = slot_of_place_digit.get(
                                next_place_digit, 0
                            )
                            release_number = next_place_digit
                        else:
                            release_target = (
                                slot_of_place_digit.get(last_placed_digit, 0)
                                if last_placed_digit is not None else 0
                            )
                            release_number = last_placed_digit or 0
                        release_action = PLACE_ACTION
                    enqueue_latest(q, VisionToGimbal(
                        target=release_target if release_target is not None else 0,
                        number=release_number,
                        action=release_action,
                        capture=False,
                    ))
                    print(f"[动作完成] 已补发 capture=0 "
                          f"(target={release_target}, number={release_number}, "
                          f"action={release_action})")

                    if tray_phase_active and tray_pending_digit is not None:
                        tray_pending_digit = None
                        tray_pending_slot = None
                        grabbed_slots = set()
                        last_grabbed_slot = None
                        reset_action_state()
                        if TRAY_PHASE_WAIT_ARRIVED:
                            # actual 顺序下每抓完一个托盘，发下一个托盘的移动指令
                            # （number=圆环数字, target=槽位），让下位机前往下一个
                            # 托盘；并吞掉遗留 arrived=1，等新的 0→1 再抓。
                            if tray_plan and not TRAY_PHASE_IS_REVERSE:
                                _next_digit, _ = tray_plan[0]
                                _next_slot = slot_of_place_digit.get(
                                    _next_digit, _next_digit
                                )
                                # q.put 保证在“补发 capture=0”之后按序发送
                                q.put(VisionToGimbal(
                                    target=_next_slot,
                                    number=_next_digit,
                                    action=TRAY_PHASE_ACTION,
                                    capture=False,
                                ))
                                print(
                                    f"[托盘] 已发送前往下一托盘 {_next_digit} "
                                    f"的移动指令 (target={_next_slot}, "
                                    f"number={_next_digit})"
                                )
                                prev_arrived = 1
                            print("托盘抓取完成，已补发 capture=0，"
                                  "等待下位机前往下一托盘（arrived=1）")
                        else:
                            print("托盘抓取完成，已补发 capture=0，"
                                  "立即开始下一托盘")
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
                                    if TRAY_PHASE_IS_REVERSE
                                    else "实际顺序（第一组抓取顺序）"
                                )
                                # actual = 按第一组物块抓取顺序（颜色顺序）对应的
                                # 放置圆环数字；reverse = 该顺序的倒序。
                                # 例如抓取颜色 [3,4,5]、放置序列 [1,3,2] 时，
                                # actual 托盘顺序 = [1,3,2]。
                                _grab_colors = list(rounds[current_round]["grab"])
                                _place_digits = list(rounds[current_round]["place"])
                                _grab_slot_of_color = {
                                    c: i + 1 for i, c in enumerate(_grab_colors)
                                }
                                tray_targets = []
                                for _color in _grab_colors:
                                    _g_slot = _grab_slot_of_color.get(_color)
                                    if (
                                        _g_slot is not None
                                        and 1 <= _g_slot <= len(_place_digits)
                                    ):
                                        tray_targets.append(
                                            int(_place_digits[_g_slot - 1])
                                        )
                                if TRAY_PHASE_IS_REVERSE:
                                    tray_targets.reverse()
                                # 进入阶段时一次算好每个托盘对应的物块颜色，
                                # 避免后面 slot_of_color 被单托盘映射覆盖后找不到颜色
                                tray_plan = []
                                for _tray_digit in tray_targets:
                                    _place_slot = slot_of_place_digit.get(_tray_digit)
                                    _tray_color = None
                                    if _place_slot is not None:
                                        _tray_color = next(
                                            (c for c, s in _grab_slot_of_color.items()
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
                                    tray_targets,
                                    tray_order_label,
                                )
                                # 进入托盘阶段：
                                # 倒序时第一个托盘就是刚放置的位置，放置完成后的
                                # arrived=1 可直接作为第一个托盘的到达信号。
                                # actual 顺序下第一个托盘不是当前位置：
                                # 先发移动指令（number=圆环数字, target=槽位）
                                # 让下位机前往第一个托盘，并吞掉遗留 arrived=1，
                                # 等它到达后新的 0→1 再开始抓取。
                                prev_arrived = 0
                                if (
                                    tray_plan
                                    and TRAY_PHASE_WAIT_ARRIVED
                                    and not TRAY_PHASE_IS_REVERSE
                                ):
                                    _first_digit, _ = tray_plan[0]
                                    _first_slot = slot_of_place_digit.get(
                                        _first_digit, _first_digit
                                    )
                                    tray_move_delay_vg = VisionToGimbal(
                                        target=_first_slot,
                                        number=_first_digit,
                                        action=TRAY_PHASE_ACTION,
                                        capture=False,
                                    )
                                    if TRAY_PHASE_ENTRY_DELAY_S > 0:
                                        tray_move_delay_until = (
                                            time.time() + TRAY_PHASE_ENTRY_DELAY_S
                                        )
                                        print(
                                            f"[托盘] 放置完成，延时 "
                                            f"{TRAY_PHASE_ENTRY_DELAY_S:.1f}s 后发送"
                                            f"前往第一个托盘 {_first_digit} 的移动指令"
                                            f" (target={_first_slot}, "
                                            f"number={_first_digit})"
                                        )
                                    else:
                                        # 用 q.put 保证在“补发 capture=0”之后按序发送
                                        q.put(tray_move_delay_vg)
                                        tray_move_delay_vg = None
                                        print(
                                            f"[托盘] 已发送前往第一个托盘 {_first_digit} "
                                            f"的移动指令 (target={_first_slot}, "
                                            f"number={_first_digit})"
                                        )
                                    prev_arrived = 1
                                reset_action_state()
                                tray_pending_digit = None
                                arrival_wait_txt = (
                                    "等待下位机 arrived"
                                    if TRAY_PHASE_WAIT_ARRIVED
                                    else "不等待 arrived，立即开始抓取"
                                )
                                print(f"第 {current_round + 1} 轮第 {round_cycles_done + 1} 次"
                                      f"放置完成，进入托盘阶段（{tray_order_label}抓取）: "
                                      f"{tray_targets}，"
                                      f"{arrival_wait_txt}")
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

                # 下位机还没回 arrived=1（0→1 到达信号）时：
                # 不识别、不发送跟踪/抓取指令，避免车还没到位就提前动作。
                # 放置阶段原本有自己的拦截，这里统一补上抓取等阶段的拦截。
                if waiting_for_arrive:
                    if frame is not None:
                        cv2.putText(frame, "Waiting for arrived=1 ...",
                                    (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 255), 2)
                    show_detection(frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # ==================== 托盘阶段（按配置顺序抓取托盘上的物块） ====================
                # 复用抓取阶段的视觉跟踪逻辑：按托盘对应物块颜色识别、
                # 动态调夹爪，位置稳定且对准后才发 capture=1。
                if tray_phase_active:
                    # 进入托盘阶段的延时等待：放置完最后一个物块后不立即发
                    # “前往第一个托盘”的移动指令，等延时结束再发送。
                    # 延时期间持续吞掉当前 arrived=1，避免被误当成“已到达托盘”。
                    if tray_move_delay_until is not None:
                        if time.time() < tray_move_delay_until:
                            prev_arrived = 1
                            if frame is not None:
                                cv2.putText(
                                    frame,
                                    f"Tray phase: sending move in "
                                    f"{tray_move_delay_until - time.time():.1f}s",
                                    (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 255), 2,
                                )
                                show_detection(frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                            continue
                        if tray_move_delay_vg is not None:
                            q.put(tray_move_delay_vg)
                            print(
                                f"[托盘] 延时结束，已发送前往第一个托盘 "
                                f"{tray_move_delay_vg.number_} 的移动指令"
                                f" (target={tray_move_delay_vg.target_}, "
                                f"number={tray_move_delay_vg.number_})"
                            )
                        tray_move_delay_vg = None
                        tray_move_delay_until = None
                        prev_arrived = 1
                    if arrived_rising:
                        if tray_pending_digit is not None:
                            # 上一托盘还没抓完：这个 arrived 上升沿只是干扰，
                            # 必须直接跳过本帧，否则会掉进下面“所有托盘已抓完”
                            # 的分支，把托盘阶段提前结束（表现为目标颜色突然跳变）。
                            continue
                        elif tray_plan:
                            tray_digit, tray_color = tray_plan.pop(0)
                            tray_pending_digit = tray_digit
                            tray_pending_slot = slot_of_place_digit.get(
                                tray_digit, tray_digit
                            )
                            target_colors = [tray_color]
                            target_index = 0
                            slot_of_color = {tray_color: tray_pending_slot}
                            grabbed_slots = set()
                            last_grabbed_slot = None
                            phase = GRAB_ACTION
                            prev_arrived = 1
                            reset_action_state()
                            remaining = [
                                f"{d}:{c}" for d, c in tray_plan
                            ]
                            print(f"[托盘] 当前抓取对象: 托盘{tray_digit}，"
                                  f"物块颜色={tray_color}，目标槽位={tray_pending_slot}，"
                                  f"剩余计划: {remaining}")
                            append_placement_record(
                                f"第{current_round + 1}轮 "
                                f"第{round_cycles_done + 1}遍 托盘阶段 "
                                f"当前抓取对象: 托盘{tray_digit} 颜色={tray_color} "
                                f"目标槽位={tray_pending_slot}"
                            )
                            continue

                        # 所有托盘已抓完，收到 arrived 说明最后一个托盘抓取已结束
                        tray_phase_active = False
                        tray_targets = []
                        tray_plan = []
                        tray_pending_digit = None
                        tray_pending_slot = None
                        tray_move_delay_until = None
                        tray_move_delay_vg = None
                        print("托盘阶段完成，前往下一个放置区/下一轮")
                        # 临时绕过：这个 arrived 0→1 直接当作“已到达下一个放置区”，
                        # advance 不再重发 action=2、不再等新的上升沿。
                        # 若流程实际是进下一轮抓取区（already_at_next_area 不生效），
                        # advance 仍会发 action=1 并等新的 0→1，这里再置 1 吞掉旧电平。
                        result = advance_after_placement_cycle(
                            already_at_next_area=True
                        )
                        prev_arrived = 1
                        if result == "done":
                            all_done = True
                            break
                        continue

                    if tray_pending_digit is None:
                        if not TRAY_PHASE_WAIT_ARRIVED:
                            if tray_plan:
                                tray_digit, tray_color = tray_plan.pop(0)
                                tray_pending_digit = tray_digit
                                tray_pending_slot = slot_of_place_digit.get(
                                    tray_digit, tray_digit
                                )
                                target_colors = [tray_color]
                                target_index = 0
                                slot_of_color = {tray_color: tray_pending_slot}
                                grabbed_slots = set()
                                last_grabbed_slot = None
                                phase = GRAB_ACTION
                                prev_arrived = 1
                                reset_action_state()
                                remaining = [
                                    f"{d}:{c}" for d, c in tray_plan
                                ]
                                print(f"[托盘] 当前抓取对象: 托盘{tray_digit}，"
                                      f"物块颜色={tray_color}，目标槽位={tray_pending_slot}，"
                                      f"剩余计划: {remaining}")
                                append_placement_record(
                                    f"第{current_round + 1}轮 "
                                    f"第{round_cycles_done + 1}遍 托盘阶段 "
                                    f"当前抓取对象: 托盘{tray_digit} 颜色={tray_color} "
                                    f"目标槽位={tray_pending_slot}"
                                )
                                continue

                            # 所有托盘已抓完：立即结束托盘阶段，前往下一个放置区/下一轮。
                            # 先吞掉遗留 arrived=1，避免被当成“已到达下一区域”。
                            tray_phase_active = False
                            tray_targets = []
                            tray_plan = []
                            tray_pending_digit = None
                            tray_pending_slot = None
                            tray_move_delay_until = None
                            tray_move_delay_vg = None
                            prev_arrived = 1
                            print("托盘阶段完成，前往下一个放置区/下一轮")
                            result = advance_after_placement_cycle()
                            # advance 内部会把 prev_arrived 置 0；
                            # 这里再置 1 吞掉尚未掉低的 arrived，
                            # 等下位机真正驶向下一区域（arr 掉 0）再到位（arr 升 1）触发。
                            prev_arrived = 1
                            if result == "done":
                                all_done = True
                                break
                            continue

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

                    # 抓取区阶段（每轮所有物块抓取）用 detection_area；
                    # 放置/托盘阶段（含前往放置区的路上）用 detection_area_after_first。
                    at_placement_area = (
                        tray_phase_active
                        or phase == PLACE_ACTION
                        or (
                            waiting_for_arrive
                            and phase_after_arrival == PLACE_ACTION
                        )
                    )
                    if at_placement_area and DETECTION_AREA_AFTER_FIRST is not None:
                        active_roi = DETECTION_AREA_AFTER_FIRST
                        roi_label = "ROI2"
                    else:
                        active_roi = DETECTION_AREA_FIRST
                        roi_label = "ROI"
                    if active_roi is not None:
                        active_roi = clamp_roi(active_roi, frame.shape)
                        detector.detection_area = active_roi
                        rx, ry, rw, rh = active_roi
                        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh),
                                      (0, 255, 0), 2)
                        cv2.putText(frame, roi_label, (rx + 5, ry + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        detector.detection_area = None

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
                            # 重置底盘指令平滑/发送状态，避免沿用上一个槽位的
                            # 旧目标（last_smooth_cmd_mm），防止恢复跟踪时车被
                            # 斜坡指令从旧位置冲到新位置（表现为突然向前冲）。
                            reset_action_state()

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
                                if TRAY_PHASE_IS_REVERSE
                                else ",".join(map(str, placed_order))
                            )
                        else:
                            tray_order = "-"
                        cv2.putText(frame,
                                    f"Exp: {expected_order}  Got: {got_order}  Tray: {tray_order}",
                                    (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 255), 2)

                        # 与抓取阶段一致：只识别检测 ROI 内的圆环。
                        # 圆环数字被上一轮留在放置区的物块遮挡时
                        # （第 2 轮放置区 B），按圆环上物块颜色反推数字。
                        prev_digit_of_color = (
                            round_digit_of_color[current_round - 1]
                            if current_round > 0
                            and current_round - 1 < len(round_digit_of_color)
                            else None
                        )
                        all_rings = placement_recognizer.recognize_all(
                            frame, roi=detector.detection_area,
                            digit_of_color=prev_digit_of_color,
                        )

                        # 所有检测到的圆环都画到主画面上，方便看识别情况
                        for ring in all_rings:
                            rx, ry = int(ring["center"][0]), int(ring["center"][1])
                            rr = int(ring["radius"])
                            cv2.circle(frame, (rx, ry), rr, (0, 255, 0), 2)
                            cv2.circle(frame, (rx, ry), 3, (0, 255, 0), -1)
                            label = str(ring["digit"]) if ring["digit"] is not None else "?"
                            conf_txt = (f"{ring['confidence']:.2f}"
                                        if ring["digit"] is not None else "")
                            if ring.get("inferred_by_color"):
                                conf_txt += f" C{ring.get('color_code', '?')}"
                            cv2.putText(frame, f"{label} {conf_txt}",
                                        (rx - 30, ry - rr - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 0), 1)

                        # 放置顺序按二维码放置序列执行（如 132）：
                        # 到达放置区后先进入预对准，拿最近可见圆环（如数字 2）
                        # 调车，只发 capture=0、不发 capture=1；对准完成后，
                        # 若它不是应放数字，就发应放数字（如 1）让下位机前往
                        # 对应圆环放置；若它就是应放数字，则直接进入放置。
                        # 预对准结束后只认当前应放数字，避免 number/target 为 0。
                        next_place_digit = next(
                            (int(d) for d in rounds[current_round]["place"]
                             if int(d) not in placed_digits),
                            None,
                        )
                        placed_set = set(placed_digits)
                        preferred = [
                            r for r in all_rings
                            if r["digit"] is not None
                            and r["digit"] not in placed_set
                            and r["digit"] in slot_of_place_digit
                            and (next_place_digit is None
                                 or r["digit"] == next_place_digit)
                        ]
                        fallback = [
                            r for r in all_rings
                            if r["digit"] is not None
                            and r["digit"] not in placed_set
                            and r["digit"] in slot_of_place_digit
                        ]
                        if place_prealign_active and fallback:
                            # 预对准：先拿最近可见圆环（如数字 2）调车，不放置；
                            # 若最近可见的正好是应放数字，对准后直接进入放置
                            candidates = fallback
                        elif place_prealign_active:
                            candidates = []
                        else:
                            # 预对准结束后只认当前应放数字，不越序放置
                            candidates = preferred
                        using_prealign = place_prealign_active and bool(candidates)

                        if not candidates:
                            now = time.time()
                            if all_rings and now - last_place_dbg >= 1.0:
                                last_place_dbg = now
                                digits_seen = sorted(
                                    {r["digit"] for r in all_rings
                                     if r["digit"] is not None}
                                )
                                print(
                                    f"[放置] 下一个应放数字={next_place_digit}，"
                                    f"识别到数字 {digits_seen}，"
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

                        if using_prealign and time.time() - last_place_dbg >= 1.0:
                            last_place_dbg = time.time()
                            digits_seen = sorted(
                                {r["digit"] for r in candidates
                                 if r["digit"] is not None}
                            )
                            print(
                                f"[放置] 当前应放数字={next_place_digit} 不可见，"
                                f"先用可见圆环 {digits_seen} 调整车的位置（暂不放置）"
                            )

                        # 候选优先是“当前应放数字”；多个同名圆环（异常情况）
                        # 取离图像中心最近的一个；若上一帧已锁定同一数字的圆环，
                        # 则优先沿用其圆心附近的候选，
                        # 避免识别在相邻圆环/边缘残缺圆环之间来回切换（圆心跳边）。
                        if (
                            place_last_center is not None
                            and place_last_digit is not None
                            and any(r["digit"] == place_last_digit for r in candidates)
                        ):
                            target = min(
                                candidates,
                                key=lambda r: (
                                    (r["center"][0] - place_last_center[0]) ** 2
                                    + (r["center"][1] - place_last_center[1]) ** 2
                                ),
                            )
                        else:
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
                        if using_prealign:
                            cv2.putText(
                                frame,
                                f"PREALIGN digit {digit} (no capture)",
                                (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), 2,
                            )
                        # 对圆心做轻量 EMA 平滑：识别偶发跳边只被衰减，
                        # 不会像“丢弃帧”那样把车辆完全卡住；切换数字时重新初始化。
                        if place_smooth_center is None or place_last_digit != digit:
                            place_smooth_center = (float(ring_cx), float(ring_cy))
                        else:
                            place_smooth_center = (
                                PLACE_CENTER_SMOOTH_ALPHA * ring_cx
                                + (1.0 - PLACE_CENTER_SMOOTH_ALPHA) * place_smooth_center[0],
                                PLACE_CENTER_SMOOTH_ALPHA * ring_cy
                                + (1.0 - PLACE_CENTER_SMOOTH_ALPHA) * place_smooth_center[1],
                            )
                        ring_cx, ring_cy = (
                            int(round(place_smooth_center[0])),
                            int(round(place_smooth_center[1])),
                        )

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

                        cur_offset = abs(ring_cx - w_img // 2)
                        cur_y_offset = abs(ring_cy - h_img // 2)
                        aligned = (
                            position_stable
                            and cur_offset <= PLACE_CENTER_TOLERANCE_PX
                            and cur_y_offset <= PLACE_CENTER_TOLERANCE_Y_PX
                        )
                        if using_prealign:
                            # 预对准阶段只调车，不执行放置动作
                            capture = False
                            prealign_finished = aligned
                        else:
                            capture = aligned
                            prealign_finished = False
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
                            # 绝对目标：目标位置 = 回传位置 + 视觉误差 × 增益。
                            # 增益 < 1 时只把偏差修正一部分，底盘响应更平稳。
                            chassis_x_mm = int(round(
                                chassis_x + chassis_x_mm * PLACE_VISUAL_GAIN
                            ))
                            chassis_y_mm = int(round(
                                chassis_y + chassis_y_mm * PLACE_VISUAL_GAIN
                            ))
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

                        # capture=1 触发帧不携带残余底盘位移：
                        # 底盘字段改写为下位机当前回传位置，动作只在本位置执行，
                        # 避免下位机收到动作请求后继续走完剩余几毫米目标。
                        send_x_mm, send_y_mm = chassis_x_mm, chassis_y_mm
                        if capture:
                            send_x_mm = int(round(chassis_x))
                            send_y_mm = int(round(chassis_y))

                        vg = VisionToGimbal(
                            target=slot_index,
                            number=digit,
                            action=PLACE_ACTION,
                            capture=capture,
                            chassis_x_mm=send_x_mm,
                            chassis_y_mm=send_y_mm,
                            gripper_mm=gripper_mm,
                        )
                        sent = tracking_send_allowed(
                            capture,
                            (send_x_mm, send_y_mm, gripper_mm),
                            deadband_mm=PLACE_CHASSIS_DEADBAND_MM,
                        )
                        if sent:
                            enqueue_latest(q, vg)
                            log_command(
                                "放置", slot_index, digit, PLACE_ACTION, capture,
                                send_x_mm, send_y_mm, gripper_mm,
                                chassis_x, chassis_y, chassis_vx, chassis_vy,
                                fb_gripper_mm=fb_gripper_mm,
                                camera_coord=coord,
                                world_coord=transformer.camera_to_world(coord),
                                image_center=(
                                    ring_cx, ring_cy,
                                    w_img // 2, h_img // 2,
                                ),
                            )
                        else:
                            now = time.time()
                            if now - last_place_dbg >= 1.0:
                                last_place_dbg = now
                                print(
                                    f"[放置] 识别到了但未发送: capture={capture} "
                                    f"cmd={(chassis_x_mm, chassis_y_mm, gripper_mm)} "
                                    f"上次发送={last_sent_tracking_mm} "
                                    f"距上次发送={now - last_tracking_send:.2f}s "
                                    f"x偏移={cur_offset}px"
                                )

                        if prealign_finished:
                            if digit == next_place_digit:
                                # 预对准用的正好是当前应放数字：不移动，直接开始放置
                                print(
                                    f"[放置] 已用圆环数字 {digit} 完成预对准，"
                                    "当前即应放数字，开始放置"
                                )
                                place_prealign_active = False
                                reset_action_state()
                                continue
                            if next_place_digit is not None:
                                # 预对准完成：不再用数字 2 继续调车，发当前应放数字
                                # （如 1），让下位机前往对应圆环后由视觉继续对准放置。
                                next_slot = slot_of_place_digit.get(next_place_digit, 0)
                                print(
                                    f"[放置] 已用圆环数字 {digit} 调整完车的位置，"
                                    f"发送应放数字 {next_place_digit} "
                                    f"(target={next_slot})，等待下位机到达指定圆环"
                                )
                                enqueue_latest(q, VisionToGimbal(
                                    target=next_slot,
                                    number=next_place_digit,
                                    action=PLACE_ACTION,
                                    capture=False,
                                ))
                                place_prealign_active = False
                                place_waiting_arrived = True
                                reset_action_state()
                                continue
                            place_prealign_active = False

                        if capture and last_placed_digit is None:
                            last_placed_digit = digit
                            detection_sent = True
                            waiting_for_next = True
                            sent_time = time.time()
                            capture_ack_received = False
                            capture_last_sent_time = time.time()
                            last_capture_vg = vg
                            place_dist = np.hypot(
                                ring_cx - w_img // 2,
                                ring_cy - h_img // 2,
                            )
                            infer_txt = (
                                "（颜色推断）"
                                if target.get("inferred_by_color") else ""
                            )
                            print(
                                f"圆环数字 {digit} 已对准{infer_txt}"
                                f"(x偏移={cur_offset}px, "
                                f"y偏移={cur_y_offset}px, "
                                f"图像中心距离={place_dist:.0f}px)，"
                                f"放置槽位 {slot_index} 的物块"
                            )

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
                        # 抓取区第一个目标：初始化“第一目标已在场”判定窗口
                        if (GRAB_SKIP_FIRST_ENABLED and not tray_phase_active
                                and target_index == 0):
                            grab_skip_first_checked = False
                            grab_skip_first_active = False
                            grab_skip_first_miss_seen = False
                            grab_skip_first_fake_miss = False
                            grab_skip_first_eval_left = GRAB_SKIP_FIRST_EVAL_FRAMES
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

                    # ---- “开始识别时第一目标已在场”跳过逻辑（仅抓取区第一个目标） ----
                    # 到达抓取区开始识别时，若判定窗口内就检测到第一目标颜色，
                    # 说明复位期间它已经/正在转走，本圈不跟踪不抓取；
                    # 等它真正离开画面、下一圈重新出现后再按正常流程抓取。
                    if (GRAB_SKIP_FIRST_ENABLED and not tray_phase_active
                            and target_index == 0):
                        if not grab_skip_first_checked:
                            if grab_skip_first_eval_left > 0:
                                grab_skip_first_eval_left -= 1
                                if data and current_color:
                                    print(
                                        f"[跳过本圈] 开始识别时第一目标颜色"
                                        f"{current_color}已在场，本圈不抓，"
                                        "等它转回来"
                                    )
                                    grab_skip_first_checked = True
                                    grab_skip_first_active = True
                                    grab_skip_first_miss_seen = False
                                    data = None
                                    current_center = None
                                    current_color = None
                                    grab_skip_first_fake_miss = True
                            else:
                                # 判定窗口结束仍没看到第一目标：按正常流程
                                grab_skip_first_checked = True
                        elif grab_skip_first_active:
                            if data and current_color:
                                if grab_skip_first_miss_seen:
                                    # 已真正离开过画面，这是下一圈：恢复正常
                                    grab_skip_first_active = False
                                    grab_skip_first_miss_seen = False
                                    print(
                                        f"[恢复跟踪] 第一目标颜色{current_color}转回来了，"
                                        "按正常流程抓取"
                                    )
                                else:
                                    # 同一圈内继续忽略本帧（不更新检测/滤波/发送）
                                    data = None
                                    current_center = None
                                    current_color = None
                                    grab_skip_first_fake_miss = True

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
                        elif FILTER_TYPE == "kalman_world":
                            # 世界系卡尔曼：测量 = 像素 → 车中心系坐标(mm)；
                            # 底盘速度/加速度作为已知控制输入参与预测。
                            dt = 1.0 / 30.0
                            if last_kf_time is not None:
                                dt = min(max(frame_t - last_kf_time, 0.005), 0.2)
                            kwf.set_dt(dt)

                            block_height = (
                                transformer.BLOCK_HEIGHT_PLACED_CM
                                if (tray_phase_active
                                    or slot_index in placed_block_slots)
                                else transformer.BLOCK_HEIGHT_CM
                            )
                            # 用回传的夹爪绝对位置作为相机位置；
                            # 反馈可能量化/抖动，先低通再灌进卡尔曼测量，防止相机原点跳变
                            gripper_cm_meas += (
                                KALMAN_WORLD_GRIPPER_MEAS_FILTER
                                * (current_gripper_cm - gripper_cm_meas)
                            )
                            gripper_cm = gripper_cm_meas
                            world_measure = None
                            if transformer.CAMERA_FOCAL_PX_X is not None:
                                cam_coord = transformer.pixel_to_camera(
                                    current_center[0], current_center[1],
                                    image_width=w_img,
                                    image_height=h_img,
                                    block_height_cm=block_height,
                                    gripper_extension_cm=gripper_cm,
                                )
                                if cam_coord is not None:
                                    world_coord = transformer.camera_to_world(
                                        cam_coord, gripper_extension_cm=gripper_cm
                                    )
                                    world_measure = (
                                        world_coord[0] * 10.0,
                                        world_coord[1] * 10.0,
                                    )

                            kwf.predict(
                                chassis_vx_mm_s=(
                                    chassis_vx if KALMAN_WORLD_USE_CHASSIS_VEL else 0.0
                                ),
                                chassis_vy_mm_s=(
                                    chassis_vy if KALMAN_WORLD_USE_CHASSIS_VEL else 0.0
                                ),
                                chassis_ax_mm_s2=0.0,   # 原始协议无加速度回传
                                chassis_ay_mm_s2=0.0,
                            )
                            if world_measure is not None:
                                kwf.update(world_measure[0], world_measure[1])
                            fx, fy, fvx, fvy, fax, fay = kwf.get_state()
                            last_kf_time = frame_t
                            # 终端调试：滤波状态 + 原始测量（0.5s 一条）
                            if time.time() - last_worldkf_print >= 0.5:
                                last_worldkf_print = time.time()
                                raw_txt = (
                                    f"raw=({world_measure[0]:.1f},{world_measure[1]:.1f})mm"
                                    if world_measure is not None else "raw=无效"
                                )
                                print(
                                    f"[WorldKF] pos=({fx:.1f},{fy:.1f})mm "
                                    f"vel=({fvx:.1f},{fvy:.1f})mm/s "
                                    f"acc=({fax:.1f},{fay:.1f})mm/s² "
                                    f"chassis=({chassis_x},{chassis_y})mm "
                                    f"cvel=({chassis_vx},{chassis_vy})mm/s "
                                    f"gripper={gripper_cm:.1f}cm {raw_txt}"
                                )
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
                        if FILTER_TYPE == "kalman_world":
                            # 世界系状态是 mm：反投影回像素画滤波圈/速度箭头
                            filtered_center = current_center
                            world_px = transformer.world_to_pixel(
                                (fx, fy), gripper_cm, block_height, w_img, h_img
                            )
                            if viz_on and viz["draw_filtered"] and world_px is not None:
                                cv2.circle(frame, world_px, radius, draw_color, 2)
                                cv2.circle(frame, world_px, 4, draw_color, -1)
                            if viz_on and viz["draw_speed"] and world_px is not None:
                                speed = np.hypot(fvx, fvy)
                                if speed > 1.0:
                                    k = min(80.0, speed * 0.25) / speed
                                    tip = (
                                        int(world_px[0] + fvx * k),
                                        int(world_px[1] + fvy * k),
                                    )
                                    cv2.arrowedLine(frame, world_px, tip,
                                                    (0, 200, 255), 2, tipLength=0.25)
                                    cv2.putText(frame, f"V={speed:.0f}mm/s",
                                                (tip[0] + 5, tip[1]),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                                (0, 200, 255), 1)
                        else:
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
                            if viz["draw_speed"] and FILTER_TYPE != "kalman_world":
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
                            if (viz["draw_trajectory"] or viz["draw_intercept"]) \
                                    and FILTER_TYPE != "kalman_world":
                                if FILTER_TYPE == "kalman":
                                    future = kf.predict_future()
                                elif FILTER_TYPE == "one_euro":
                                    future = one_euro_tracker.predict_future(
                                        T=KALMAN_CFG["predict"]["horizon_s"],
                                        steps=KALMAN_CFG["predict"]["steps"],
                                    )
                                else:
                                    future = []
                            if viz["draw_trajectory"] and FILTER_TYPE != "kalman_world":
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
                            if viz["draw_intercept"] and FILTER_TYPE != "kalman_world":
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

                            # ── 世界系：预测轨迹反投影回像素 ──
                            if viz["draw_trajectory"] and FILTER_TYPE == "kalman_world":
                                prev_pt = current_center
                                for wx, wy in kwf.predict_future():
                                    pt = transformer.world_to_pixel(
                                        (wx, wy), gripper_cm, block_height, w_img, h_img
                                    )
                                    if pt is None:
                                        continue
                                    cv2.line(frame, prev_pt, pt, (0, 255, 255), 1)
                                    cv2.circle(frame, pt, 2, (0, 255, 255), -1)
                                    prev_pt = pt

                            # ── 世界系：滤波状态数值叠加显示 ──
                            if FILTER_TYPE == "kalman_world":
                                cv2.putText(frame,
                                            f"World pos=({fx:.0f},{fy:.0f})mm "
                                            f"vel=({fvx:.0f},{fvy:.0f})mm/s "
                                            f"acc=({fax:.0f},{fay:.0f})mm/s²",
                                            (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 255, 0), 1)

                            # 显示检测状态
                            track_label = {
                                "kalman": "KF",
                                "kalman_world": "WorldKF",
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
                            if KALMAN_WORLD_ENABLED:
                                # 世界系模式：不做像素拦截规划，显示目标用原始检测点
                                target_x, target_y = current_center
                                T_solve = 0.0
                            else:
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

                            if KALMAN_WORLD_ENABLED:
                                # 世界系下 fx/fy 是 mm，对准判断仍用原始像素中心
                                cur_offset = abs(current_center[0] - w_img // 2)
                                cur_y_offset = abs(current_center[1] - h_img // 2)
                            else:
                                cur_offset = abs(fx - w_img // 2)
                                cur_y_offset = abs(fy - h_img // 2)

                            # 目标中心必须在检测 ROI 内（含滤波滞后），
                            # 防止物块已经跑出区域仍被切边检测/旧滤波位置触发抓取
                            roi = detector.detection_area
                            if KALMAN_WORLD_ENABLED:
                                target_in_roi = (
                                    roi is None
                                    or (roi[0] <= current_center[0] <= roi[0] + roi[2]
                                        and roi[1] <= current_center[1] <= roi[1] + roi[3])
                                )
                            else:
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
                                and cur_y_offset <= GRAB_CENTER_TOLERANCE_Y_PX
                            )
                            capture = would_capture and target_in_roi
                            if would_capture and not target_in_roi:
                                print(
                                    f"[ROI] 目标已出检测区 ({fx:.0f},{fy:.0f})，"
                                    f"不在 {roi} 内，暂不抓取"
                                )

                            # 抓取区（含后续遍次）和非倒序托盘阶段复用第一次抓取的方式
                            # （含 grab_gripper_fixed）；倒序托盘阶段保持原策略不变。
                            # 夹爪长度固定后只靠底盘把物块带到固定距离，
                            # 避免“相机随夹爪伸缩 → 测量滞后 → 闭环震荡”。
                            _tray_fixed_gripper = (
                                tray_phase_active
                                and (TRAY_GRIPPER_FIXED_CUSTOM
                                     or TRAY_GRIPPER_FIXED_MIN
                                     or TRAY_GRIPPER_FIXED_MAX)
                            )
                            # 倒序托盘阶段走原策略；其余（抓取区、后续遍次、
                            # actual 托盘）都按第一次抓取的方式处理。
                            _follow_first_grab = (
                                not tray_phase_active or not TRAY_PHASE_IS_REVERSE
                            )
                            _follow_first_grab_fixed = (
                                _follow_first_grab and GRAB_GRIPPER_FIXED_CUSTOM
                            )
                            if _follow_first_grab_fixed:
                                track_gripper_cm = GRAB_GRIPPER_EXTEND_CM
                                track_fixed_gripper_cm = (
                                    transformer.min_jar_dis[1]
                                    + GRAB_GRIPPER_EXTEND_CM
                                )
                            elif _follow_first_grab:
                                # 第一次抓取未配置固定夹爪时同样动态调夹爪
                                track_gripper_cm = current_gripper_cm
                                track_fixed_gripper_cm = None
                            elif _tray_fixed_gripper:
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

                            # 非倒序固定夹爪抓取：先等夹爪反馈到位，再允许底盘跟踪
                            if _follow_first_grab_fixed:
                                gripper_ready = update_gripper_settle(
                                    GRAB_GRIPPER_MM,
                                    time.time(),
                                    chassis_data2 is not None,
                                    fb_gripper_mm,
                                )
                            else:
                                update_gripper_settle(
                                    None, time.time(), True, 0.0
                                )
                                gripper_ready = True

                            if not gripper_ready:
                                # 夹爪还没到位：不允许请求抓取，也不动底盘
                                would_capture = False
                                capture = False

                            if KALMAN_WORLD_ENABLED:
                                # 世界系：直接用卡尔曼输出的车中心系坐标(mm→cm)生成指令
                                block_camera_coord = None
                                cmd_mm = transformer.world_to_protocol_mm(
                                    [fx / 10.0, fy / 10.0, 0.0],
                                    fixed_gripper_cm=track_fixed_gripper_cm,
                                )
                                if cmd_mm == (0, 0, 0):
                                    warn_zero_command(
                                        "抓取", None, current_center[0], current_center[1]
                                    )
                                chassis_x_mm, chassis_y_mm, gripper_mm = cmd_mm
                                if track_fixed_gripper_cm is not None:
                                    gripper_mm = (
                                        GRAB_GRIPPER_MM
                                        if _follow_first_grab_fixed
                                        else TRAY_GRIPPER_MM
                                    )
                                # 绝对目标模式（像夹爪一样）：
                                # 目标位置 = 回传位置 + 视觉误差（可选加前瞻）
                                # x：把物块带到车中心线；y：把物块带到夹爪目标距离处
                                grab_dist_mm = (
                                    transformer.min_jar_dis[1] * 10.0 + gripper_mm
                                )
                                if CHASSIS_LOOKAHEAD_S > 0.0:
                                    # 前瞻需要物块“地面速度”：
                                    # use_chassis_velocity=true 时卡尔曼速度已是地面速度；
                                    # false 时速度是相对车中心的速度，要补回底盘速度。
                                    if KALMAN_WORLD_USE_CHASSIS_VEL:
                                        la_vx, la_vy = fvx, fvy
                                    else:
                                        la_vx = fvx + chassis_vx
                                        la_vy = fvy + chassis_vy
                                    la_speed = np.hypot(la_vx, la_vy)
                                    if (
                                        la_speed > CHASSIS_LOOKAHEAD_MAX_SPEED_MM_S
                                        and la_speed > 1e-6
                                    ):
                                        k = CHASSIS_LOOKAHEAD_MAX_SPEED_MM_S / la_speed
                                        la_vx *= k
                                        la_vy *= k
                                    target_fx = fx + la_vx * CHASSIS_LOOKAHEAD_S
                                    target_fy = fy + la_vy * CHASSIS_LOOKAHEAD_S
                                else:
                                    target_fx, target_fy = fx, fy
                                chassis_x_mm = int(round(chassis_x + target_fx))
                                chassis_y_mm = int(round(
                                    chassis_y + (target_fy - grab_dist_mm)
                                ))
                                if not gripper_ready:
                                    # 只发夹爪到位指令，底盘目标保持当前位置
                                    chassis_x_mm = int(round(chassis_x))
                                    chassis_y_mm = int(round(chassis_y))
                                    gripper_mm = GRAB_GRIPPER_MM
                                chassis_x_mm, chassis_y_mm, gripper_mm = sanitize_protocol_mm(
                                    (chassis_x_mm, chassis_y_mm, gripper_mm), last_valid_mm
                                )
                                desired_mm = (chassis_x_mm, chassis_y_mm, gripper_mm)
                                chassis_x_mm, chassis_y_mm, gripper_mm = smooth_tracking_cmd(
                                    desired_mm, time.time()
                                )
                                last_valid_mm = (chassis_x_mm, chassis_y_mm, gripper_mm)
                            else:
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
                                        gripper_mm = (
                                            GRAB_GRIPPER_MM
                                            if _follow_first_grab_fixed
                                            else TRAY_GRIPPER_MM
                                        )
                                    # 绝对目标：目标位置 = 回传位置 + 视觉误差
                                    chassis_x_mm = int(round(chassis_x + chassis_x_mm))
                                    chassis_y_mm = int(round(chassis_y + chassis_y_mm))
                                    if not gripper_ready:
                                        # 只发夹爪到位指令，底盘目标保持当前位置
                                        chassis_x_mm = int(round(chassis_x))
                                        chassis_y_mm = int(round(chassis_y))
                                        gripper_mm = GRAB_GRIPPER_MM
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
                                    if not gripper_ready:
                                        chassis_x_mm = int(round(chassis_x))
                                        chassis_y_mm = int(round(chassis_y))
                                        gripper_mm = GRAB_GRIPPER_MM
                                        desired_mm = (
                                            chassis_x_mm, chassis_y_mm, gripper_mm
                                        )
                                    warn_invalid_coord(
                                        "抓取", target_x, target_y,
                                        frame.shape[1], frame.shape[0],
                                    )

                            if not gripper_ready:
                                cv2.putText(
                                    frame,
                                    f"Gripper settling {fb_gripper_mm:.0f}/{GRAB_GRIPPER_MM}mm",
                                    (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 200, 255), 1,
                                )

                            # capture=1 触发帧不携带残余底盘位移：
                            # 底盘字段改写为下位机当前回传位置，动作只在本位置执行。
                            send_x_mm, send_y_mm = chassis_x_mm, chassis_y_mm
                            if capture:
                                send_x_mm = int(round(chassis_x))
                                send_y_mm = int(round(chassis_y))

                            # 托盘阶段复用抓取逻辑，但 number 保持当前托盘号
                            # （如 1），避免后续跟踪/抓取包把 num 置 0。
                            track_number = (
                                tray_pending_digit
                                if tray_phase_active and tray_pending_digit is not None
                                else 0
                            )
                            vg = VisionToGimbal(
                                target=slot_index,
                                number=track_number,
                                action=GRAB_ACTION,
                                capture=capture,
                                chassis_x_mm=send_x_mm,
                                chassis_y_mm=send_y_mm,
                                gripper_mm=gripper_mm,
                            )
                            if tracking_send_allowed(
                                capture,
                                (send_x_mm, send_y_mm, gripper_mm),
                            ):
                                enqueue_latest(q, vg)
                                log_command(
                                    "抓取", slot_index, track_number, GRAB_ACTION, capture,
                                    send_x_mm, send_y_mm, gripper_mm,
                                    chassis_x, chassis_y, chassis_vx, chassis_vy,
                                    fb_gripper_mm=fb_gripper_mm,
                                    camera_coord=block_camera_coord,
                                    world_coord=(
                                        [fx / 10.0, fy / 10.0, 0.0]
                                        if KALMAN_WORLD_ENABLED
                                        else (transformer.camera_to_world(block_camera_coord)
                                              if block_camera_coord is not None else None)
                                    ),
                                    image_center=(
                                        (current_center[0], current_center[1],
                                         w_img // 2, h_img // 2)
                                        if KALMAN_WORLD_ENABLED
                                        else (fx, fy, w_img // 2, h_img // 2)
                                    ),
                                )

                                if capture:
                                    grab_dist = np.hypot(cur_offset, cur_y_offset)
                                    print(
                                        f"物块已对准 (x偏移={cur_offset:.0f}px, "
                                        f"y偏移={cur_y_offset:.0f}px, "
                                        f"图像中心距离={grab_dist:.0f}px)，请求抓取 "
                                        f"颜色{current_color} → 槽位{slot_index}"
                                    )
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
                                    f"Aligning... off={cur_offset:.0f}/{cur_y_offset:.0f}px "
                                    f"T={T_solve:.2f}s "
                                    f"-> ({target_x},{target_y}) capture=0",
                                    (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 200, 255), 1)

                    else:
                        # 本帧未识别到目标颜色：颜色计数清零，
                        # 重新出现后必须再连续攒够 color_stable_threshold 帧
                        target_label = COLOR_LABEL_EN.get(
                            CODE_TO_KEY.get(current_target_code), current_target_code
                        )
                        if grab_skip_first_fake_miss:
                            # 目标实际还在画面里，只是被“第一目标已在场”逻辑有意忽略，
                            # 不能当作真正消失（否则会误判为下一圈已到）
                            grab_skip_first_fake_miss = False
                            hint_txt = "第一目标已在场，等待下一圈..."
                        else:
                            if grab_skip_first_active:
                                grab_skip_first_miss_seen = True
                            hint_txt = f"Looking for {target_label}..."
                        detector.on_miss()
                        # 同步重置平滑状态，恢复识别后从 0 重新小步爬升，
                        # 避免用断帧前/后的陈旧时间差一次性补大步。
                        last_smooth_cmd_mm = (
                            chassis_x, chassis_y, last_smooth_cmd_mm[2]
                        )
                        last_smooth_time = time.time()
                        last_sent_tracking_mm = None

                        cv2.putText(frame, hint_txt,
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

        if (
            _QR_DISPLAY_PROC is not None
            and _QR_DISPLAY_PROC.poll() is None
        ):
            print(
                f"[QR显示] 外接屏显示进程保持运行 "
                f"(pid={_QR_DISPLAY_PROC.pid})，关闭本程序不影响显示"
            )

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


class _LogTee:
    """把 print 同时写到原控制台流和 log.txt，且立即落盘。"""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, text):
        try:
            self._stream.write(text)
        except Exception:
            pass
        try:
            self._log_file.write(text)
            self._log_file.flush()
        except Exception:
            pass
        return len(text)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            self._log_file.flush()
        except Exception:
            pass


if __name__ == "__main__":
    import sys
    from datetime import datetime

    # systemd 停止服务时默认发 SIGTERM；转成 KeyboardInterrupt，
    # 让主程序的 finally 清理逻辑（释放资源）照常执行。
    import signal

    def _handle_sigterm(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_sigterm)

    _log_file = None
    try:
        _log_file = open(LOG_FILE, "a", encoding="utf-8")
    except OSError as _log_err:
        print(f"[日志] 无法打开 {LOG_FILE}：{_log_err}")

    if _log_file is not None:
        sys.stdout = _LogTee(sys.stdout, _log_file)
        sys.stderr = _LogTee(sys.stderr, _log_file)
        _started_at = datetime.now()
        print(f"\n===== 程序启动 {_started_at:%Y-%m-%d %H:%M:%S} =====")

    try:
        main()
    finally:
        if _log_file is not None:
            try:
                _log_file.write(
                    f"===== 程序退出 {datetime.now():%Y-%m-%d %H:%M:%S}"
                    f"（本次运行 {datetime.now() - _started_at}） =====\n"
                )
                _log_file.flush()
            except Exception:
                pass

            # 每次运行结束后自动画图（只画最近一次运行的数据）
            try:
                from plot_log import plot_log as _render_log_plot
                _render_log_plot(
                    LOG_FILE,
                    Path(__file__).resolve().parent / "log_plot.png",
                    last_run=True,
                )
            except Exception as _plot_err:
                print(f"[日志] 绘图失败: {_plot_err}")

            try:
                _log_file.close()
            except Exception:
                pass
