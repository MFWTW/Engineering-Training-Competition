"""
坐标转换 + 夹爪/底盘协同决策

坐标系约定（单位：cm）：
    world_coordinate      世界系原点 = 车中心在地面的投影
    x 轴 = 车左侧（x 正方向朝左），y 轴 = 车前方，z 轴 = 上方

当前假设：摄像头与车轴平行安装、无旋转，因此
    世界坐标 = 相机坐标 + camera1world_coordinate + 当前夹爪伸长量（仅 y 方向）

注意：摄像头装在夹爪/云台上，会随夹爪一起伸缩，
所以相机相对车中心的前方距离 = camera1world_coordinate[1] + 当前夹爪伸长量。

决策逻辑：
    1. 先让底盘左右移动，使物块与夹爪在同一 x 轴上；
    2. 按物块的纵向距离 dy 判断：
       - dy < 最短距离：底盘前后移动，夹爪保持最短距离；
       - dy > 最长距离：底盘前后移动，夹爪保持最长距离；
       - 最短~最长之间：只动夹爪，夹爪伸到 dy。
"""

import math
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


# 世界坐标系原点（车中心）
world_coordinate = [0, 0, 0]

# ==================== 可调参数统一从 config.yaml → transformer 段读取 ====================
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _load_transformer_cfg():
    """读取 config.yaml 的 transformer 段；读取失败时返回空 dict 使用内置默认值"""
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("transformer", {}) or {}
    except Exception as exc:
        print(f"[transformer] 读取 {CONFIG_PATH} 失败，使用内置默认参数: {exc}")
        return {}


_TRANSFORMER_CFG = _load_transformer_cfg()


def _cfg_float(key, default):
    try:
        return float(_TRANSFORMER_CFG[key])
    except (KeyError, TypeError, ValueError):
        return default


def _cfg_int(key, default):
    try:
        return int(_TRANSFORMER_CFG[key])
    except (KeyError, TypeError, ValueError):
        return default


def _cfg_list(key, default):
    try:
        return [float(v) for v in _TRANSFORMER_CFG[key]]
    except (KeyError, TypeError, ValueError):
        return list(default)


# 摄像头光心相对世界坐标（车中心）的位置，单位 cm
camera1world_coordinate = _cfg_list("camera1world_coordinate", [0, 24.49, 20])

# 架爪中心距离车中心的最远/最近距离，z 轴忽略（下位机夹的时候写死高度）
max_jar_dis = _cfg_list("max_jar_dis", [0, 30.005, 0])
min_jar_dis = _cfg_list("min_jar_dis", [0, 21.595, 0])
# 夹爪最长行程（cm / mm）：放置阶段夹爪固定最长时使用
MAX_GRIPPER_EXTEND_CM = max_jar_dis[1] - min_jar_dis[1]
MAX_GRIPPER_EXTEND_MM = int(round(MAX_GRIPPER_EXTEND_CM * 10))

# 左右对齐容差（cm），小于该值认为已经对齐；
# 注意按相机高度/物块高度换算，1cm 在画面里可能对应几十像素，取太大会让
# 20px 级别的左右偏差直接输出 0 指令，导致底盘不动。
LATERAL_TOLERANCE_CM = _cfg_float("lateral_tolerance_cm", 0.05)
LONGITUDINAL_TOLERANCE_CM = _cfg_float("longitudinal_tolerance_cm", 0.5)

# ---- 像素 → 相机坐标需要的内参（单位：像素），来自 camera_calibration.json ----
# 物块检测 USB 相机标定原图 @ 640x480（检测相机实际出图分辨率）；
# 实际出图分辨率不同时仍会按比例自动缩放
CALIB_IMAGE_WIDTH = _cfg_int("calib_image_width", 640)
CALIB_IMAGE_HEIGHT = _cfg_int("calib_image_height", 480)
_last_coord_warn_t = 0.0
CAMERA_FOCAL_PX_X = _cfg_float("focal_px_x", 366.99093917961704)        # fx
CAMERA_FOCAL_PX_Y = _cfg_float("focal_px_y", 364.25895585724794)        # fy
CAMERA_PRINCIPAL_PX_X = _cfg_float("principal_px_x", 314.8605084989312)  # cx
CAMERA_PRINCIPAL_PX_Y = _cfg_float("principal_px_y", 247.66418301001914) # cy
CAMERA_DIST_COEFFS = _cfg_list(
    "dist_coeffs",
    [-0.012076444385052124, 0.0016723981571138394, 0.0018108447968222718,
     -0.002827491845898441, -0.024264465499757897],
)
CAMERA_HEIGHT_CM = _cfg_float("camera_height_cm", 22.5)
BLOCK_HEIGHT_CM = _cfg_float("block_height_cm", 14.3)            # 抓取区立放物块顶面高度
BLOCK_HEIGHT_PLACED_CM = _cfg_float("block_height_placed_cm", 6.0)  # 放置后躺倒再次夹取时高度
CAMERA_PITCH_DEG = _cfg_float("camera_pitch_deg", 90.0)  # 正=向下俯拍；垂直向下时填 90


def pixel_to_camera(
    u,
    v,
    image_width=CALIB_IMAGE_WIDTH,
    image_height=CALIB_IMAGE_HEIGHT,
    focal_px_x=None,
    focal_px_y=None,
    principal_x=None,
    principal_y=None,
    camera_height_cm=CAMERA_HEIGHT_CM,
    block_height_cm=BLOCK_HEIGHT_CM,
    camera_pitch_deg=None,
    dist_coeffs=None,
    gripper_extension_cm=0.0,
):
    """
    像素坐标 (u, v) → 相机坐标系 [x, y, z]（cm）

    原理：小孔成像 + “物块在地面上”假设。
    相机支持俯仰角（正=向下俯拍），偏航仍为 0；
    相机装在夹爪/云台上，夹爪最短时在车中心前方 24.49cm、高 22.5cm；
    夹爪伸长 gripper_extension_cm 时，相机也向前移动同样的距离。

    相机系定义与世界系平行：
        x 正方向朝左，y 正方向朝前，z 正方向朝上

    Returns:
        [x_cm, y_cm, z_cm]  相机系坐标；标定未完成或无法计算时返回 None
    """
    # 使用内置标定内参时，按当前实际画面分辨率等比缩放（默认基准 640x480）；
    # 显式传入的内参保持原样，不自动缩放。
    scale_x = image_width / float(CALIB_IMAGE_WIDTH)
    scale_y = image_height / float(CALIB_IMAGE_HEIGHT)
    if focal_px_x is None:
        focal_px_x = CAMERA_FOCAL_PX_X * scale_x
    if focal_px_y is None:
        focal_px_y = CAMERA_FOCAL_PX_Y * scale_y
    if principal_x is None:
        principal_x = CAMERA_PRINCIPAL_PX_X * scale_x
    if principal_y is None:
        principal_y = CAMERA_PRINCIPAL_PX_Y * scale_y
    if dist_coeffs is None:
        dist_coeffs = CAMERA_DIST_COEFFS
    if camera_pitch_deg is None:
        camera_pitch_deg = CAMERA_PITCH_DEG

    if focal_px_x is None or focal_px_y is None:
        print("[transformer] 请先标定 fx/fy（CAMERA_FOCAL_PX_X/Y）")
        return None

    cx = principal_x if principal_x is not None else (image_width - 1) / 2.0
    cy = principal_y if principal_y is not None else (image_height - 1) / 2.0

    # 先用标定得到的畸变系数把原始像素坐标校正为理想针孔坐标
    camera_matrix = np.array(
        [[focal_px_x, 0.0, cx], [0.0, focal_px_y, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.array(dist_coeffs, dtype=np.float64).reshape(1, -1)
    pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
    corrected = cv2.undistortPoints(pts, camera_matrix, dist, P=camera_matrix)
    u_corr, v_corr = corrected[0, 0]

    delta_u = u_corr - cx
    delta_v = v_corr - cy

    h_eff = camera_height_cm - block_height_cm
    if h_eff <= 0:
        return None

    theta = math.radians(camera_pitch_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # 俯仰相机下，地面点水平前方距离：
    # Y = H * (fy*cosθ - Δv*sinθ) / (Δv*cosθ + fy*sinθ)
    denom = delta_v * cos_t + focal_px_y * sin_t
    forward_cm = h_eff * (
        focal_px_y * cos_t - delta_v * sin_t
    ) / denom

    # 相机装在夹爪上：相对车中心前方 camera1world_coordinate[1] + 当前夹爪伸长量。
    # 图像主点以下的点换算出的相机前方距离是负的，但它们仍可能在车中心前方
    # （例如车前方 13~24cm），仍然有效。只拒绝换算后已到车中心后方的点。
    world_forward_cm = (
        forward_cm + camera1world_coordinate[1] + gripper_extension_cm
    )
    if denom <= 0 or world_forward_cm <= 0:
        global _last_coord_warn_t
        now = time.time()
        if now - _last_coord_warn_t > 1.0:
            _last_coord_warn_t = now
            print(
                f"[transformer] 坐标无效: 目标({u},{v}) @ "
                f"{image_width}x{image_height}, 主点行 cy={cy:.1f}, "
                f"delta_v={delta_v:.1f}, 相机前方={forward_cm:.1f}cm, "
                f"车中心前方={world_forward_cm:.1f}cm, 俯仰角={camera_pitch_deg:.1f}°, "
                "目标已在车中心后方，无法抓取"
            )
        return None

    # 光轴方向深度（用于横向换算）
    depth_along_axis_cm = forward_cm * cos_t + h_eff * sin_t
    # 左右偏移：图像右边(u增大)对应世界左边(x负)，因为 x 正方向朝左
    lateral_cm = -delta_u * depth_along_axis_cm / focal_px_x
    # 相机系 z：地面点相对相机光心向下，因此是负的
    height_cm = -h_eff

    return [
        float(round(lateral_cm, 2)),
        float(round(forward_cm, 2)),
        float(round(height_cm, 2)),
    ]


def world_to_pixel(
    world_mm,
    gripper_extension_cm=0.0,
    block_height_cm=BLOCK_HEIGHT_CM,
    image_width=CALIB_IMAGE_WIDTH,
    image_height=CALIB_IMAGE_HEIGHT,
    camera_pitch_deg=None,
):
    """
    车中心系世界坐标(mm) → 图像像素(u, v)（pixel_to_camera 的逆变换，近似）。

    用途：把世界系卡尔曼的滤波位置/预测轨迹画回画面，便于调参观察。
    忽略畸变（畸变系数很小，仅用于可视化）。

    Returns:
        (u, v) 或 None（换算无效，例如目标在相机后方/画面外）
    """
    if world_mm is None:
        return None
    if camera_pitch_deg is None:
        camera_pitch_deg = CAMERA_PITCH_DEG

    scale_x = image_width / float(CALIB_IMAGE_WIDTH)
    scale_y = image_height / float(CALIB_IMAGE_HEIGHT)
    focal_px_x = CAMERA_FOCAL_PX_X * scale_x
    focal_px_y = CAMERA_FOCAL_PX_Y * scale_y
    principal_x = CAMERA_PRINCIPAL_PX_X * scale_x
    principal_y = CAMERA_PRINCIPAL_PX_Y * scale_y

    # 世界(mm) → 相机系(cm)
    wx_cm = world_mm[0] / 10.0
    wy_cm = world_mm[1] / 10.0
    forward_cm = wy_cm - camera1world_coordinate[1] - gripper_extension_cm
    lateral_cm = wx_cm - camera1world_coordinate[0]

    h_eff = CAMERA_HEIGHT_CM - block_height_cm
    if h_eff <= 0:
        return None

    theta = math.radians(camera_pitch_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # pixel_to_camera 的逆：
    #   forward = h*(fy*cos - Δv*sin) / (Δv*cos + fy*sin)
    #   lateral = -Δu * depth / fx，depth = Δv*cos + fy*sin
    denom = forward_cm * cos_t + h_eff * sin_t
    if abs(denom) < 1e-6:
        return None
    delta_v = focal_px_y * (h_eff * cos_t - forward_cm * sin_t) / denom
    delta_u = -lateral_cm * focal_px_x / denom

    u = principal_x + delta_u
    v = principal_y + delta_v
    if u < 0 or v < 0 or u > image_width - 1 or v > image_height - 1:
        return None
    return int(round(u)), int(round(v))


def estimate_pitch_deg(
    u,
    v,
    forward_cm,
    image_width=CALIB_IMAGE_WIDTH,
    image_height=CALIB_IMAGE_HEIGHT,
    focal_px_x=None,
    focal_px_y=None,
    principal_x=None,
    principal_y=None,
    camera_height_cm=CAMERA_HEIGHT_CM,
    block_height_cm=BLOCK_HEIGHT_CM,
    dist_coeffs=None,
):
    """
    根据一个地面物块的像素位置和实测“相机到物块的水平前方距离(cm)”，
    反推相机俯仰角（正=向下俯拍）。

    forward_cm 应是从相机正下方地面点，沿车头方向量到物块的水平距离，
    不是斜着拉的卷尺距离。
    相机垂直向下时，该函数会得到约 90°。
    """
    scale_x = image_width / float(CALIB_IMAGE_WIDTH)
    scale_y = image_height / float(CALIB_IMAGE_HEIGHT)
    if focal_px_x is None:
        focal_px_x = CAMERA_FOCAL_PX_X * scale_x
    if focal_px_y is None:
        focal_px_y = CAMERA_FOCAL_PX_Y * scale_y
    if principal_x is None:
        principal_x = CAMERA_PRINCIPAL_PX_X * scale_x
    if principal_y is None:
        principal_y = CAMERA_PRINCIPAL_PX_Y * scale_y
    if dist_coeffs is None:
        dist_coeffs = CAMERA_DIST_COEFFS

    camera_matrix = np.array(
        [[focal_px_x, 0.0, principal_x],
         [0.0, focal_px_y, principal_y],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.array(dist_coeffs, dtype=np.float64).reshape(1, -1)
    pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
    corrected = cv2.undistortPoints(pts, camera_matrix, dist, P=camera_matrix)
    delta_v = corrected[0, 0, 1] - principal_y

    h_eff = camera_height_cm - block_height_cm
    if h_eff <= 0 or forward_cm <= 0:
        return None

    theta = math.atan2(
        focal_px_y * h_eff - delta_v * forward_cm,
        delta_v * h_eff + focal_px_y * forward_cm,
    )
    return math.degrees(theta)


def pixel_to_world(u, v, gripper_extension_cm=0.0, **kwargs):
    """像素坐标 → 世界坐标（车中心系），方便直接喂给 decide_gripper_chassis"""
    camera_coord = pixel_to_camera(u, v, **kwargs)
    if camera_coord is None:
        return None
    return camera_to_world(camera_coord, gripper_extension_cm=gripper_extension_cm)


def camera_to_world(camera_coordinate, gripper_extension_cm=0.0):
    """
    相机坐标系 → 世界坐标系（车中心系）。

    摄像头装在夹爪/云台上，会随夹爪一起伸缩，因此 y 方向要多加
    当前夹爪伸长量：世界 y = 相机 y + camera1world_coordinate[1] + 伸长量。
    """
    if camera_coordinate is None:
        return None
    return [
        camera_coordinate[0] + camera1world_coordinate[0],
        camera_coordinate[1] + camera1world_coordinate[1] + gripper_extension_cm,
        camera_coordinate[2] + camera1world_coordinate[2],
    ]


def world_to_camera(world_coordinate_point, gripper_extension_cm=0.0):
    """世界坐标系 → 相机坐标系（逆变换，调试用）"""
    if world_coordinate_point is None:
        return None
    return [
        world_coordinate_point[0] - camera1world_coordinate[0],
        world_coordinate_point[1] - camera1world_coordinate[1] - gripper_extension_cm,
        world_coordinate_point[2] - camera1world_coordinate[2],
    ]


def decide_gripper_chassis(block_world, fixed_gripper_cm=None):
    """
    根据物块的世界坐标，输出夹爪目标距离和底盘移动指令。

    Args:
        block_world: [x, y, z]，物块在世界系中的坐标（cm）
        fixed_gripper_cm: 固定夹爪位置（相对车中心，cm）。
            不为 None 时夹爪长度不再参与决策，只靠底盘把目标带到该位置。

    Returns:
        dict:
            mode                  "too_close" / "in_range" / "too_far"
            gripper_target_cm     夹爪需要伸出的纵向距离
            chassis_x_cm          底盘左右需要移动的距离（正=左，负=右）
            chassis_y_cm          底盘前后需要移动的距离（正=前，负=后）
            chassis_x_direction   底盘左右方向："left"/"right"/"hold"
            chassis_y_direction   底盘前后方向："forward"/"backward"/"hold"
            block_world           物块世界坐标（便于调试）
    """
    if block_world is None:
        return None

    # 夹爪始终在车中心 y 轴上：左右偏移 dx 由底盘 x 移动消除，
    # 前后/夹爪判断只看 dy（物块在 y 轴上离车中心多远），不看含 dx 的斜线距离。
    dx = block_world[0]
    dy = block_world[1]

    min_y = min_jar_dis[1]
    max_y = max_jar_dis[1]

    # ---- 左右对齐：物块不在夹爪 x=0 的轴线上时动车 ----
    if abs(dx) <= LATERAL_TOLERANCE_CM:
        chassis_x_cm = 0.0
        chassis_x_direction = "hold"
    else:
        chassis_x_cm = dx  # x 正方向朝左：物块在左就向左开
        chassis_x_direction = "left" if dx > 0 else "right"

    # ---- 前后距离决策 ----
    if fixed_gripper_cm is not None:
        # 固定夹爪：只调底盘，夹爪目标就是固定位置
        target_gripper_y = min(max(float(fixed_gripper_cm), min_y), max_y)
        mode = "fixed_gripper"
    elif dy < min_y:
        mode = "too_close"
        target_gripper_y = min_y
    elif dy > max_y:
        mode = "too_far"
        target_gripper_y = max_y
    else:
        mode = "in_range"
        target_gripper_y = dy

    # 底盘纵向移动量 = 当前相对距离 - 目标夹爪距离
    chassis_y_cm = dy - target_gripper_y

    if abs(chassis_y_cm) <= LONGITUDINAL_TOLERANCE_CM:
        chassis_y_direction = "hold"
    else:
        chassis_y_direction = "forward" if chassis_y_cm > 0 else "backward"

    return {
        "mode": mode,
        "gripper_target_cm": round(target_gripper_y, 2),
        "chassis_x_cm": round(chassis_x_cm, 2),
        "chassis_y_cm": round(chassis_y_cm, 2),
        "chassis_x_direction": chassis_x_direction,
        "chassis_y_direction": chassis_y_direction,
        "block_world": [round(v, 2) for v in block_world[:3]],
    }


def decide_from_camera(camera_coordinate, gripper_extension_cm=0.0,
                       fixed_gripper_cm=None):
    """直接传入物块在相机坐标系中的坐标 [x, y, z]，内部先转世界系再决策"""
    block_world = camera_to_world(
        camera_coordinate, gripper_extension_cm=gripper_extension_cm
    )
    if block_world is None:
        return None
    return decide_gripper_chassis(block_world, fixed_gripper_cm=fixed_gripper_cm)


def command_to_protocol_mm(camera_coordinate, gripper_extension_cm=0.0,
                           fixed_gripper_cm=None):
    """
    把决策结果转成串口需要的整数值（单位 mm，1mm 分辨率）。

    fixed_gripper_cm: 固定夹爪位置（相对车中心，cm）；放置阶段传夹爪最长位置，
        则夹爪指令固定为最长，只输出底盘移动量。

    Returns:
        (chassis_x_mm, chassis_y_mm, gripper_mm)
        chassis_x_mm 正=左，负=右；chassis_y_mm 正=前，负=后。
        gripper_mm 是相对夹爪最短位置的伸长量：最短位置=0，
        实际夹爪位置 = min_jar_dis[1] + gripper_mm/10（相对车中心）。
    """
    result = decide_from_camera(
        camera_coordinate,
        gripper_extension_cm=gripper_extension_cm,
        fixed_gripper_cm=fixed_gripper_cm,
    )
    if result is None:
        return 0, 0, 0
    gripper_mm = int(round(
        (result["gripper_target_cm"] - min_jar_dis[1]) * 10
    ))
    if gripper_mm < 0:
        gripper_mm = 0
    return (
        int(round(result["chassis_x_cm"] * 10)),
        int(round(result["chassis_y_cm"] * 10)),
        gripper_mm,
    )


def world_to_protocol_mm(block_world, fixed_gripper_cm=None):
    """
    直接用车中心系世界坐标（cm）生成串口指令，跳过像素→相机换算。

    block_world: [x, y, z]，物块相对车中心的坐标（cm，正=左/前/上）
    fixed_gripper_cm: 固定夹爪位置（相对车中心 cm），None=动态夹爪

    Returns:
        (chassis_x_mm, chassis_y_mm, gripper_mm)  单位与 command_to_protocol_mm 一致
    """
    result = decide_gripper_chassis(block_world, fixed_gripper_cm=fixed_gripper_cm)
    if result is None:
        return 0, 0, 0
    gripper_mm = int(round(
        (result["gripper_target_cm"] - min_jar_dis[1]) * 10
    ))
    if gripper_mm < 0:
        gripper_mm = 0
    return (
        int(round(result["chassis_x_cm"] * 10)),
        int(round(result["chassis_y_cm"] * 10)),
        gripper_mm,
    )


if __name__ == "__main__":
    print("=== transformer 自测 ===\n")

    # 像素 → 相机坐标：这里假设 fx=fy=1000、主点在图像中心，仅用于演示
    print("像素(770, 800) → 相机坐标（fx=fy=1000）:")
    cam_coord = pixel_to_camera(
        770, 800, focal_px_x=1000, focal_px_y=1000,
        principal_x=720, principal_y=540, dist_coeffs=[0, 0, 0, 0, 0],
    )
    print(f"  相机坐标: {cam_coord}")
    print(f"  世界坐标: {camera_to_world(cam_coord)}")
    print(f"  决策: {decide_from_camera(cam_coord)}\n")

    # 相机坐标 z 为 0 表示物块在地面高度与相机 z 的差值，这里只测 x/y
    test_cases = {
        "夹爪范围内 (dy=25cm)": [0, 25 - 24.49, -22.5],
        "太近 (dy=15cm)":       [0, 15 - 24.49, -22.5],
        "太远 (dy=35cm)":       [0, 35 - 24.49, -22.5],
        "偏左 (dx=5cm)":        [5, 28 - 24.49, -22.5],
    }

    for name, camera_coord in test_cases.items():
        result = decide_from_camera(camera_coord)
        print(f"{name}")
        print(f"  相机坐标: {camera_coord}")
        print(f"  世界坐标: {result['block_world']}")
        print(f"  模式: {result['mode']}, 夹爪目标: {result['gripper_target_cm']}cm")
        print(f"  底盘: x={result['chassis_x_cm']}cm({result['chassis_x_direction']}), "
              f"y={result['chassis_y_cm']}cm({result['chassis_y_direction']})")
        print(f"  串口(mm): {command_to_protocol_mm(camera_coord)}")
        print()
