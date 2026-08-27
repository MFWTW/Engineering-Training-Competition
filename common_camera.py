import cv2
import glob
import json
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 摄像头角色配置：存在时优先于下方两个代码常量，不需要改代码。
# 用 python3 camera_setup.py 生成，也可手动编辑。
CAMERA_ROLES_FILE = os.path.join(BASE_DIR, "camera_roles.json")


# ==================== 两台 USB 免驱摄像头配置 ====================
# 这里统一配置摄像头来源：可以填 OpenCV 设备编号（0、1），
# 也可以填 /dev/video* 路径或 udev 符号链接（如 /dev/video_qr、/dev/video_detect）。
QR_CAMERA_SOURCE = 1          # 二维码扫描相机（默认值，可被 camera_roles.json 覆盖）
DETECTION_CAMERA_SOURCE = 0   # 物块检测/放置识别相机（默认值，可被 camera_roles.json 覆盖）

# 物块检测相机默认分辨率（USB 相机不支持时会自动使用其默认分辨率）。
# 如果换用支持更高分辨率的相机，请同时用同一分辨率重新标定相机内参。
DETECTION_FRAME_WIDTH = 640
DETECTION_FRAME_HEIGHT = 480
# 物块检测相机目标帧率（fps）。要求摄像头支持该档位，否则 V4L2 会静默忽略，
# open_camera() 会打印实际生效的帧率并告警，程序仍可运行。
DETECTION_CAMERA_FPS = 30


def _load_camera_roles():
    """从 camera_roles.json 读取角色配置；文件不存在或字段全空时返回 None。"""
    try:
        with open(CAMERA_ROLES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"读取 {CAMERA_ROLES_FILE} 失败（{e}），使用 common_camera.py 默认值")
        return None

    qr = data.get("qr_camera")
    det = data.get("detection_camera")
    if qr is None and det is None:
        return None
    return qr, det


_camera_roles = _load_camera_roles()
if _camera_roles is not None:
    _qr_role, _det_role = _camera_roles
    if _qr_role is not None:
        QR_CAMERA_SOURCE = _qr_role
    if _det_role is not None:
        DETECTION_CAMERA_SOURCE = _det_role
    print(f"摄像头角色配置: 扫码={QR_CAMERA_SOURCE}  检测={DETECTION_CAMERA_SOURCE} "
          f"（来自 camera_roles.json）")


def _normalize_source(source):
    """把 '0'、'1' 这类字符串编号转成整数，其余原样返回（如 /dev/video0）"""
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


def _resolve_v4l2_capture(index):
    """把 OpenCV 设备编号解析成实际可采集的 /dev/video* 路径。

    UVC 摄像头会额外生成 metadata 节点（如 /dev/video1、/dev/video3），
    这些节点无法采集，也不能设置分辨率。这里按 /dev/video* 顺序探测
    可用采集节点，取第 index 个。
    """
    paths = sorted(
        glob.glob("/dev/video*"),
        key=lambda p: int(p.rsplit("video", 1)[1]),
    )
    captures = []
    for path in paths:
        probe = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if probe.isOpened():
            captures.append(path)
        probe.release()
    if 0 <= index < len(captures):
        return captures[index]
    return None


def open_camera(source=0, width=None, height=None, fps=None):
    """打开 USB 摄像头，返回 VideoCapture 对象；失败返回 None。

    source: 摄像头编号或 /dev/video* 路径
    width/height: 可选，请求的分辨率（不支持时会自动回退）
    fps: 可选，请求的帧率（摄像头不支持时会自动回退并告警）
    """
    source = _normalize_source(source)

    # 优先用 V4L2 后端打开；数字编号打不开时解析成实际采集节点，
    # 避免退回默认后端后设置分辨率不生效。
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            resolved = _resolve_v4l2_capture(source)
            if resolved is None:
                cap = cv2.VideoCapture(source)
            else:
                print(f"摄像头编号 {source} 直接打开失败，改用 {resolved}")
                cap = cv2.VideoCapture(resolved, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"无法打开摄像头 (source: {source})")
        return None

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
    print(f"摄像头已打开 (source: {source}, 实际分辨率 {actual_w}x{actual_h}, "
          f"帧率 {actual_fps:.2f}fps)")
    if fps is not None and actual_fps > 0 and abs(actual_fps - fps) > 1:
        print(f"警告: 请求 {fps}fps，摄像头实际生效 {actual_fps:.2f}fps，"
              f"请用 v4l2-ctl --list-formats-ext 确认该分辨率下支持的帧率档位")

    # 部分免驱摄像头需要先读一帧才会真正开始输出
    ret, _ = cap.read()
    if not ret:
        print(f"警告: 摄像头 {source} 首帧读取失败，程序仍会继续尝试")

    return cap
