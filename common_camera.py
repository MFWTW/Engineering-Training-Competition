import cv2


# ==================== 两台 USB 免驱摄像头配置 ====================
# 这里统一配置摄像头来源：可以填 OpenCV 设备编号（0、1），
# 也可以填 /dev/video* 路径或 udev 符号链接（如 /dev/video_xia0）。
QR_CAMERA_SOURCE = 0          # 二维码扫描相机
DETECTION_CAMERA_SOURCE = 1   # 物块检测/放置识别相机

# 物块检测相机默认分辨率（USB 相机不支持时会自动使用其默认分辨率）。
# 如果换用支持更高分辨率的相机，请同时用同一分辨率重新标定相机内参。
DETECTION_FRAME_WIDTH = 640
DETECTION_FRAME_HEIGHT = 480


def _normalize_source(source):
    """把 '0'、'1' 这类字符串编号转成整数，其余原样返回（如 /dev/video0）"""
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


def open_camera(source=0, width=None, height=None):
    """打开 USB 摄像头，返回 VideoCapture 对象；失败返回 None。

    source: 摄像头编号或 /dev/video* 路径
    width/height: 可选，请求的分辨率（不支持时会自动回退）
    """
    source = _normalize_source(source)

    # 优先用 V4L2 后端打开，失败再退回默认后端
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

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头已打开 (source: {source}, 实际分辨率 {actual_w}x{actual_h})")

    # 部分免驱摄像头需要先读一帧才会真正开始输出
    ret, _ = cap.read()
    if not ret:
        print(f"警告: 摄像头 {source} 首帧读取失败，程序仍会继续尝试")

    return cap
