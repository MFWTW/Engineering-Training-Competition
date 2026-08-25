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
# 物块检测相机目标帧率（fps）。要求摄像头支持该档位，否则 V4L2 会静默忽略，
# open_camera() 会打印实际生效的帧率并告警，程序仍可运行。
DETECTION_CAMERA_FPS = 30


def _normalize_source(source):
    """把 '0'、'1' 这类字符串编号转成整数，其余原样返回（如 /dev/video0）"""
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


def open_camera(source=0, width=None, height=None, fps=None):
    """打开 USB 摄像头，返回 VideoCapture 对象；失败返回 None。

    source: 摄像头编号或 /dev/video* 路径
    width/height: 可选，请求的分辨率（不支持时会自动回退）
    fps: 可选，请求的帧率（摄像头不支持时会自动回退并告警）
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
