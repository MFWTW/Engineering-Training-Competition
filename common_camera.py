import cv2


def open_camera(camera_id=0):
    """打开摄像头，返回 VideoCapture 对象"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"无法打开摄像头 (ID: {camera_id})")
        return None
    print(f"摄像头已打开 (ID: {camera_id})")
    return cap