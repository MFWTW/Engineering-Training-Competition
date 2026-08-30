import os

import cv2
from pyzbar.pyzbar import decode

session = []
groups = []


def save_qr_result(data):
    """把扫码结果写入状态文件，供 qr_display.py 在外接屏幕上显示。"""
    path = os.environ.get("QR_DISPLAY_FILE", "/tmp/qr_display_result.txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)
    except OSError as exc:
        print(f"[QR] 写入屏幕显示文件失败 {path}: {exc}")


def scan_qrcode(thre_image, RAM_image):
    # pyzbar 解码
    barcodes = decode(thre_image)

    for barcode in barcodes:
        data = barcode.data.decode("utf-8")
        bbox = barcode.polygon

        # 绘制边框
        pts = [(p.x, p.y) for p in bbox]
        for i in range(len(pts)):
            cv2.line(RAM_image, pts[i], pts[(i + 1) % len(pts)], (0, 255, 0), 2)

        # 显示识别文字
        cv2.putText(RAM_image, data, (pts[0][0], pts[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    if barcodes and len(session) == 0:
        decode_data(data)


def decode_data(data):
    numbers = data.split('+')
    groups.clear()
    session.clear()
    for num_group in numbers:
        groups.append(num_group)
        for digit in num_group:
            session.append(digit)
    save_qr_result(data)

if __name__ == "__main__":
    scan_qrcode()
