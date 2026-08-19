import cv2
from pyzbar.pyzbar import decode

session = []
groups = []


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
        cv2.imshow("QR Code Scanner", RAM_image)
        decode_data(data)
    else:
        cv2.imshow("QR Code Scanner", RAM_image)


def decode_data(data):
    numbers = data.split('+')
    groups.clear()
    session.clear()
    for num_group in numbers:
        groups.append(num_group)
        for digit in num_group:
            session.append(digit)

if __name__ == "__main__":
    scan_qrcode()
