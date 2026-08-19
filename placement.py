"""
放置阶段视觉识别：
   1. 检测同心黑色圆环；
   2. 取最小（最内层）圆环；
   3. 以最小圆环中心裁剪 ROI；
   4. 用 felling_number 的 MNIST 模型识别中心数字。
"""

import cv2
import numpy as np

from felling_number import load_mnist_model, preprocess_for_mnist, predict_digit


# 最小圆环内用于识别数字的裁剪比例（相对半径），画面不同可调整
DIGIT_CROP_RATIO = 0.7
MIN_RING_RADIUS = 10
MIN_RING_AREA = 300
RING_CIRCULARITY = 0.6
MIN_DIGIT_CONFIDENCE = 0.8


class PlacementRecognizer:
    """识别放置区同心圆环及其中心数字。"""

    def __init__(self, model_path="/home/xu/Engineer/tiny_digit_cnn.pth"):
        self.model, self.device = load_mnist_model(model_path)

    def find_rings(self, gray):
        """
        在灰度图中找出所有同心圆环组，返回每组最内层圆环：
        [{"center": (x, y), "radius": r, "contour": cnt}, ...]
        """
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_RING_AREA:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < RING_CIRCULARITY:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius < MIN_RING_RADIUS:
                continue
            candidates.append(
                {"center": (float(cx), float(cy)), "radius": float(radius), "contour": cnt}
            )

        if not candidates:
            return []

        # 同心圆环按中心距离分组，每组取半径最小的（最内层圆环）
        candidates.sort(key=lambda c: c["radius"])
        groups = []
        for cand in candidates:
            matched = False
            for group in groups:
                dx = cand["center"][0] - group["center"][0]
                dy = cand["center"][1] - group["center"][1]
                max_r = max(cand["radius"], group["rings"][0]["radius"])
                if dx * dx + dy * dy < (max_r * 0.8) ** 2:
                    group["rings"].append(cand)
                    matched = True
                    break
            if not matched:
                groups.append({"center": cand["center"], "rings": [cand]})

        result = []
        for group in groups:
            innermost = min(group["rings"], key=lambda c: c["radius"])
            result.append(innermost)
        return result

    def read_digit_at_ring(self, gray, ring):
        """
        在最小圆环中心裁剪 ROI，识别数字。
        返回 (digit, confidence, debug_canvas) 或 None。
        """
        cx, cy = ring["center"]
        radius = ring["radius"]
        crop_r = max(12, int(radius * DIGIT_CROP_RATIO))

        x0 = max(0, int(cx - crop_r))
        y0 = max(0, int(cy - crop_r))
        x1 = min(gray.shape[1], int(cx + crop_r))
        y1 = min(gray.shape[0], int(cy + crop_r))
        if x1 <= x0 or y1 <= y0:
            return None

        roi = gray[y0:y1, x0:x1]

        # 只保留中心圆形区域，排除圆环边界干扰
        mask = np.zeros_like(roi)
        cv2.circle(
            mask,
            (int(cx - x0), int(cy - y0)),
            crop_r,
            255,
            -1,
        )
        roi_masked = cv2.bitwise_and(roi, roi, mask=mask)

        tensor, _, debug_canvas = preprocess_for_mnist(roi_masked)
        if tensor is None:
            return None

        digit, confidence = predict_digit(self.model, tensor, self.device)
        return digit, float(confidence), debug_canvas

    def recognize(self, frame):
        """
        识别一帧放置区画面，返回所有可识别数字：
        [{"digit": n, "center": (x,y), "radius": r, "confidence": c}, ...]
        """
        if frame is None:
            return []
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        results = []
        for ring in self.find_rings(gray):
            recognized = self.read_digit_at_ring(gray, ring)
            if recognized is None:
                continue
            digit, confidence, _ = recognized
            if digit not in (1, 2, 3) or confidence < MIN_DIGIT_CONFIDENCE:
                continue
            results.append(
                {
                    "digit": digit,
                    "center": ring["center"],
                    "radius": ring["radius"],
                    "confidence": confidence,
                }
            )
        return results
