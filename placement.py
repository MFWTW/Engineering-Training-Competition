"""
放置阶段视觉识别：
   1. 检测同心黑色圆环；
   2. 取最小（最内层）圆环；
   3. 以最小圆环中心裁剪 ROI；
   4. 用 felling_number 的 MNIST 模型识别中心数字。
"""

import cv2
import numpy as np
import time
import yaml
from pathlib import Path

from felling_number import load_mnist_model, preprocess_for_mnist, predict_digit
from felling_color import CODE_TO_KEY, build_color_masks


CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _load_placement_cfg():
    """从 config.yaml 读取 placement 段；缺失/无该段时返回空字典。"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f).get("placement", {})


_PLACE_CFG = _load_placement_cfg()


def _place_float(key, default):
    return float(_PLACE_CFG.get(key, default))


# 放置识别参数（config.yaml → placement 段，改后重启生效）
MODEL_PATH = str(_PLACE_CFG.get("model_path", "/home/xu/Engineer/tiny_digit_cnn.pth"))
DIGIT_CROP_RATIO = _place_float("digit_crop_ratio", 0.7)      # 数字裁剪比例（相对半径）
MIN_RING_RADIUS = _place_float("min_ring_radius", 10)         # 圆环最小半径（px）
MIN_RING_AREA = _place_float("min_ring_area", 300)            # 圆环最小面积（px²）
RING_CIRCULARITY = _place_float("ring_circularity", 0.6)      # 圆环圆度下限
MIN_DIGIT_CONFIDENCE = _place_float("min_digit_confidence", 0.8)  # 数字置信度下限
RING_GROUP_OVERLAP = _place_float("ring_group_overlap", 0.8)  # 同心圆分组判定系数
MORPH_KERNEL_SIZE = int(_PLACE_CFG.get("morph_kernel_size", 5))  # 闭运算核大小
MIN_CROP_PX = _place_float("min_crop_px", 12)                 # 数字裁剪最小半径（px）
DEBUG = bool(_PLACE_CFG.get("debug", False))                  # true=打印识别调试日志
SHOW_DEBUG = bool(_PLACE_CFG.get("show_debug", False))        # true=弹窗显示处理过程

# 圆环数字被上一轮物块遮挡时，按物块颜色反推数字（config.yaml → placement 段）
OCCLUDED_DIGIT_BY_COLOR = bool(_PLACE_CFG.get("occluded_digit_by_color", True))
OCCLUDED_COLOR_CROP_RATIO = _place_float("occluded_color_crop_ratio", 0.6)
OCCLUDED_COLOR_MIN_AREA = _place_float("occluded_color_min_area", 150)
OCCLUDED_COLOR_AREA_MARGIN = _place_float("occluded_color_area_margin", 1.4)
OCCLUDED_COLOR_OVERRIDE_DIGIT = bool(_PLACE_CFG.get("occluded_color_override_digit", True))

_last_debug_print_t = 0.0


def _debug_throttle(interval_s=1.0):
    """调试日志节流：至少间隔 interval_s 秒打印一次。"""
    global _last_debug_print_t
    now = time.time()
    if now - _last_debug_print_t < interval_s:
        return False
    _last_debug_print_t = now
    return True


class PlacementRecognizer:
    """识别放置区同心圆环及其中心数字。"""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODEL_PATH
        self.model, self.device = load_mnist_model(model_path)
        # 3 分类模型输出索引 0/1/2，映射回业务数字 1/2/3
        self.class_labels = (1, 2, 3) if self.model.fc.out_features == 3 else None
        self.last_ring_debug = {
            "contours": 0,
            "area_rejected": 0,
            "circularity_rejected": 0,
            "radius_rejected": 0,
        }

    def find_rings(self, gray, roi=None):
        """
        在灰度图中找出所有同心圆环组，返回每组最内层圆环：
        [{"center": (x, y), "radius": r, "contour": cnt}, ...]

        roi: [x, y, w, h]，非 None 时只保留圆心在该区域内的圆环（坐标仍为全图坐标）。
        """
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
        )
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        if SHOW_DEBUG:
            # 保存二值化/闭运算结果，供 recognize() 弹窗显示
            self.last_binary = binary
            self.last_closed = closed

        contours, _ = cv2.findContours(
            closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        self.last_ring_debug = {
            "contours": len(contours),
            "area_rejected": 0,
            "circularity_rejected": 0,
            "radius_rejected": 0,
        }
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_RING_AREA:
                self.last_ring_debug["area_rejected"] += 1
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                self.last_ring_debug["circularity_rejected"] += 1
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < RING_CIRCULARITY:
                self.last_ring_debug["circularity_rejected"] += 1
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius < MIN_RING_RADIUS:
                self.last_ring_debug["radius_rejected"] += 1
                continue
            candidates.append(
                {"center": (float(cx), float(cy)), "radius": float(radius), "contour": cnt}
            )

        if not candidates:
            return []

        if roi is not None:
            rx, ry, rw, rh = (int(v) for v in roi)
            candidates = [
                c for c in candidates
                if rx <= c["center"][0] <= rx + rw
                and ry <= c["center"][1] <= ry + rh
            ]
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
                if dx * dx + dy * dy < (max_r * RING_GROUP_OVERLAP) ** 2:
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
        crop_r = max(int(MIN_CROP_PX), int(radius * DIGIT_CROP_RATIO))

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

        digit, confidence = predict_digit(
            self.model, tensor, self.device, class_labels=self.class_labels
        )
        if SHOW_DEBUG:
            self.last_digit_crop = roi_masked
            self.last_digit_canvas = debug_canvas
        return digit, float(confidence), debug_canvas

    def detect_ring_block_color(self, frame, ring, candidate_codes):
        """
        在圆环中心裁剪彩色 ROI，用 HSV 阈值识别圆环上物块颜色。

        candidate_codes: 待识别颜色代码列表（如 ["1","5","6"]）。

        返回 (color_code, area, confidence)：
          - color_code: 识别出的颜色代码；未识别到或区分度不足时为 None
          - area: 最佳颜色掩膜面积（px²）
          - confidence: 0~1，最佳面积相对次佳面积的领先度
        """
        if not candidate_codes or frame is None or len(frame.shape) != 3:
            return None, 0, 0.0
        cx, cy = ring["center"]
        radius = ring["radius"]
        crop_r = max(int(MIN_CROP_PX), int(radius * OCCLUDED_COLOR_CROP_RATIO))
        x0 = max(0, int(cx - crop_r))
        y0 = max(0, int(cy - crop_r))
        x1 = min(frame.shape[1], int(cx + crop_r))
        y1 = min(frame.shape[0], int(cy + crop_r))
        if x1 <= x0 or y1 <= y0:
            return None, 0, 0.0

        crop = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        keys = [CODE_TO_KEY[c] for c in candidate_codes if c in CODE_TO_KEY]
        if not keys:
            return None, 0, 0.0
        masks = build_color_masks(hsv, color_keys=keys)

        # 只统计圆环中心圆形区域内的颜色，排除圆环本身黑色边框的干扰
        circle_mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.circle(
            circle_mask,
            (int(cx - x0), int(cy - y0)),
            crop_r,
            255,
            -1,
        )
        areas = {}
        for code in candidate_codes:
            key = CODE_TO_KEY.get(code)
            if key is not None:
                masked = cv2.bitwise_and(
                    masks[key], masks[key], mask=circle_mask
                )
                areas[code] = int(cv2.countNonZero(masked))
        if not areas:
            return None, 0, 0.0

        best_code = max(areas, key=areas.get)
        best_area = areas[best_code]
        if best_area < OCCLUDED_COLOR_MIN_AREA:
            return None, 0, 0.0
        second_area = sorted(areas.values())[-2] if len(areas) > 1 else 0
        if best_area < second_area * OCCLUDED_COLOR_AREA_MARGIN:
            return None, 0, 0.0
        confidence = float(
            np.clip((best_area - second_area) / best_area, 0.0, 1.0)
        )
        return best_code, best_area, confidence

    def recognize_all(self, frame, roi=None, digit_of_color=None):
        """
        识别一帧放置区画面的所有圆环（含数字识别结果）。

        roi: [x, y, w, h]，非 None 时只识别圆心在该区域内的圆环。
        digit_of_color: 上一轮“物块颜色代码 → 圆环数字”映射；
            圆环数字被上一轮物块遮挡时，按圆环上物块颜色反推数字。

        返回:
            [{"center", "radius", "contour", "digit", "confidence"}, ...]
            digit: 数字 1~3 且置信度达标时为该数字，否则为 None
        """
        if frame is None:
            return []
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        rings = self.find_rings(gray, roi=roi)
        all_rings = []
        for ring in rings:
            item = dict(ring)
            item["digit"] = None
            item["confidence"] = 0.0
            item["raw_digit"] = None
            item["raw_confidence"] = 0.0
            recognized = self.read_digit_at_ring(gray, ring)
            if recognized is not None:
                digit, confidence, _ = recognized
                # 记录模型原始输出（不管是否通过），供调试查看
                item["raw_digit"] = digit
                item["raw_confidence"] = float(confidence)
                if digit in (1, 2, 3) and confidence >= MIN_DIGIT_CONFIDENCE:
                    item["digit"] = digit
                    item["confidence"] = float(confidence)

            # 数字被上一轮物块遮挡时，用圆环上物块颜色反推数字
            if digit_of_color and OCCLUDED_DIGIT_BY_COLOR:
                color_code, color_area, color_conf = self.detect_ring_block_color(
                    frame, ring, list(digit_of_color.keys())
                )
                if color_code is not None and color_code in digit_of_color:
                    item["color_code"] = color_code
                    item["color_area"] = color_area
                    item["color_conf"] = color_conf
                    item["inferred_by_color"] = True
                    inferred_digit = digit_of_color[color_code]
                    if OCCLUDED_COLOR_OVERRIDE_DIGIT or item["digit"] is None:
                        item["digit"] = inferred_digit
                        item["confidence"] = color_conf
            all_rings.append(item)

        if SHOW_DEBUG:
            # ---- 弹窗可视化处理过程 ----
            if hasattr(self, "last_binary"):
                cv2.imshow("placement_binary", self.last_binary)
            if hasattr(self, "last_closed"):
                cv2.imshow("placement_closed", self.last_closed)

            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for ring in rings:
                cx, cy = int(ring["center"][0]), int(ring["center"][1])
                r = int(ring["radius"])
                cv2.circle(vis, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
            cv2.imshow("placement_rings", vis)

            if hasattr(self, "last_digit_crop"):
                crop = self.last_digit_crop
                cv2.imshow("placement_digit_crop",
                           cv2.resize(crop, (280, 280),
                                      interpolation=cv2.INTER_NEAREST))
            if hasattr(self, "last_digit_canvas"):
                canvas = self.last_digit_canvas
                cv2.imshow("placement_digit_mnist",
                           cv2.resize(canvas, (280, 280),
                                      interpolation=cv2.INTER_NEAREST))

        if DEBUG and _debug_throttle(1.0):
            d = self.last_ring_debug
            digits_ok = [r for r in all_rings if r["digit"] is not None]
            if not all_rings and not d["contours"]:
                print("[放置调试] 画面里没有任何轮廓，先检查光照/对比度（圆环应是暗色、背景亮）")
            else:
                info = []
                for ring in all_rings:
                    area = cv2.contourArea(ring["contour"])
                    perimeter = cv2.arcLength(ring["contour"], True)
                    circ = (4.0 * np.pi * area / (perimeter * perimeter)
                            if perimeter > 0 else 0.0)
                    if ring["raw_digit"] is None:
                        digit_txt = "无数字轮廓"
                    elif ring.get("inferred_by_color"):
                        digit_txt = (f"颜色推断 {ring.get('color_code')}→"
                                     f"数字{ring['digit']} "
                                     f"面积{ring.get('color_area')} "
                                     f"置信度{ring['confidence']:.2f}")
                    elif ring["digit"] is not None:
                        digit_txt = (f"数字{ring['digit']} 通过 "
                                     f"置信度{ring['confidence']:.2f}")
                    else:
                        digit_txt = (f"预测{ring['raw_digit']} "
                                     f"置信度{ring['raw_confidence']:.2f} 未通过")
                    info.append(
                        f"(r={ring['radius']:.0f},A={area:.0f},C={circ:.2f},{digit_txt})"
                    )
                print(f"[放置调试] 找到{len(all_rings)}个圆环: "
                      f"{', '.join(info) if info else '无'}")
                if not all_rings:
                    print(f"[放置调试] 圆环被过滤: 轮廓={d['contours']}, "
                          f"面积淘汰={d['area_rejected']}, 圆度淘汰={d['circularity_rejected']}, "
                          f"半径淘汰={d['radius_rejected']}")
                elif not digits_ok:
                    print(f"[放置调试] 数字未通过: 需为1~3且置信度≥{MIN_DIGIT_CONFIDENCE:.2f} "
                          f"（检查数字是否完整在圆环中心、裁剪是否合适）")
        return all_rings

    def recognize(self, frame, roi=None):
        """兼容旧接口：只返回数字识别通过的圆环。"""
        return [
            r for r in self.recognize_all(frame, roi=roi)
            if r["digit"] is not None
        ]
