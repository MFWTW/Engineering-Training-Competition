import cv2
import os
import numpy as np
from collections import defaultdict

# ==================== 颜色阈值配置 ====================
color_thresholds = {
    'red': {'lower1': [0, 43, 46], 'upper1': [10, 255, 255],
                'lower2': [156, 43, 46], 'upper2': [180, 255, 255]},
    'green':      {'lower':  [69, 127, 54],  'upper':  [92, 255, 158]},
    'blue':       {'lower':  [110, 138, 87], 'upper':  [122, 255, 155]},
    'light_blue': {'lower':  [94, 134, 162], 'upper':  [123, 218, 255]},
    'black':      {'lower':  [0, 0, 0],      'upper':  [180, 255, 33]},
    'yellow':     {'lower':  [22, 78, 30],   'upper':  [42, 255, 255]},
}

# 颜色代码 -> 索引映射（用于 max 比较）
COLOR_KEYS = ['red', 'green', 'blue', 'light_blue', 'black', 'yellow']
COLOR_CODE_MAP = {'red': '1', 'green': '2', 'blue': '3',
                  'light_blue': '4', 'black': '5', 'yellow': '6'}

# 反向映射：颜色代码 → 颜色名（用于根据 QR 任务只检测目标颜色）
CODE_TO_KEY = {'1': 'red', '2': 'green', '3': 'blue',
               '4': 'light_blue', '5': 'black', '6': 'yellow'}

# 每种颜色的形态学参数：(erode_iter, dilate_iter)
MORPH_PARAMS = {
    'red':        (1, 2),
    'green':      (2, 6),
    'blue':       (1, 3),
    'light_blue': (1, 3),
    'black':      (1, 3),
    'yellow':     (1, 3),
}


class BlockDetector:
    """物块颜色检测器 —— 封装所有检测状态，支持 reset 重置"""

    def __init__(self, detection_area=None, circle_params=None,
                 stability_settings=None, timeout_ms=100, kernel_size=3):
        # 检测区域 [x, y, w, h]，None 表示全图
        self.detection_area = detection_area
        # 霍夫圆参数
        self.circle_params = circle_params or {
            'min_radius': 50, 'max_radius': 450, 'param1': 25, 'param2': 25
        }
        # 稳定性设置
        self.stability_settings = stability_settings or {
            'threshold': 30,
            'max_pixel_move': 10,
            'color_stable_threshold': 15,
            'color_confidence': 0.7
        }
        self.timeout_settings = {'timeout_ms': timeout_ms}
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # ---- 运行状态 ----
        self.reset()

    def reset(self):
        """重置所有检测状态（新一轮检测前调用）"""
        self.stable_count = 0
        self.first_center = None
        self.last_detection_time = None
        self.color_history = []
        self.final_color = None
        self.last_radius = 0
        self.last_color_code = None
        self.detection_sent = False  # 防止重复发送

    def _build_color_masks(self, hsv_img, target_key=None):
        """构建颜色掩膜。target_key 不为 None 时只构建目标颜色掩膜"""
        masks = {}
        keys = [target_key] if target_key else COLOR_KEYS
        for color_key in keys:
            cfg = color_thresholds[color_key]
            if 'lower1' in cfg:  # 双阈值（红色）
                m1 = cv2.inRange(hsv_img,
                                 np.array(cfg['lower1']), np.array(cfg['upper1']))
                m2 = cv2.inRange(hsv_img,
                                 np.array(cfg['lower2']), np.array(cfg['upper2']))
                mask = cv2.bitwise_or(m1, m2)
            else:
                mask = cv2.inRange(hsv_img,
                                   np.array(cfg['lower']), np.array(cfg['upper']))
            # 形态学
            e_iter, d_iter = MORPH_PARAMS[color_key]
            mask = cv2.erode(mask, self.kernel, iterations=e_iter)
            mask = cv2.dilate(mask, self.kernel, iterations=d_iter)
            masks[color_key] = mask
        return masks

    def detect(self, frame, target_code=None):
        """
        1. 全图做灰度和高斯模糊
        2. HSV 转换
        3. 找出目标颜色 → 建掩膜 → 轮廓筛选 → 真正的目标区域
        4. 对目标区域构建 ROI，画圆
        5. 输出 (formatted_data, center, color_code)

        Returns:
            (formatted_data, center, color_code)  或  (None, None, None)
        """
        h, w_full = frame.shape[:2]

        # ── 步骤 1：全图灰度 + 高斯模糊 ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 2, 2)

        # ── 步骤 2：全图 HSV 转换 ──
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ── 步骤 3：找出目标颜色 → 掩膜 → 轮廓 → 真正目标区域 ──

        # 3a) 构建颜色掩膜（全图范围，不先用 ROI 裁剪）
        if target_code is not None and target_code in CODE_TO_KEY:
            target_key = CODE_TO_KEY[target_code]
            color_masks = self._build_color_masks(hsv_img, target_key=target_key)
        else:
            color_masks = self._build_color_masks(hsv_img)

        # 3b) 合并掩膜 → 候选区域
        combined_mask = None
        for m in color_masks.values():
            if combined_mask is None:
                combined_mask = m.copy()
            else:
                combined_mask = cv2.bitwise_or(combined_mask, m)

        if combined_mask is None:
            return None, None, None

        # 3c) 轮廓筛选：找到真正目标
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None

        # 面积门限
        min_r = max(1, int(self.circle_params.get('min_radius', 30)))
        min_area_thresh = np.pi * (min_r ** 2) * 0.2

        best_candidate = None
        best_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_thresh:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            radius = int(radius)
            if radius < min_r or radius > self.circle_params.get('max_radius', 160):
                continue

            # 记录面积最大的候选
            if area > best_area:
                best_area = area
                best_candidate = (cx, cy, radius, cnt)

        if best_candidate is None:
            return None, None, None

        cx, cy, radius, cnt = best_candidate

        # ── 步骤 4：对目标区域（ROI）画圆、判定颜色 ──

        # 4a) 在全图尺寸上构建圆形掩膜
        roi_mask = np.zeros((h, w_full), dtype=np.uint8)
        cv2.circle(roi_mask, (int(cx), int(cy)), radius, 255, -1)

        # 4b) 在该圆内判定主颜色
        areas = {}
        for color_key in color_masks:
            areas[color_key] = cv2.countNonZero(
                cv2.bitwise_and(color_masks[color_key], roi_mask)
            )

        if not areas:
            return None, None, None

        best_color_key = max(areas, key=areas.get)
        if areas[best_color_key] == 0:
            return None, None, None

        color_code = COLOR_CODE_MAP[best_color_key]

        # 目标颜色过滤
        if target_code is not None and color_code != target_code:
            return None, None, None

        # ── 步骤 5：输出 ──
        self.last_radius = radius
        self.last_color_code = color_code

        center = (int(cx), int(cy))
        formatted_data = f"{center[0]:04}{center[1]:04}{color_code}"
        return formatted_data, center, color_code

    def update_stability(self, current_center, current_color):
        """
        更新稳定性状态。应在 detect() 返回有效数据后调用。

        Returns:
            is_stable (bool): 是否达到稳定条件
        """
        self.color_history.append(current_color)

        if self.first_center is None:
            self.first_center = current_center
            self.stable_count = 1
        else:
            distance = np.sqrt(
                (current_center[0] - self.first_center[0]) ** 2 +
                (current_center[1] - self.first_center[1]) ** 2
            )
            if distance < self.stability_settings['max_pixel_move']:
                self.stable_count += 1
            else:
                self.stable_count = 1
                self.first_center = current_center
                self.color_history = []

        # 颜色置信度检查
        if len(self.color_history) >= self.stability_settings['color_stable_threshold']:
            counter = defaultdict(int)
            for c in self.color_history:
                counter[c] += 1
            most_common = max(counter.items(), key=lambda x: x[1])[0]
            confidence = counter[most_common] / len(self.color_history)
            if confidence >= self.stability_settings['color_confidence']:
                self.final_color = most_common
                print(f"颜色识别稳定: {most_common} (置信度: {confidence:.2f})")

        return (self.stable_count >= self.stability_settings['threshold']
                and self.final_color is not None)

    def get_result_data(self, center):
        """稳定后获取格式化发送数据 (9字符: XXXXYYYYC)"""
        return f"{center[0]:04}{center[1]:04}{self.final_color}"


# ==================== 向后兼容：模块级默认实例（全图模式） ====================
_default_detector = BlockDetector(detection_area=None)

# 将属性暴露为模块变量（方便 src.py 中 import felling_color 后直接访问）
detection_area      = _default_detector.detection_area
circle_params       = _default_detector.circle_params
stability_settings  = _default_detector.stability_settings
timeout_settings    = _default_detector.timeout_settings
kernel              = _default_detector.kernel
stable_count        = _default_detector.stable_count        # 注意：这是引用，reset 后会变
first_center        = _default_detector.first_center
last_detection_time = _default_detector.last_detection_time
color_history       = _default_detector.color_history
final_color         = _default_detector.final_color
last_radius         = _default_detector.last_radius
last_color_code     = _default_detector.last_color_code


def block_preprocessing(frame, target=None):
    """向后兼容：直接委托给默认检测器。target 为目标颜色代码如 "1" """
    return _default_detector.detect(frame, target_code=target)


def reset_detector():
    """重置默认检测器状态"""
    _default_detector.reset()


def get_detector():
    """获取默认检测器实例"""
    return _default_detector


# ==================== HSV 取色工具 ====================
def hsv_picker(image_path):
    """点击图像查看 HSV 值（调试用）"""
    img = cv2.imread(image_path)
    if img is None:
        print("图片读取失败")
        return
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            h, s, v = hsv[y, x]
            b, g, r = img[y, x]
            print(f"({x},{y}) BGR=({b},{g},{r}) HSV=({h},{s},{v})")

    win_name = "HSV Picker"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, hsv)
    cv2.waitKey(100)
    cv2.setMouseCallback(win_name, on_mouse)

    print("点击图像区域查看 HSV，按 ESC 退出")
    while cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) >= 1:
        if cv2.waitKey(100) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    block_path = r"/home/xu/Engineer/block"
    block_list = [os.path.join(block_path, f) for f in os.listdir(block_path)]
    print(block_list[3] if len(block_list) > 3 else "不足4张图片")

    detector = BlockDetector()
    result = detector.detect(cv2.imread(block_list[3]))
    print(f"检测结果: {result}")
