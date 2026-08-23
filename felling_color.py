import cv2
import os
import numpy as np
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path=CONFIG_PATH):
    """从 YAML 文件加载颜色检测配置"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到颜色检测配置文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config=None, path=CONFIG_PATH):
    """把配置写回 YAML 文件（供 hsv_tuner 等调试工具保存调参结果）"""
    if config is None:
        config = CONFIG
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True,
                       sort_keys=False, default_flow_style=False)


CONFIG = load_config()

# ==================== 颜色阈值配置（由 config.yaml 加载） ====================
_COLORS_CFG = CONFIG["colors"]
COLOR_KEYS = list(_COLORS_CFG["order"])

color_thresholds = {}
MORPH_PARAMS = {}
COLOR_CODE_MAP = {}
for _name in COLOR_KEYS:
    _cfg = _COLORS_CFG[_name]
    color_thresholds[_name] = {
        k: list(v) for k, v in _cfg.items()
        if k in ('lower', 'upper', 'lower1', 'upper1', 'lower2', 'upper2')
    }
    MORPH_PARAMS[_name] = (int(_cfg['morph']['erode']),
                           int(_cfg['morph']['dilate']))
    COLOR_CODE_MAP[_name] = str(_cfg['code'])

# 反向映射：颜色代码 → 颜色名（用于根据 QR 任务只检测目标颜色）
CODE_TO_KEY = {code: name for name, code in COLOR_CODE_MAP.items()}

_DET_CFG = CONFIG["detection"]


class BlockDetector:
    """物块颜色检测器 —— 封装所有检测状态，支持 reset 重置"""

    def __init__(self, detection_area=None, circle_params=None,
                 stability_settings=None, timeout_ms=None, kernel_size=None):
        # 默认值统一取自 config.yaml；显式传入的参数优先
        if detection_area is None:
            detection_area = _DET_CFG.get("detection_area")
        self.detection_area = detection_area  # [x, y, w, h]，None 表示全图
        # 霍夫圆参数
        if circle_params is None:
            circle_params = dict(_DET_CFG.get("circle", {}))
        self.circle_params = circle_params
        # 稳定性设置
        if stability_settings is None:
            stability_settings = dict(_DET_CFG.get("stability", {}))
        self.stability_settings = stability_settings
        if timeout_ms is None:
            timeout_ms = _DET_CFG.get("timeout_ms", 100)
        self.timeout_settings = {'timeout_ms': timeout_ms}
        if kernel_size is None:
            kernel_size = _DET_CFG.get("kernel_size", 3)
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # ---- 运行状态 ----
        self.reset()

    def reset(self):
        """重置所有检测状态（新一轮检测前调用）"""
        self.stable_count = 0
        self.color_stable_count = 0
        self.last_center = None
        self.last_color = None
        self.final_color = None
        self.last_radius = 0

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
        2. （若配置了 detection_area）只在该 ROI 内做 HSV 与轮廓筛选，
           排除画面边缘/自身车轮等误检区域；输出坐标映射回全图
        3. 找出目标颜色 → 建掩膜 → 轮廓筛选 → 真正的目标区域
        4. 对目标区域构建 ROI，画圆
        5. 输出 (formatted_data, center, color_code)

        Returns:
            (formatted_data, center, color_code)  或  (None, None, None)
        """
        h, w_full = frame.shape[:2]

        # ── 步骤 0：按配置裁剪检测区域（ROI 内检测，坐标后面映射回全图） ──
        roi = self.detection_area
        if roi is not None:
            rx, ry, rw, rh = (int(v) for v in roi)
            rx = max(0, min(rx, w_full - 1))
            ry = max(0, min(ry, h - 1))
            rw = max(0, min(rw, w_full - rx))
            rh = max(0, min(rh, h - ry))
            if rw <= 0 or rh <= 0:
                return None, None, None
            frame = frame[ry:ry + rh, rx:rx + rw]
        else:
            rx = ry = 0
            rw = w_full
            rh = h

        # ── 步骤 1：ROI 内灰度 + 高斯模糊 ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_cfg = _DET_CFG.get("blur", {})
        blur_k = int(blur_cfg.get("kernel", 5))
        gray = cv2.GaussianBlur(gray, (blur_k, blur_k),
                                float(blur_cfg.get("sigma_x", 2)),
                                float(blur_cfg.get("sigma_y", 2)))

        # ── 步骤 2：ROI 内 HSV 转换 ──
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ── 步骤 3：找出目标颜色 → 掩膜 → 轮廓 → 真正目标区域 ──

        # 3a) 构建颜色掩膜（仅 ROI 范围）
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
        min_area_factor = float(_DET_CFG.get("circle", {}).get("min_area_factor", 0.2))
        min_area_thresh = np.pi * (min_r ** 2) * min_area_factor

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
        # ROI 内坐标 → 全图坐标
        cx_full = cx + rx
        cy_full = cy + ry

        # 只要求圆心在 ROI 内。原“整个圆必须完整落在 ROI 内”的校验会
        # 因形态学膨胀使半径略大，导致物块还没出 ROI（尤其 ROI 上边在
        # y=0 时）就被误判为无效；抓取前 src.py 还有圆心在 ROI 内的门禁。
        if roi is not None:
            if not (rx <= cx_full <= rx + rw
                    and ry <= cy_full <= ry + rh):
                return None, None, None

        # ── 步骤 4：对目标区域画圆、判定颜色 ──

        # 4a) 在全图尺寸上构建圆形掩膜（用映射回全图的中心）
        roi_mask = np.zeros((h, w_full), dtype=np.uint8)
        cv2.circle(roi_mask, (int(cx_full), int(cy_full)), radius, 255, -1)

        # 4b) 在该圆内判定主颜色
        areas = {}
        for color_key in color_masks:
            areas[color_key] = cv2.countNonZero(
                cv2.bitwise_and(
                    color_masks[color_key],
                    roi_mask[ry:ry + rh, rx:rx + rw],
                )
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
        center = (int(cx_full), int(cy_full))
        formatted_data = f"{center[0]:04}{center[1]:04}{color_code}"
        return formatted_data, center, color_code

    def update_stability(self, current_center, current_color):
        """
        更新稳定性状态。应在 detect() 返回有效数据后调用。

        stable_count        —— 物块位置稳定计数：连续多少帧圆心位移 < max_pixel_move
        color_stable_count  —— 颜色稳定计数：连续多少帧识别为同一颜色
        两者互相独立，各自达到阈值后才算整体稳定。

        Returns:
            is_stable (bool): 位置与颜色两个稳定条件是否同时满足
            （src.py 会单独读取 stable_count / color_stable_count
             做两段式控制：颜色稳定先跟踪，位置稳定后抓取）
        """
        # ── 物块位置稳定计数：连续 N 帧圆心位移 < max_pixel_move ──
        if self.last_center is None:
            # 第一次检测到目标
            self.last_center = current_center
            self.stable_count = 1
        else:
            # 与上一帧位置比较，而不是与第一帧比较：
            # 目标持续移动时，与第一帧的偏差必然越来越大，
            # 旧逻辑会反复重置稳定计数，导致跟踪指令一直发不出去（表现很卡）。
            distance = np.sqrt(
                (current_center[0] - self.last_center[0]) ** 2 +
                (current_center[1] - self.last_center[1]) ** 2
            )
            if distance < self.stability_settings['max_pixel_move']:
                self.stable_count += 1
            else:
                # 位置跳变（目标切换/误检）：重置计数，
                # 同时保留当前帧作为新一轮的起点，加快重新锁定
                self.stable_count = 1

        self.last_center = current_center

        # ── 颜色稳定计数：连续同色才累计，与位置计数互相独立 ──
        if self.last_color is None or current_color != self.last_color:
            # 第一次检测到颜色 / 颜色跳变：重新开始计数，未确认的颜色作废
            self.color_stable_count = 1
            self.final_color = None
        else:
            self.color_stable_count += 1
        self.last_color = current_color

        if (self.color_stable_count >= self.stability_settings['color_stable_threshold']
                and self.final_color != current_color):
            self.final_color = current_color
            print(f"颜色识别稳定: {current_color} "
                  f"(连续 {self.color_stable_count} 帧)")

        return (self.stable_count >= self.stability_settings['threshold']
                and self.color_stable_count >= self.stability_settings['color_stable_threshold'])

    def on_miss(self):
        """检测缺失/颜色未命中时调用：颜色与位置计数都清零。

        位置计数也清零并清掉上一帧中心，避免识别闪烁时位置“稳定数”
        继续累积——必须连续、不间断地识别到目标，才允许抓取。
        """
        self.color_stable_count = 0
        self.last_color = None
        self.final_color = None
        self.stable_count = 0
        self.last_center = None

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
color_stable_count  = _default_detector.color_stable_count  # 注意：这是引用，reset 后会变
last_color          = _default_detector.last_color
final_color         = _default_detector.final_color
last_radius         = _default_detector.last_radius


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
