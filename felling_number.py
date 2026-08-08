"""
MNIST 手写数字识别 —— 海康摄像头实时采集 + 模型推理
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import sys
import time

from hikrobot_camera import (
    enum_devices, create_camera_handle, start_grabbing, read_frame,
)


# ===================== 模型定义 =====================

class TinyCNN(nn.Module):
    """
    面向 MNIST 手写数字分类的轻量 CNN。
    设计目标：参数量 < 25K，准确率 99%+。
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.25):
        super().__init__()

        # ---- 卷积层：使用 depthwise-separable 思路进一步压缩 ----
        # Block 1: 1 → 8 channels
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        # Block 2: 8 → 16 channels
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)

        # Block 3: 16 → 32 channels (stride=2 下采样)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)

        # Block 4: 32 → 32 channels
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(32)

        # ---- 全局平均池化 → 避免大 FC 层 ----
        self.gap = nn.AdaptiveAvgPool2d(1)  # 输出 (B, 32, 1, 1)

        # ---- 分类头：极小的全连接层 ----
        self.fc = nn.Linear(32, num_classes, bias=True)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))  # 28x28
        x = F.relu(self.bn2(self.conv2(x)))  # 28x28
        x = F.max_pool2d(x, 2)  # 14x14
        x = F.relu(self.bn3(self.conv3(x)))  # 7x7  (stride=2)
        x = F.relu(self.bn4(self.conv4(x)))  # 7x7
        x = self.gap(x)  # 1x1
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



# ===================== 模型加载 =====================

def load_mnist_model(weight_path="/home/xu/Engineer/tiny_digit_cnn.pth"):
    """
    加载 MNIST 模型并返回 (model, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyCNN()
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"模型已加载，设备: {device}")
    return model, device


# ===================== 图像预处理 =====================

# MNIST 数据集的标准均值和标准差（训练时通常使用这些值做 Normalize）
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def preprocess_for_mnist(frame, target_size=28):
    """
    将摄像头拍摄的灰度图像预处理为 MNIST 模型输入。

    参数:
        frame: 灰度图 (H, W)，uint8（暗色数字 + 亮色背景）
        target_size: 输出尺寸，默认 28

    返回:
        tensor:  shape (1, 1, 28, 28)，MNIST 标准化
        roi:    提取的 ROI 图像（用于调试显示），可能为 None
        debug_canvas: 28x28 送入模型的图像（用于调试显示），可能为 None
    """
    if frame is None:
        return None, None, None

    # 1. 高斯模糊去噪
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # 2. OTSU 自适应二值化 → 白字黑底（模拟 MNIST）
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. 形态学闭运算：连接断裂笔画
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 4. 查找轮廓，定位数字区域
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    # 过滤太小或太大的轮廓（去噪）
    valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
    if not valid_contours:
        return None, None, None

    # 取面积最大的有效轮廓
    cnt = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # 扩展边界（保留边距，模拟 MNIST 的留白）
    margin = int(max(w, h) * 0.2)
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(closed.shape[1] - x, w + 2 * margin)
    h = min(closed.shape[0] - y, h + 2 * margin)

    # ★ 关键修复：从二值化图像（白字黑底）提取 ROI，而非原始灰度图
    roi = closed[y:y+h, x:x+w]
    roi_original = frame[y:y+h, x:x+w]  # 原始灰度 ROI（仅用于显示）

    # 5. 保持宽高比缩放至 20×20，放入 28×28 画布中央
    scale = 20.0 / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w < 1 or new_h < 1:
        return None, None, None
    roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 6. 创建 28x28 黑色画布，居中放置白字
    canvas = np.zeros((target_size, target_size), dtype=np.float32)
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = roi_resized.astype(np.float32)

    # 7. MNIST 标准化：(x/255 - mean) / std
    canvas /= 255.0
    canvas = (canvas - MNIST_MEAN) / MNIST_STD

    tensor = torch.from_numpy(canvas).float().unsqueeze(0).unsqueeze(0)

    # 8. 构建调试用的可视化（反标准化回 [0,255] 用于显示）
    debug_canvas = ((canvas * MNIST_STD + MNIST_MEAN) * 255).clip(0, 255).astype(np.uint8)

    return tensor, roi_original, debug_canvas


# ===================== 推理 =====================

def predict_digit(model, tensor, device):
    """
    对预处理后的 tensor 进行推理，返回预测数字和置信度。
    """
    with torch.no_grad():
        tensor = tensor.to(device)
        output = model(tensor)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1).item()
        confidence = prob[0, pred].item()
    return pred, confidence


# ===================== 海康摄像头实时识别 =====================

def hik_camera_recognition(model, device, cam_index=0, width=1440, height=1080):
    """
    海康摄像头实时采集 + MNIST 数字识别
    """
    # 1. 枚举并打开海康摄像头
    dev_list = enum_devices()
    if dev_list is None:
        print("未检测到海康摄像头！")
        return

    cam = create_camera_handle(dev_list, cam_index, width=width, height=height)
    if cam is None:
        return

    if not start_grabbing(cam):
        print("启动取流失败！")
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        return

    print("\n开始实时识别，按 'q' 退出，按 's' 截图保存\n")

    try:
        while True:
            # 2. 读取一帧
            frame = read_frame(cam)  # 灰度图 (Mono8) 或 BGR 图

            if frame is None:
                continue

            # 3. 若为彩色图则转为灰度
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # 4. 显示原始灰度图
            cv2.imshow("Gray", gray)

            # 5. 预处理
            tensor, roi, debug_canvas = preprocess_for_mnist(gray)

            # 6. 推理
            if tensor is not None:
                pred, conf = predict_digit(model, tensor, device)
                print(f"\r>>> 识别结果: {pred}  |  置信度: {conf:.3f}", end="")

                # 在画面上标注结果
                display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.putText(display, f"Pred: {pred} ({conf:.2f})",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

                # 显示 28x28 输入（放大显示，关键调试窗口）
                if debug_canvas is not None:
                    debug_disp = cv2.resize(debug_canvas, (280, 280), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("MNIST Input (28x28)", debug_disp)

                # 显示 ROI
                if roi is not None:
                    cv2.imshow("ROI", roi)
            else:
                display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.putText(display, "No digit found",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            cv2.imshow("Recognition", display)

            # 7. 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.png"
                cv2.imwrite(filename, gray)
                print(f"\n已保存截图: {filename}")

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        # 8. 清理资源
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        cv2.destroyAllWindows()
        print("资源已释放")


# ===================== 主入口 =====================

if __name__ == "__main__":
    model, device = load_mnist_model()
    hik_camera_recognition(model, device)

