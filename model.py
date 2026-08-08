"""轻量级数字分类 CNN 模型 —— 极小参数量，高准确率"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyDigitCNN(nn.Module):
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


def count_parameters(model: nn.Module) -> int:
    """返回可训练参数总量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---- 快速自测 ----
if __name__ == "__main__":
    model = TinyDigitCNN()
    print(f"参数量: {count_parameters(model):,}")
    x = torch.randn(2, 1, 28, 28)
    y = model(x)
    print(f"输入: {x.shape} → 输出: {y.shape}")
