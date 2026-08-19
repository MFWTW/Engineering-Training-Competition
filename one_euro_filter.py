"""
一欧元低通滤波器（One Euro Filter）—— 2D 位置平滑 + 速度估计

原理：
    普通低通滤波在“平滑”和“滞后”之间只能二选一；
    一欧元滤波会根据当前目标速度动态调整截止频率：
      - 目标慢 / 静止：用低截止频率，大力平滑，去掉检测抖动；
      - 目标快：自动提高截止频率，减少滞后，保证跟手。

    本文件提供：
      - OneEuroFilter    ：单轴滤波器（标准实现）
      - OneEuroTracker2D ：两轴封装，供主程序直接使用
                           （位置 + 速度 + 历史 + 匀速外推）
        速度用最近 N 帧做最小二乘线性拟合，比直接取导数稳定得多，
        避免“检测抖动 → 速度乱跳 → 拦截点乱跳”的连锁反应。

参数（config.yaml → one_euro）：
    min_cutoff : 最小截止频率（Hz），越小越平滑；太小时目标静止也会显得“拖”。
    beta       : 速度自适应系数，越大高速时越跟手；0 时退化为固定截止频率低通。
    d_cutoff   : 速度估计的截止频率（Hz），用于平滑导数，通常取 0.5~2.0。
    velocity_window : 速度拟合窗口（帧），越大速度越稳，但变向时反应越慢。
    dt_min_s / dt_max_s：帧间隔裁剪范围，防止帧率抖动或暂停导致 dt 异常。
"""

import math
from collections import deque
from typing import Deque, List, Optional, Tuple


class OneEuroFilter:
    """单轴一欧元低通滤波器"""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self):
        """清空内部状态（切换目标 / 重新开始跟踪时调用）"""
        self.initialized = False
        self.value_hat = 0.0
        self.value_prev = 0.0
        self.dx_hat = 0.0

    @staticmethod
    def _smoothing_factor(cutoff: float, dt: float) -> float:
        """由截止频率计算低通系数 alpha"""
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-4))

    def filter(self, value: float, dt: float) -> Tuple[float, float]:
        """
        输入一帧原始值，返回 (平滑值, 平滑后的一阶导数)。
        dt 为距离上一帧的时间（秒）。
        """
        dt = max(float(dt), 1e-4)
        value = float(value)

        if not self.initialized:
            self.value_hat = value
            self.value_prev = value
            self.dx_hat = 0.0
            self.initialized = True
            return self.value_hat, self.dx_hat

        # 1) 原始导数 → 低通平滑
        dx = (value - self.value_prev) / dt
        alpha_d = self._smoothing_factor(self.d_cutoff, dt)
        self.dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_hat

        # 2) 截止频率随速度自适应：越快越跟手
        cutoff = self.min_cutoff + self.beta * abs(self.dx_hat)
        alpha = self._smoothing_factor(cutoff, dt)
        self.value_hat = alpha * value + (1.0 - alpha) * self.value_hat
        self.value_prev = value

        return self.value_hat, self.dx_hat


class OneEuroTracker2D:
    """
    两轴一欧元跟踪器：平滑 (x, y)，并输出滤波后的速度 (px/s)。
    主程序用它替代/对比卡尔曼滤波，只需要调用 update() 和 reset()。
    """

    def __init__(
        self,
        min_cutoff: float = 0.8,
        beta: float = 0.05,
        d_cutoff: float = 0.4,
        dt_min: float = 0.005,
        dt_max: float = 0.2,
        velocity_window: int = 12,
        history_len: int = 100,
    ):
        self.fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        self.velocity_window = max(3, int(velocity_window))
        self.history = deque(maxlen=int(history_len))
        self._velocity_samples: Deque[Tuple[float, float, float]] = deque(
            maxlen=self.velocity_window
        )
        self.last_time: Optional[float] = None
        self.last_dt = 1.0 / 30.0

    def reset(self):
        """清空滤波状态与历史（切换目标时调用）"""
        self.fx.reset()
        self.fy.reset()
        self.history.clear()
        self._velocity_samples.clear()
        self.last_time = None
        self.last_dt = 1.0 / 30.0

    def update(
        self,
        x: float,
        y: float,
        timestamp: Optional[float] = None,
    ) -> Tuple[float, float, float, float]:
        """
        输入原始检测中心，返回 (sx, sy, vx, vy)。
        timestamp 传 frame_t（秒），内部按实际帧间隔计算 dt；
        不传则沿用上一次的 dt。
        """
        if timestamp is not None:
            now = float(timestamp)
            if self.last_time is not None:
                self.last_dt = min(
                    max(now - self.last_time, self.dt_min), self.dt_max
                )
            self.last_time = now

        dt = self.last_dt
        sx, vx = self.fx.filter(x, dt)
        sy, vy = self.fy.filter(y, dt)
        t_now = self.last_time if self.last_time is not None else 0.0
        self._velocity_samples.append((t_now, sx, sy))
        self.history.append((sx, sy))
        rx, ry = self._fit_velocity()
        return sx, sy, rx, ry

    def get_position(self) -> Tuple[float, float]:
        """返回平滑后的位置 (x, y)"""
        return self.fx.value_hat, self.fy.value_hat

    def get_velocity(self) -> Tuple[float, float]:
        """返回最近 velocity_window 帧最小二乘拟合的速度 (vx, vy)，单位 px/s"""
        return self._fit_velocity()

    def _fit_velocity(self) -> Tuple[float, float]:
        """对最近 N 个滤波位置做线性拟合，返回斜率 (vx, vy)"""
        n = len(self._velocity_samples)
        if n < 3:
            return self.fx.dx_hat, self.fy.dx_hat

        ts = [p[0] for p in self._velocity_samples]
        xs = [p[1] for p in self._velocity_samples]
        ys = [p[2] for p in self._velocity_samples]

        t_mean = sum(ts) / n
        denom = sum((t - t_mean) ** 2 for t in ts)
        if denom < 1e-9:
            return self.fx.dx_hat, self.fy.dx_hat

        vx = sum((t - t_mean) * (x - sum(xs) / n) for t, x in zip(ts, xs)) / denom
        vy = sum((t - t_mean) * (y - sum(ys) / n) for t, y in zip(ts, ys)) / denom
        return vx, vy

    def get_speed(self) -> float:
        """返回速率（px/s）"""
        vx, vy = self.get_velocity()
        return math.hypot(vx, vy)

    def predict_future(self, T: float = 2.0, steps: int = 6) -> List[Tuple[float, float]]:
        """按当前滤波速度做匀速外推（一欧元本身不估计加速度）"""
        x, y = self.get_position()
        vx, vy = self.get_velocity()
        steps = max(1, int(steps))
        return [
            (x + vx * T * i / steps, y + vy * T * i / steps)
            for i in range(1, steps + 1)
        ]


# ==================== 自测 ====================
if __name__ == "__main__":
    import random

    print("=== 一欧元低通滤波自测 ===\n")

    # 场景1：静止目标 + 5px 抖动，看平滑后抖动是否明显下降
    f = OneEuroTracker2D(min_cutoff=0.8, beta=0.05, d_cutoff=0.4)
    raw_jitter = 0.0
    sm_jitter = 0.0
    for i in range(200):
        rx = 320.0 + random.gauss(0, 5)
        ry = 240.0 + random.gauss(0, 5)
        sx, sy, _, _ = f.update(rx, ry, timestamp=i / 30.0)
        if i >= 30:
            raw_jitter += abs(rx - 320.0)
            sm_jitter += abs(sx - 320.0)
    n = 170
    print(f"静止+5px抖动: 原始平均抖动 = {raw_jitter / n:.2f}px -> "
          f"滤波后 = {sm_jitter / n:.2f}px")

    # 场景2：匀速移动目标，看滤波后位置是否跟得上、速度估计是否正确
    f = OneEuroTracker2D(min_cutoff=0.8, beta=0.05, d_cutoff=0.4)
    x = 100.0
    for i in range(120):
        x += 100.0 / 30.0
        sx, sy, vx, vy = f.update(x + random.gauss(0, 4), 200.0, timestamp=i / 30.0)
    print(f"100px/s 匀速: 滤波位置={sx:.1f} 速度估计=({vx:.1f}, {vy:.1f})px/s")

    print("\n=== 测试完成 ===")
