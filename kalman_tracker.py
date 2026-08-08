"""
卡尔曼滤波物块追踪器 —— 6 维状态（位置+速度+加速度）
替代神经网络，实现：状态估计 → 去噪 → 速度/加速度推导 → 未来轨迹预测 → 拦截点解算
"""

import numpy as np
from collections import deque


class KalmanBlockTracker:
    """
    6 维卡尔曼滤波：X = [x, y, vx, vy, ax, ay]^T

    用法：
        kf = KalmanBlockTracker(dt=1/30)       # 30fps → dt≈0.033s
        kf.predict()                             # 每帧先预测
        kf.update(measured_x, measured_y)        # 用视觉测量更新
        x, y, vx, vy, ax, ay = kf.get_state()   # 获取平滑状态
        fx, fy = kf.predict_future(T=2.0)        # 预测T秒后的位置
        ix, iy = kf.compute_intercept(car_speed) # 解算拦截点
    """

    def __init__(self, dt: float = 1.0 / 30.0):
        """
        dt: 帧间时间间隔（秒），如 30fps → 1/30 ≈ 0.0333
        """
        self.dt = dt

        # ── 状态向量 X = [x, y, vx, vy, ax, ay] ──
        self.X = np.zeros((6, 1), dtype=np.float64)

        # ── 状态协方差矩阵 P ──
        self.P = np.eye(6, dtype=np.float64) * 100.0

        # ── 状态转移矩阵 F (6×6) ──
        self.F = np.eye(6, dtype=np.float64)
        self._update_F()

        # ── 观测矩阵 H (2×6)，只观测位置 ──
        self.H = np.zeros((2, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        # ── 过程噪声协方差 Q ──
        # 加速度的变化是主要噪声源
        q_acc = 5.0  # 加速度噪声 (px/s²)²
        dt2 = dt * dt / 2.0
        self.Q = np.zeros((6, 6), dtype=np.float64)
        # 位置受加速度噪声影响：0.5*a*dt²
        self.Q[0, 0] = q_acc * dt2 ** 2
        self.Q[1, 1] = q_acc * dt2 ** 2
        # 速度受加速度噪声影响：a*dt
        self.Q[2, 2] = q_acc * dt ** 2
        self.Q[3, 3] = q_acc * dt ** 2
        # 加速度自身
        self.Q[4, 4] = q_acc
        self.Q[5, 5] = q_acc

        # ── 测量噪声协方差 R ──
        # 视觉检测的抖动（像素²）
        meas_std = 8.0  # 测量标准差（像素）
        self.R = np.eye(2, dtype=np.float64) * (meas_std ** 2)

        # ── 身份矩阵 ──
        self.I = np.eye(6, dtype=np.float64)

        # 状态
        self.initialized = False
        self.last_update_time: float | None = None

        # 历史记录（调试用）
        self.history = deque(maxlen=100)

    def _update_F(self):
        """更新状态转移矩阵（dt 变化时调用）"""
        dt, dt2 = self.dt, self.dt * self.dt / 2.0
        self.F[0, 2] = dt        # x += vx * dt
        self.F[0, 4] = dt2       # x += 0.5 * ax * dt²
        self.F[1, 3] = dt        # y += vy * dt
        self.F[1, 5] = dt2       # y += 0.5 * ay * dt²
        self.F[2, 4] = dt        # vx += ax * dt
        self.F[3, 5] = dt        # vy += ay * dt

    def set_dt(self, dt: float):
        """更新帧间隔"""
        self.dt = dt
        self._update_F()

    # ==================== 核心：预测-更新 ====================

    def predict(self):
        """
        状态预测（Predict 步）
        X_pred = F * X
        P_pred = F * P * F^T + Q
        """
        self.X = self.F @ self.X
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measured_x: float, measured_y: float):
        """
        测量更新（Update 步）
        融合视觉测量值 Z = [measured_x, measured_y]^T

        1) 卡尔曼增益 K = P * H^T * (H * P * H^T + R)^-1
        2) X = X + K * (Z - H * X)
        3) P = (I - K * H) * P
        """
        Z = np.array([[measured_x], [measured_y]], dtype=np.float64)

        if not self.initialized:
            # 第一帧：直接用测量值初始化位置，速度加速度为0
            self.X[0, 0] = measured_x
            self.X[1, 0] = measured_y
            self.X[2, 0] = 0.0  # vx
            self.X[3, 0] = 0.0  # vy
            self.X[4, 0] = 0.0  # ax
            self.X[5, 0] = 0.0  # ay
            self.initialized = True
            self.last_update_time = None
            self.history.clear()
            return

        # ── 卡尔曼增益 ──
        S = self.H @ self.P @ self.H.T + self.R   # 2×2，创新协方差
        K = self.P @ self.H.T @ np.linalg.inv(S)   # 6×2

        # ── 状态更新 ──
        innovation = Z - self.H @ self.X           # 2×1，测量残差
        self.X = self.X + K @ innovation

        # ── 协方差更新 ──
        self.P = (self.I - K @ self.H) @ self.P

        # 记录
        self.history.append((self.X[0, 0], self.X[1, 0]))

    # ==================== 状态读取 ====================

    def get_state(self) -> tuple:
        """返回平滑后的状态: (x, y, vx, vy, ax, ay)"""
        return (
            float(self.X[0, 0]), float(self.X[1, 0]),
            float(self.X[2, 0]), float(self.X[3, 0]),
            float(self.X[4, 0]), float(self.X[5, 0]),
        )

    def get_position(self) -> tuple:
        """返回平滑位置: (x, y)"""
        return float(self.X[0, 0]), float(self.X[1, 0])

    def get_velocity(self) -> tuple:
        """返回速度: (vx, vy) px/s"""
        return float(self.X[2, 0]), float(self.X[3, 0])

    def get_speed(self) -> float:
        """返回速率: sqrt(vx²+vy²) px/s"""
        vx, vy = self.get_velocity()
        return float(np.sqrt(vx ** 2 + vy ** 2))

    # ==================== 未来轨迹预测 ====================

    def predict_future(self, T: float, steps: int = 1) -> list:
        """
        前向推算 T 秒后的位置（以及中间步）。
        x_future = x + vx*T + 0.5*ax*T²
        y_future = y + vy*T + 0.5*ay*T²

        Returns: [(x1,y1), ..., (xN,yN)] for N steps evenly spaced to T
        """
        x, y, vx, vy, ax, ay = self.get_state()
        result = []
        for i in range(1, steps + 1):
            t = T * i / steps
            fx = x + vx * t + 0.5 * ax * t ** 2
            fy = y + vy * t + 0.5 * ay * t ** 2
            result.append((fx, fy))
        return result

    # ==================== 持久化 ====================

    def get_state_dict(self) -> dict:
        """导出完整状态（用于保存）"""
        return {
            "X": self.X.copy(),
            "P": self.P.copy(),
            "dt": self.dt,
            "initialized": self.initialized,
        }

    def load_state_dict(self, d: dict):
        """恢复状态"""
        self.X = d["X"].copy()
        self.P = d["P"].copy()
        self.dt = d["dt"]
        self._update_F()
        self.initialized = d["initialized"]

    def reset(self):
        """重置滤波器"""
        self.X = np.zeros((6, 1), dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 100.0
        self.initialized = False
        self.last_update_time = None
        self.history.clear()


# ==================== 自测 ====================

if __name__ == "__main__":
    print("=== 卡尔曼滤波自测 ===\n")

    np.random.seed(42)
    kf = KalmanBlockTracker(dt=1.0 / 30.0)

    # 模拟真实运动：x 方向匀速 100px/s + 少量随机加速度
    true_x, true_y = 100.0, 200.0
    true_vx, true_vy = 100.0, -20.0
    true_ax, true_ay = 0.0, 0.0

    errors_raw = []
    errors_filtered = []

    for i in range(200):
        dt = 1.0 / 30.0
        # 真实运动
        true_ax = np.random.randn() * 5.0
        true_ay = np.random.randn() * 3.0
        true_vx += true_ax * dt
        true_vy += true_ay * dt
        true_x += true_vx * dt + 0.5 * true_ax * dt ** 2
        true_y += true_vy * dt + 0.5 * true_ay * dt ** 2

        # 模拟视觉测量（+ 8px 标准差高斯噪声）
        meas_x = true_x + np.random.randn() * 8.0
        meas_y = true_y + np.random.randn() * 8.0

        # 卡尔曼滤波
        kf.predict()
        kf.update(meas_x, meas_y)
        fx, fy, fvx, fvy, fax, fay = kf.get_state()

        if i >= 20:  # 跳过初始化阶段
            errors_raw.append(np.sqrt((meas_x - true_x) ** 2 + (meas_y - true_y) ** 2))
            errors_filtered.append(np.sqrt((fx - true_x) ** 2 + (fy - true_y) ** 2))

    print(f"测量噪声(原始):  平均误差 = {np.mean(errors_raw):.2f} px")
    print(f"卡尔曼滤波后:    平均误差 = {np.mean(errors_filtered):.2f} px")
    print(f"降噪比例:        {100 * (1 - np.mean(errors_filtered) / np.mean(errors_raw)):.1f}%")

    # 测试未来预测
    x, y, vx, vy, ax, ay = kf.get_state()
    print(f"\n当前状态: pos=({x:.1f}, {y:.1f}) vel=({vx:.1f}, {vy:.1f}) acc=({ax:.1f}, {ay:.1f})")
    print(f"真实速度:        ({true_vx:.1f}, {true_vy:.1f})")

    future = kf.predict_future(T=2.0, steps=5)
    print(f"\n预测未来2秒轨迹:")
    for i, (fx, fy) in enumerate(future):
        t = 2.0 * (i + 1) / 5
        print(f"  t={t:.1f}s: ({fx:.1f}, {fy:.1f})")

    print("\n=== 测试完成 ===")
