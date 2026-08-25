"""
卡尔曼滤波物块追踪器 —— 6 维状态（位置+速度+加速度）
替代神经网络，实现：状态估计 → 去噪 → 速度/加速度推导 → 未来轨迹预测 → 拦截点解算

可选平台反馈（默认关闭）：
    下位机回传底盘 X/Y 加速度和夹爪加速度后，可把这些量当作“已知控制输入”
    接入预测步，补偿相机随底盘/夹爪运动造成的物块视运动，减小急加速/减速时的
    跟踪滞后。关闭时行为与原来的纯视觉卡尔曼完全一致。
"""

import numpy as np
from collections import deque
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _load_kalman_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["kalman"]


# 卡尔曼滤波调参变量统一从 config.yaml 读取
KALMAN_CFG = _load_kalman_config()


def _load_kalman_world_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("kalman_world", {})


# 世界系卡尔曼调参变量（config.yaml → kalman_world）
KALMAN_WORLD_CFG = _load_kalman_world_config()


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

    def __init__(self, dt: float = None, q_acc: float = None,
                 meas_std: float = None, initial_p: float = None,
                 history_len: int = None):
        """
        dt: 帧间时间间隔（秒），如 30fps → 1/30 ≈ 0.0333；
            未传时使用 config.yaml 中 kalman 段的默认值
        """
        self.dt = KALMAN_CFG["dt"] if dt is None else dt

        # ── 状态向量 X = [x, y, vx, vy, ax, ay] ──
        self.X = np.zeros((6, 1), dtype=np.float64)

        # ── 状态协方差矩阵 P ──
        init_p = KALMAN_CFG["initial_p"] if initial_p is None else initial_p
        self.P = np.eye(6, dtype=np.float64) * init_p

        # ── 状态转移矩阵 F (6×6) ──
        self.F = np.eye(6, dtype=np.float64)
        self._update_F()

        # ── 观测矩阵 H (2×6)，只观测位置 ──
        self.H = np.zeros((2, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        # ── 平台运动反馈（可选，默认关闭） ──
        pf = KALMAN_CFG.get("platform_feedback", {})
        self.platform_feedback_enabled = bool(pf.get("enabled", False))
        pf_px = pf.get("px_per_mm")
        if pf_px is None:
            # 未在 kalman 段配置时，跟随 chassis.px_per_mm（下位机 mm → 图像 px）
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                _full_cfg = yaml.safe_load(f)
            pf_px = _full_cfg.get("chassis", {}).get("px_per_mm", 1.0)
        self.platform_px_per_mm = float(pf_px)
        self.platform_gripper_axis = str(pf.get("gripper_axis", "y")).lower()
        # 平台视加速度 u（px/s²）：u = 相机运动造成的物块视加速度 = -(底盘+夹爪)加速度
        self.platform_u = np.zeros((2, 1), dtype=np.float64)
        # 最近一次原始回传值（调试用，当前只有加速度参与滤波）
        self.last_platform_feedback = {
            "chassis_ax_mm_s2": 0.0,
            "chassis_ay_mm_s2": 0.0,
            "gripper_pos_mm": 0.0,
            "gripper_vel_mm_s": 0.0,
            "gripper_acc_mm_s2": 0.0,
        }
        self._B = np.zeros((6, 2), dtype=np.float64)

        # ── 过程噪声协方差 Q ──
        # 加速度的变化是主要噪声源
        q_acc = KALMAN_CFG["q_acc"] if q_acc is None else q_acc
        dt2 = self.dt * self.dt / 2.0
        self.Q = np.zeros((6, 6), dtype=np.float64)
        # 位置受加速度噪声影响：0.5*a*dt²
        self.Q[0, 0] = q_acc * dt2 ** 2
        self.Q[1, 1] = q_acc * dt2 ** 2
        # 速度受加速度噪声影响：a*dt
        self.Q[2, 2] = q_acc * self.dt ** 2
        self.Q[3, 3] = q_acc * self.dt ** 2
        # 加速度自身
        self.Q[4, 4] = q_acc
        self.Q[5, 5] = q_acc

        # ── 测量噪声协方差 R ──
        # 视觉检测的抖动（像素²）
        meas_std = KALMAN_CFG["meas_std"] if meas_std is None else meas_std
        self.R = np.eye(2, dtype=np.float64) * (meas_std ** 2)

        # ── 身份矩阵 ──
        self.I = np.eye(6, dtype=np.float64)

        # 状态
        self.initialized = False
        self.last_update_time: float | None = None

        # 历史记录（调试用）
        hist_len = KALMAN_CFG["history_len"] if history_len is None else history_len
        self.history = deque(maxlen=hist_len)

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
        X_pred = F * X + B * u
        P_pred = F * P * F^T + Q

        u 为平台视加速度（px/s²），来自底盘/夹爪加速度反馈；
        平台反馈未启用时 u = 0，等价于原来的 X_pred = F * X。
        """
        # 控制输入矩阵 B（每个轴：位置 0.5*dt²，速度 dt）
        # 平台加速度 u 只补偿位置/速度，不写入加速度状态；
        # 状态里的加速度始终是“物块自身（非平台）”加速度，
        # 画面里的视加速度 = 状态加速度 + u，由 predict_future 统一叠加。
        dt, dt2 = self.dt, self.dt * self.dt / 2.0
        self._B[0, 0] = dt2
        self._B[1, 1] = dt2
        self._B[2, 0] = dt
        self._B[3, 1] = dt

        self.X = self.F @ self.X + self._B @ self.platform_u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def set_platform_feedback(self, chassis_ax_mm_s2=0.0, chassis_ay_mm_s2=0.0,
                              gripper_pos_mm=0.0, gripper_vel_mm_s=0.0,
                              gripper_acc_mm_s2=0.0, px_per_mm=None):
        """
        把下位机回传的底盘/夹爪运动反馈换算成平台视加速度 u（px/s²）。

        原理：相机装在底盘+夹爪上，平台加速时物块在画面里会产生相反的视运动；
        已知平台加速度后，预测步用 u 提前补偿，滤波就不需要靠测量慢慢“追”上。
        语义约定：状态里的 ax/ay 是物块自身（非平台）加速度；
        画面里的视加速度 = 状态加速度 + u，未来预测时会自动叠加。

        - chassis_ax/ay_mm_s2：底盘 X/Y 加速度（mm/s²）
        - gripper_pos_mm：夹爪当前位置（mm，当前只保存不参与滤波）
        - gripper_vel_mm_s：夹爪速度（mm/s，当前只保存不参与滤波）
        - gripper_acc_mm_s2：夹爪加速度（mm/s²，按 platform_gripper_axis 叠加）
        - px_per_mm：mm → px 换算；不传用 kalman.platform_feedback.px_per_mm
        """
        self.last_platform_feedback = {
            "chassis_ax_mm_s2": float(chassis_ax_mm_s2),
            "chassis_ay_mm_s2": float(chassis_ay_mm_s2),
            "gripper_pos_mm": float(gripper_pos_mm),
            "gripper_vel_mm_s": float(gripper_vel_mm_s),
            "gripper_acc_mm_s2": float(gripper_acc_mm_s2),
        }
        if not self.platform_feedback_enabled:
            self.platform_u = np.zeros((2, 1), dtype=np.float64)
            return

        if px_per_mm is None:
            px_per_mm = self.platform_px_per_mm
        px_per_mm = float(px_per_mm)

        # 视加速度 = -平台加速度（平台向前，物块在画面里向后）
        ux = -float(chassis_ax_mm_s2) * px_per_mm
        uy = -float(chassis_ay_mm_s2) * px_per_mm
        # 夹爪伸长方向（世界坐标一般为前进方向 y），映射到图像 x 或 y 轴
        g_acc_px = -float(gripper_acc_mm_s2) * px_per_mm
        if self.platform_gripper_axis == "x":
            ux += g_acc_px
        else:
            uy += g_acc_px
        self.platform_u = np.array([[ux], [uy]], dtype=np.float64)

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

    def predict_future(self, T: float = None, steps: int = None) -> list:
        """
        前向推算 T 秒后的位置（以及中间步）。
        x_future = x + vx*T + 0.5*ax*T²
        y_future = y + vy*T + 0.5*ay*T²

        Returns: [(x1,y1), ..., (xN,yN)] for N steps evenly spaced to T
        """
        if T is None:
            T = float(KALMAN_CFG["predict"]["horizon_s"])
        if steps is None:
            steps = int(KALMAN_CFG["predict"]["steps"])
        x, y, vx, vy, ax, ay = self.get_state()
        # 平台反馈启用时，物块的视加速度 = 状态加速度 + 平台视加速度 u
        # （假设预测时段内 u 保持最近一次回传值不变）
        ux = self.platform_u[0, 0] if self.platform_feedback_enabled else 0.0
        uy = self.platform_u[1, 0] if self.platform_feedback_enabled else 0.0
        ax += ux
        ay += uy
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
        self.P = np.eye(6, dtype=np.float64) * KALMAN_CFG["initial_p"]
        self.initialized = False
        self.last_update_time = None
        self.platform_u = np.zeros((2, 1), dtype=np.float64)
        self.history.clear()


class KalmanWorldTracker:
    """
    世界系（车中心坐标系）物块卡尔曼 —— X = [x, y, vx, vy, ax, ay]

    单位：
        x/y      物块相对车中心的位置（mm，正=左/前，与下位机指令同坐标系）
        vx/vy    物块相对地面的速度（mm/s，沿车中心系轴向）
        ax/ay    物块相对地面的加速度（mm/s²）

    测量：每帧把像素检测点经 pixel_to_camera + camera_to_world
          换算成车中心系坐标（mm）后喂入，速度/加速度由滤波器自行估计。

    控制输入（下位机回传，预测步补偿车自身运动）：
        ṙ = v_block - v_chassis
        r̈ = a_block - a_chassis
    所以底盘速度/加速度真正参与了卡尔曼预测。
    """

    def __init__(self, dt: float = None, q_acc: float = None,
                 meas_std: float = None, initial_p: float = None,
                 history_len: int = None):
        self.dt = float(KALMAN_WORLD_CFG.get("dt", 1.0 / 30.0)) if dt is None else dt

        # 状态 X = [x, y, vx, vy, ax, ay]（mm、mm/s、mm/s²）
        self.X = np.zeros((6, 1), dtype=np.float64)

        init_p = KALMAN_WORLD_CFG.get("initial_p", 100.0) if initial_p is None else initial_p
        self.P = np.eye(6, dtype=np.float64) * init_p

        self.F = np.eye(6, dtype=np.float64)
        self._update_F()

        # 观测矩阵：只观测车中心系位置 x/y（mm）
        self.H = np.zeros((2, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        # 过程噪声（单位 mm² 相关）；q_acc 单位 (mm/s²)²
        q_acc = KALMAN_WORLD_CFG.get("q_acc", 500.0) if q_acc is None else q_acc
        dt2 = self.dt * self.dt / 2.0
        self.Q = np.zeros((6, 6), dtype=np.float64)
        self.Q[0, 0] = q_acc * dt2 ** 2
        self.Q[1, 1] = q_acc * dt2 ** 2
        self.Q[2, 2] = q_acc * self.dt ** 2
        self.Q[3, 3] = q_acc * self.dt ** 2
        self.Q[4, 4] = q_acc
        self.Q[5, 5] = q_acc

        # 测量噪声（mm）
        meas_std = KALMAN_WORLD_CFG.get("meas_std", 2.0) if meas_std is None else meas_std
        self.R = np.eye(2, dtype=np.float64) * (meas_std ** 2)

        self.I = np.eye(6, dtype=np.float64)
        self.initialized = False
        self.last_update_time = None

        hist_len = KALMAN_WORLD_CFG.get("history_len", 100) if history_len is None else history_len
        self.history = deque(maxlen=hist_len)

    def _update_F(self):
        """状态转移（不含控制输入）"""
        dt, dt2 = self.dt, self.dt * self.dt / 2.0
        self.F[0, 2] = dt
        self.F[0, 4] = dt2
        self.F[1, 3] = dt
        self.F[1, 5] = dt2
        self.F[2, 4] = dt
        self.F[3, 5] = dt

    def set_dt(self, dt: float):
        self.dt = dt
        self._update_F()

    def predict(self, chassis_vx_mm_s: float = 0.0, chassis_vy_mm_s: float = 0.0,
                chassis_ax_mm_s2: float = 0.0, chassis_ay_mm_s2: float = 0.0):
        """
        预测步：X = F·X + B·u
        u = [底盘vx, 底盘vy, 底盘ax, 底盘ay]（下位机回传）
        车向前/向左运动时，物块相对车中心的位置反向变化。
        """
        dt, dt2 = self.dt, self.dt * self.dt / 2.0
        # 控制输入矩阵 B（6×4）：速度输入影响位置(-dt)、加速度输入影响位置(-dt²/2)和速度(-dt)
        self.B = np.zeros((6, 4), dtype=np.float64)
        self.B[0, 0] = -dt
        self.B[1, 1] = -dt
        self.B[0, 2] = -dt2
        self.B[1, 3] = -dt2
        self.B[2, 2] = -dt
        self.B[3, 3] = -dt
        u = np.array([
            [float(chassis_vx_mm_s)],
            [float(chassis_vy_mm_s)],
            [float(chassis_ax_mm_s2)],
            [float(chassis_ay_mm_s2)],
        ], dtype=np.float64)
        self.X = self.F @ self.X + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measured_x_mm: float, measured_y_mm: float):
        """测量更新：Z = [物块车中心系x(mm), 物块车中心系y(mm)]"""
        Z = np.array([[float(measured_x_mm)], [float(measured_y_mm)]], dtype=np.float64)
        if not self.initialized:
            self.X[0, 0] = float(measured_x_mm)
            self.X[1, 0] = float(measured_y_mm)
            self.X[2, 0] = 0.0
            self.X[3, 0] = 0.0
            self.X[4, 0] = 0.0
            self.X[5, 0] = 0.0
            self.initialized = True
            self.last_update_time = None
            self.history.clear()
            return

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        innovation = Z - self.H @ self.X
        self.X = self.X + K @ innovation
        self.P = (self.I - K @ self.H) @ self.P
        self.history.append((self.X[0, 0], self.X[1, 0]))

    def get_state(self) -> tuple:
        return (
            float(self.X[0, 0]), float(self.X[1, 0]),
            float(self.X[2, 0]), float(self.X[3, 0]),
            float(self.X[4, 0]), float(self.X[5, 0]),
        )

    def get_position(self) -> tuple:
        return float(self.X[0, 0]), float(self.X[1, 0])

    def get_velocity(self) -> tuple:
        return float(self.X[2, 0]), float(self.X[3, 0])

    def get_speed(self) -> float:
        vx, vy = self.get_velocity()
        return float(np.sqrt(vx ** 2 + vy ** 2))

    def predict_future(self, T: float = None, steps: int = None) -> list:
        """按物块自身运动外推（不叠加底盘运动，车会主动跟随）"""
        if T is None:
            T = float(KALMAN_WORLD_CFG.get("predict", {}).get("horizon_s", 2.0))
        if steps is None:
            steps = int(KALMAN_WORLD_CFG.get("predict", {}).get("steps", 6))
        x, y, vx, vy, ax, ay = self.get_state()
        result = []
        for i in range(1, steps + 1):
            t = T * i / steps
            result.append((x + vx * t + 0.5 * ax * t ** 2,
                           y + vy * t + 0.5 * ay * t ** 2))
        return result

    def reset(self):
        self.X = np.zeros((6, 1), dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * KALMAN_WORLD_CFG.get("initial_p", 100.0)
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
