"""
拦截点动态规划器 —— 博弈 T 求解可行拦截点

输入：KF 滤波后的物块状态 + 小车当前位姿/速度
输出：可行拦截点坐标 + 所需 T，供路径规划（TEB / 纯追踪）使用

原理：
  1) 假设预测时间 T，算物块 T 秒后位置 P_target
  2) 计算小车到达 P_target 所需时间 T_car
  3) 如果 T_car ≈ T → 可行；如果 T_car ≫ T → 增大 T 重算
  4) 输出匹配的 (x, y, T)
"""

import numpy as np
from typing import Optional


class InterceptPlanner:
    """
    动态拦截规划器

    car_max_speed:  小车最大速度 (px/s，图像坐标系)
    car_accel:      小车加速度 (px/s²)
    car_decel:      小车减速度 (px/s²)，用于计算停止距离
    """

    def __init__(
        self,
        car_max_speed: float = 200.0,
        car_accel: float = 100.0,
        car_decel: float = 150.0,
        time_resolution: float = 0.05,      # T 搜索步长（秒）
        max_predict_time: float = 5.0,       # 最大预测时间
        tolerance: float = 0.1,              # |T_car - T| 收敛容差（秒）
    ):
        self.car_max_speed = car_max_speed
        self.car_accel = car_accel
        self.car_decel = car_decel
        self.dt = time_resolution
        self.T_max = max_predict_time
        self.tol = tolerance

    def _car_travel_time(
        self,
        target_x: float,
        target_y: float,
        car_x: float,
        car_y: float,
        car_v: float,      # 当前速率
    ) -> float:
        """
        估算小车从 (car_x, car_y) 以当前速率 car_v 到达 target 的最短时间。

        简化模型：加速→匀速→减速，计算总时间。
        若距离很近则只用减速段。
        """
        dist = np.sqrt((target_x - car_x) ** 2 + (target_y - car_y) ** 2)
        if dist < 1e-6:
            return 0.0

        # 三段式时间估计（梯形速度曲线）
        # 加速段时间 t1，匀速段时间 t2，减速段时间 t3
        v_max = min(self.car_max_speed, car_v + self.car_accel * 10.0)

        # 加速到 v_max 的距离
        d_accel = (v_max ** 2 - car_v ** 2) / (2 * self.car_accel) if self.car_accel > 0 else 0.0
        if d_accel < 0:
            d_accel = 0.0

        # 从 v_max 减速到 0 的距离
        d_decel = (v_max ** 2) / (2 * self.car_decel) if self.car_decel > 0 else 0.0

        if d_accel + d_decel >= dist:
            # 距离太短，无法加速到 v_max，直接用三角形速度曲线
            # 加速段 + 减速段，中间无匀速
            # 解: a*t_a = d*t_d, v_end = a*t_a, 总距离 = 0.5*a*t_a² + 0.5*d*t_d²
            # 简化：加速到某个速度后立即减速
            v_peak = np.sqrt(
                (2 * self.car_accel * self.car_decel * dist) / (self.car_accel + self.car_decel)
            )
            t_accel = v_peak / self.car_accel if self.car_accel > 0 else dist / car_v if car_v > 0 else 999
            t_decel = v_peak / self.car_decel if self.car_decel > 0 else 0.0
            return t_accel + t_decel
        else:
            # 有匀速段
            d_cruise = dist - d_accel - d_decel
            t_accel = (v_max - car_v) / self.car_accel if self.car_accel > 0 else 0.0
            t_cruise = d_cruise / v_max if v_max > 0 else 999
            t_decel = v_max / self.car_decel if self.car_decel > 0 else 0.0
            return t_accel + t_cruise + t_decel

    # ==================== 核心：博弈 T 搜索 ====================

    def solve(
        self,
        block_x: float,         # 物块当前位置 (KF)
        block_y: float,
        block_vx: float,        # 物块当前速度 (KF)
        block_vy: float,
        block_ax: float = 0.0,  # 物块当前加速度 (KF)
        block_ay: float = 0.0,
        car_x: float = 0.0,     # 小车当前位置
        car_y: float = 0.0,
        car_v: float = 0.0,     # 小车当前速率 (px/s)
    ) -> Optional[dict]:
        """
        搜索可行的拦截 T。

        Returns:
            None  → 无法拦截（物块太快/太远）
            dict  → {
                "x": float,       拦截点 X
                "y": float,       拦截点 Y
                "T": float,       预测时间（秒）
                "T_car": float,   小车到达时间（秒）
                "feasible": bool, 是否可行
            }
        """
        best_result = None
        best_diff = float("inf")

        T = 0.1
        while T <= self.T_max:
            # 1) 物块 T 秒后位置
            fx = block_x + block_vx * T + 0.5 * block_ax * T ** 2
            fy = block_y + block_vy * T + 0.5 * block_ay * T ** 2

            # 2) 小车到达该点的时间
            T_car = self._car_travel_time(fx, fy, car_x, car_y, car_v)

            # 3) 判断差距
            diff = abs(T_car - T)

            if diff < best_diff:
                best_diff = diff
                best_result = {
                    "x": fx,
                    "y": fy,
                    "T": T,
                    "T_car": T_car,
                    "feasible": diff <= self.tol,
                }

            # 提前收敛
            if diff <= self.tol:
                break

            T += self.dt

        return best_result

    def solve_with_path(
        self,
        block_x: float,
        block_y: float,
        block_vx: float,
        block_vy: float,
        block_ax: float = 0.0,
        block_ay: float = 0.0,
        car_x: float = 0.0,
        car_y: float = 0.0,
        car_v: float = 0.0,
        num_waypoints: int = 10,
    ) -> Optional[dict]:
        """
        同 solve()，额外返回从当前位置到拦截点的路径采样点（供纯追踪/TEB 使用）。
        """
        result = self.solve(
            block_x, block_y, block_vx, block_vy,
            block_ax, block_ay, car_x, car_y, car_v,
        )
        if result is None:
            return None

        # 生成物块轨迹采样点
        T = result["T"]
        waypoints = []
        for i in range(num_waypoints + 1):
            t = T * i / num_waypoints
            wx = block_x + block_vx * t + 0.5 * block_ax * t ** 2
            wy = block_y + block_vy * t + 0.5 * block_ay * t ** 2
            waypoints.append((wx, wy))

        result["waypoints"] = waypoints
        return result


# ==================== 自测 ====================

if __name__ == "__main__":
    print("=== 拦截点规划器自测 ===\n")

    planner = InterceptPlanner(
        car_max_speed=200.0,
        car_accel=100.0,
        car_decel=150.0,
        time_resolution=0.02,
        max_predict_time=5.0,
    )

    # 场景：物块在 (500, 300) 处以 (80, -30) px/s 移动
    #       小车在 (0, 0)，静止
    print("场景1: 小车静止，物块中速")
    r = planner.solve(
        block_x=500, block_y=300,
        block_vx=80, block_vy=-30,
        car_x=0, car_y=0, car_v=0,
    )
    if r:
        print(f"  拦截点: ({r['x']:.1f}, {r['y']:.1f})")
        print(f"  T={r['T']:.2f}s  T_car={r['T_car']:.2f}s  可行={r['feasible']}")
    else:
        print("  无法拦截")

    # 场景2: 物块很近，慢速
    print("\n场景2: 物块近，慢速")
    r = planner.solve(
        block_x=100, block_y=50,
        block_vx=20, block_vy=5,
        car_x=0, car_y=0, car_v=0,
    )
    if r:
        print(f"  拦截点: ({r['x']:.1f}, {r['y']:.1f})")
        print(f"  T={r['T']:.2f}s  T_car={r['T_car']:.2f}s  可行={r['feasible']}")
    else:
        print("  无法拦截")

    # 场景3: 物块很快，小车在移动中
    print("\n场景3: 物块快，小车已有速度")
    r = planner.solve_with_path(
        block_x=800, block_y=400,
        block_vx=150, block_vy=-50,
        car_x=100, car_y=200, car_v=80,
        num_waypoints=8,
    )
    if r and r["feasible"]:
        print(f"  拦截点: ({r['x']:.1f}, {r['y']:.1f})")
        print(f"  T={r['T']:.2f}s  T_car={r['T_car']:.2f}s  可行={r['feasible']}")
        print(f"  路径采样点({len(r['waypoints'])}个):")
        for i, (wx, wy) in enumerate(r["waypoints"]):
            print(f"    [{i}] ({wx:.1f}, {wy:.1f})")
    elif r:
        print(f"  最优解但不可行: ({r['x']:.1f}, {r['y']:.1f}) T={r['T']:.2f}s T_car={r['T_car']:.2f}s")
    else:
        print("  无法拦截")

    print("\n=== 测试完成 ===")
