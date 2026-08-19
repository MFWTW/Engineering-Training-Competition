#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐标换算 / 底盘与夹爪指令验证工具
════════════════════════════════════

用途：
    在真机上验证 pixel_to_camera / command_to_protocol_mm 算出来的
    “底盘移动量、夹爪伸长量”和实际测量是否一致。

用法：
    python3 validate_transform.py
    python3 validate_transform.py --actual-distance-cm 42

操作：
    鼠标左键  点击画面中的物块/圆环中心，立即打印坐标和指令
    M         输入实际测量距离(cm)，对比计算误差是否 <= 5cm
    P         输入“相机正下方到物块的水平前方距离(cm)”，自动反推俯仰角
    Q         退出

测量参考点：
    - 车中心距离 = 车中心在地面的投影 → 物块中心的水平距离
    - 相机距离   = 相机镜头正下方地面点 → 物块中心的水平距离
"""

import argparse
import math

import cv2

from common_camera import (
    open_camera,
    DETECTION_CAMERA_SOURCE,
    DETECTION_FRAME_WIDTH,
    DETECTION_FRAME_HEIGHT,
)
import transformer


WINDOW_NAME = "transform_validate"


def parse_args():
    parser = argparse.ArgumentParser(
        description="验证像素→相机坐标、底盘移动量、夹爪伸长量",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(DETECTION_CAMERA_SOURCE),
        help="摄像头编号或 /dev/videoX 路径",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DETECTION_FRAME_WIDTH,
        help="采集宽度，需与主程序一致",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DETECTION_FRAME_HEIGHT,
        help="采集高度，需与主程序一致",
    )
    parser.add_argument(
        "--actual-distance-cm",
        type=float,
        default=None,
        help="已实测的“车中心到物块”水平距离(cm)；不填则按 M 后手动输入",
    )
    parser.add_argument(
        "--pitch-deg",
        type=float,
        default=None,
        help="相机俯仰角（度，正=向下俯拍，90=光轴垂直向下）；不填用 transformer.CAMERA_PITCH_DEG",
    )
    parser.add_argument(
        "--camera-forward-cm",
        type=float,
        default=None,
        help="已实测的“相机正下方地面点到物块”的水平前方距离(cm)，用于反推俯仰角",
    )
    return parser.parse_args()


def evaluate_point(u, v, image_w, image_h, pitch_deg=None):
    """计算单个像素点的坐标换算结果和底盘/夹爪指令。"""
    camera_coord = transformer.pixel_to_camera(
        u, v,
        image_width=image_w,
        image_height=image_h,
        camera_pitch_deg=pitch_deg,
    )
    if camera_coord is None:
        print(
            f"[无效] 点({u},{v}) 无法换算：物块必须出现在主点下方，"
            "请检查相机俯仰角和物块在画面中的位置"
        )
        return None

    world_coord = transformer.camera_to_world(camera_coord)
    chassis_x_mm, chassis_y_mm, gripper_mm = (
        transformer.command_to_protocol_mm(camera_coord)
    )
    result = transformer.decide_from_camera(camera_coord)

    dist_car_dy_cm = world_coord[1]
    dist_car_euclid_cm = math.hypot(world_coord[0], world_coord[1])
    dist_camera_cm = math.hypot(camera_coord[0], camera_coord[1])

    print("=" * 62)
    used_pitch = transformer.CAMERA_PITCH_DEG if pitch_deg is None else pitch_deg
    print(f"相机俯仰角: {used_pitch:.1f}°")
    print(f"像素坐标: ({u}, {v}) @ {image_w}x{image_h}")
    print(f"相机坐标: x={camera_coord[0]:.1f}cm, "
          f"y={camera_coord[1]:.1f}cm, z={camera_coord[2]:.1f}cm")
    print(f"车中心坐标: x={world_coord[0]:.1f}cm, "
          f"y={world_coord[1]:.1f}cm, z={world_coord[2]:.1f}cm")
    print(f"底盘移动: x={result['chassis_x_cm']:.1f}cm "
          f"({result['chassis_x_direction']}), "
          f"y={result['chassis_y_cm']:.1f}cm "
          f"({result['chassis_y_direction']})")
    mode_text = {
        "too_close": "物块比最短距离更近",
        "in_range": "物块在夹爪行程范围内",
        "too_far": "物块比最长距离更远",
    }.get(result["mode"], result["mode"])
    print(f"决策模式: {result['mode']} ({mode_text})")
    print(f"夹爪行程(相对车中心): 最短={transformer.min_jar_dis[1]:.2f}cm, "
          f"最长={transformer.max_jar_dis[1]:.2f}cm")
    print(f"夹爪目标位置(相对车中心): {result['gripper_target_cm']:.1f}cm")
    print(f"下位机伸长量(相对最短): {gripper_mm}mm "
          f"({gripper_mm / 10.0:.1f}cm)")
    print(f"校验: 夹爪目标 {result['gripper_target_cm']:.1f} + 底盘纵向 "
          f"{result['chassis_y_cm']:+.1f} = "
          f"{result['gripper_target_cm'] + result['chassis_y_cm']:.1f}cm "
          f"(= 物块dy {world_coord[1]:.1f}cm)")
    print(f"串口指令(mm): 底盘=({chassis_x_mm:+d},{chassis_y_mm:+d}), "
          f"夹爪={gripper_mm}")
    print(f"计算-车中心到物块纵向距离(dy): {dist_car_dy_cm:.1f}cm")
    print(f"计算-车中心到物块水平距离(含左右偏移): {dist_car_euclid_cm:.1f}cm")
    print(f"计算-相机到物块水平距离: {dist_camera_cm:.1f}cm")
    return {
        "u": u,
        "v": v,
        "camera_coord": camera_coord,
        "world_coord": world_coord,
        "dist_car_cm": dist_car_dy_cm,
        "dist_car_euclid_cm": dist_car_euclid_cm,
        "dist_camera_cm": dist_camera_cm,
    }


def draw_state(frame, points):
    disp = frame.copy()
    for i, (u, v) in enumerate(points):
        cv2.circle(disp, (u, v), 6, (0, 255, 255), -1)
        cv2.putText(
            disp,
            f"#{i + 1}",
            (u + 10, v - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
    cv2.putText(
        disp,
        "L-click block | M input measured dist | Q quit",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        4,
    )
    cv2.putText(
        disp,
        "L-click block | M input measured dist | Q quit",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    return disp


def main():
    args = parse_args()
    cap = open_camera(args.source, args.width, args.height)
    if cap is None or not cap.isOpened():
        print("无法打开摄像头")
        return

    pending = []
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending.append((x, y))

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    except cv2.error as e:
        print(f"无法创建显示窗口: {e}")
        print("请在带桌面的本机终端运行，或先配置 X11 转发")
        cap.release()
        return

    print("已打开相机，点击画面中的物块/圆环中心查看计算结果。")
    print("按 M 后输入实际测量的车中心到物块距离(cm)，脚本会判断误差是否 <= 5cm。\n")
    print("按 P 后输入相机正下方到物块的水平前方距离(cm)，脚本会给出 CAMERA_PITCH_DEG。\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            while pending:
                u, v = pending.pop(0)
                points.append((u, v))
                info = evaluate_point(
                    u, v, frame.shape[1], frame.shape[0], args.pitch_deg
                )
                if info is not None and args.actual_distance_cm is not None:
                    measured = args.actual_distance_cm
                    err = abs(info["dist_car_cm"] - measured)
                    status = "OK <=5cm" if err <= 5.0 else "BAD >5cm"
                    print(
                        f"实测={measured:.1f}cm, 计算={info['dist_car_cm']:.1f}cm, "
                        f"误差={err:.1f}cm -> {status}"
                    )

            cv2.imshow(WINDOW_NAME, draw_state(frame, points))
            key = cv2.waitKey(1) & 0xFF

            if key == ord("m"):
                if not points:
                    print("请先用鼠标点一个物块中心")
                    continue
                info = None
                for u, v in reversed(points):
                    info = evaluate_point(
                        u, v, frame.shape[1], frame.shape[0], args.pitch_deg
                    )
                    if info is not None:
                        break
                if info is None:
                    continue

                if args.actual_distance_cm is not None:
                    measured = args.actual_distance_cm
                else:
                    try:
                        measured = float(
                            input("请输入实际测量的车中心到物块距离(cm): ")
                        )
                    except (ValueError, EOFError):
                        print("输入无效，已取消对比")
                        continue

                err = abs(info["dist_car_cm"] - measured)
                status = "OK <=5cm" if err <= 5.0 else "BAD >5cm"
                print(
                    f"实测={measured:.1f}cm, 计算={info['dist_car_cm']:.1f}cm, "
                    f"误差={err:.1f}cm -> {status}"
                )
            elif key == ord("p"):
                if not points:
                    print("请先用鼠标点一个物块中心")
                    continue
                u, v = points[-1]

                if args.camera_forward_cm is not None:
                    forward_cm = args.camera_forward_cm
                else:
                    try:
                        forward_cm = float(
                            input(
                                "请输入相机正下方到物块的水平前方距离(cm): "
                            )
                        )
                    except (ValueError, EOFError):
                        print("输入无效，已取消")
                        continue

                pitch = transformer.estimate_pitch_deg(
                    u, v, forward_cm,
                    image_width=frame.shape[1],
                    image_height=frame.shape[0],
                )
                if pitch is None:
                    print("无法反推俯仰角，请检查输入距离和相机参数")
                    continue
                print(
                    f"[结果] 建议 CAMERA_PITCH_DEG = {pitch:.2f} "
                    f"(当前 transformer 默认 {transformer.CAMERA_PITCH_DEG})"
                )
                info = evaluate_point(
                    u, v, frame.shape[1], frame.shape[0], pitch
                )
                if info is not None and args.actual_distance_cm is not None:
                    err = abs(info["dist_car_cm"] - args.actual_distance_cm)
                    status = "OK <=5cm" if err <= 5.0 else "BAD >5cm"
                    print(
                        f"按该俯仰角: 实测={args.actual_distance_cm:.1f}cm, "
                        f"计算={info['dist_car_cm']:.1f}cm, "
                        f"误差={err:.1f}cm -> {status}"
                    )
            elif key == ord("q"):
                break
    except KeyboardInterrupt:
        print("\n用户手动终止")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
