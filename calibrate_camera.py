#!/usr/bin/env python3
"""
相机内参标定脚本（棋盘格）

用法：
    python3 calibrate_camera.py --cols 9 --rows 6 --square-mm 25

按键：
    s / 空格   保存当前检测到棋盘格的画面
    d          删除最近保存的一张
    c          开始标定
    q          退出

标定结果保存到 camera_calibration.json；
加 --update-transformer 会把 fx/fy/cx/cy 自动写回 config.yaml（transformer 段）。
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from common_camera import (
    open_camera,
    DETECTION_CAMERA_SOURCE,
    DETECTION_FRAME_WIDTH,
    DETECTION_FRAME_HEIGHT,
)


BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="棋盘格相机标定")
    parser.add_argument("--usb-id", type=str, default=str(DETECTION_CAMERA_SOURCE),
                        help="USB 摄像头编号或 /dev/video* 路径（默认物块检测相机）")
    parser.add_argument("--cols", type=int, default=9,
                        help="棋盘格内角点数（列）")
    parser.add_argument("--rows", type=int, default=6,
                        help="棋盘格内角点数（行）")
    parser.add_argument("--square-mm", type=float, default=25.0,
                        help="单个棋盘格边长（mm）")
    parser.add_argument("--min-images", type=int, default=10,
                        help="建议最少采集图像数")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "camera_calibration.json",
                        help="标定结果保存路径")
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="同时保存标定原图到该目录（可选）")
    parser.add_argument("--update-transformer", action="store_true",
                        help="标定后自动把 fx/fy/cx/cy 写回 config.yaml（transformer 段）")
    return parser.parse_args()


def read_usb_frame(cap):
    ret, frame = cap.read()
    return frame if ret else None


def calibrate(objpoints, imgpoints, image_size, square_mm):
    """执行标定并计算重投影误差"""
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    # 平均重投影误差
    mean_error = 0.0
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx, dist
        )
        error = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
        mean_error += error
    mean_error /= len(objpoints)

    return {
        "image_size": [int(image_size[0]), int(image_size[1])],
        "square_mm": float(square_mm),
        "num_images": len(objpoints),
        "rms": float(rms),
        "mean_reprojection_error_px": float(mean_error),
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.flatten().tolist(),
        "focal_px_x": float(mtx[0, 0]),
        "focal_px_y": float(mtx[1, 1]),
        "principal_px_x": float(mtx[0, 2]),
        "principal_px_y": float(mtx[1, 2]),
    }


def update_transformer(calib):
    """把标定得到的 fx/fy/cx/cy 和畸变系数写回 config.yaml 的 transformer 段"""
    path = BASE_DIR / "config.yaml"
    if not path.exists():
        print(f"[警告] 找不到 {path}，跳过写回")
        return

    replacements = [
        ("focal_px_x", f"{calib['focal_px_x']:.6f}"),
        ("focal_px_y", f"{calib['focal_px_y']:.6f}"),
        ("principal_px_x", f"{calib['principal_px_x']:.6f}"),
        ("principal_px_y", f"{calib['principal_px_y']:.6f}"),
        ("dist_coeffs", "[" + ", ".join(repr(float(v)) for v in calib["dist_coeffs"]) + "]"),
    ]

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    updated = 0
    in_transformer = False
    for i, line in enumerate(lines):
        # 顶层段名（无缩进）决定是否在 transformer 段内
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*:", line):
            in_transformer = line.startswith("transformer:")
            continue
        if not in_transformer:
            continue
        for name, value_str in replacements:
            match = re.match(r"(\s*)" + re.escape(name) + r":\s*(.*)$", line)
            if match:
                lines[i] = f"{match.group(1)}{name}: {value_str}\n"
                updated += 1
                # dist_coeffs 若原来是列表块格式（- 项逐行），改成单行列表
                if name == "dist_coeffs" and not match.group(2).strip():
                    j = i + 1
                    while j < len(lines) and re.match(r"^\s+-\s", lines[j]):
                        j += 1
                    del lines[i + 1:j]
                break

    if updated == len(replacements):
        path.write_text("".join(lines), encoding="utf-8")
        print(f"[OK] fx/fy/cx/cy 和畸变系数已写回 {path}（transformer 段）")
    else:
        print(f"[警告] 只更新了 {updated}/{len(replacements)} 项，请手动检查 config.yaml 的 transformer 段")


def main():
    args = parse_args()
    pattern_size = (args.cols, args.rows)

    # 棋盘格三维坐标（单位用 cm，标定内参不受单位影响）
    square_cm = args.square_mm / 10.0
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * square_cm

    cap = open_camera(
        args.usb_id,
        width=DETECTION_FRAME_WIDTH,
        height=DETECTION_FRAME_HEIGHT,
    )
    if cap is None:
        print(f"[错误] 无法打开 USB 摄像头 {args.usb_id}")
        sys.exit(1)

    objpoints = []
    imgpoints = []
    image_paths = []
    frame_index = 0

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    print(f"棋盘格内角点: {args.cols}x{args.rows}，方格边长: {args.square_mm}mm")
    print("按 s/空格 保存当前帧，d 删除上一张，c 标定，q 退出")

    try:
        while True:
            frame = read_usb_frame(cap)

            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray,
                pattern_size,
                None,
                cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_NORMALIZE_IMAGE
                | cv2.CALIB_CB_FAST_CHECK,
            )

            display = frame.copy()
            corners_refined = corners
            if found:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )
                cv2.drawChessboardCorners(
                    display, pattern_size, corners_refined, found
                )

            status = f"Saved: {len(objpoints)}/{args.min_images}"
            cv2.putText(display, status, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "S:Save  D:Delete  C:Calibrate  Q:Quit",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)

            if not found:
                cv2.putText(display, "Chessboard not detected",
                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
            elif len(objpoints) >= args.min_images:
                cv2.putText(display, "Enough frames, press C to calibrate",
                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)

            cv2.imshow("camera", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key in (ord("s"), ord(" "), 13) and found:
                objpoints.append(objp)
                imgpoints.append(corners_refined)
                frame_index += 1
                path = ""
                if args.save_dir is not None:
                    path = str(args.save_dir / f"calib_{frame_index:03d}.jpg")
                    cv2.imwrite(path, frame)
                image_paths.append(path)
                print(f"[{len(objpoints)}] 已保存第 {frame_index} 张")

            if key == ord("d") and objpoints:
                objpoints.pop()
                imgpoints.pop()
                removed = image_paths.pop()
                if removed and Path(removed).exists():
                    Path(removed).unlink()
                print(f"[删除] 当前剩 {len(objpoints)} 张")

            if key == ord("c"):
                if len(objpoints) < 3:
                    print("[错误] 至少需要 3 张不同角度的棋盘格图片")
                    continue

                h, w = gray.shape[:2]
                calib = calibrate(objpoints, imgpoints, (w, h), args.square_mm)
                args.out.write_text(
                    json.dumps(calib, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                print("\n===== 标定结果 =====")
                print(f"图像尺寸: {calib['image_size']}")
                print(f"使用图像数: {calib['num_images']}")
                print(f"RMS: {calib['rms']:.4f}")
                print(f"平均重投影误差: {calib['mean_reprojection_error_px']:.4f} px")
                print(f"fx = {calib['focal_px_x']:.4f}")
                print(f"fy = {calib['focal_px_y']:.4f}")
                print(f"cx = {calib['principal_px_x']:.4f}")
                print(f"cy = {calib['principal_px_y']:.4f}")
                print(f"畸变系数: {calib['dist_coeffs']}")
                print(f"结果已保存: {args.out}")

                if args.update_transformer:
                    update_transformer(calib)

                break

    finally:
        cv2.destroyAllWindows()
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    main()
