#!/usr/bin/env python3
"""
摄像头角色配置工具
────────────────────────────
把两台 USB 摄像头分别固定为「二维码扫描相机」和「物块检测/放置识别相机」，
写入 camera_roles.json。之后 src.py、hsv_tuner、标定脚本都会自动使用该配置，
不用再改 common_camera.py 里的编号。

用法：
    python3 camera_setup.py              # 逐个预览摄像头，按键指定角色
    python3 camera_setup.py --list       # 只列出当前可用的摄像头
    python3 camera_setup.py --udev       # 额外生成固定符号链接规则
                                         # （/dev/video_qr、/dev/video_detect）

预览画面内按键：
    Q     把当前摄像头设为二维码扫描相机
    D     把当前摄像头设为物块检测/放置识别相机
    S     跳过当前摄像头
    ESC   退出，不保存
"""

import argparse
import glob
import json
import re
import subprocess
from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parent
ROLES_FILE = BASE_DIR / "camera_roles.json"
UDEV_RULES_FILE = BASE_DIR / "99-robomaster-cameras.rules"

ROLE_KEYS = {
    "qr": ("qr_camera", "二维码扫描相机"),
    "detection": ("detection_camera", "物块检测/放置识别相机"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="USB 摄像头角色配置")
    parser.add_argument("--list", action="store_true",
                        help="只列出可用摄像头，不进入指定流程")
    parser.add_argument("--udev", action="store_true",
                        help="同时生成 udev 固定符号链接规则并写回 camera_roles.json")
    return parser.parse_args()


def list_capture_nodes():
    """按 /dev/video* 顺序探测能真正读到画面的采集节点。

    UVC 摄像头常带有 metadata 节点（能 open 但读不出画面），
    这里通过实际读帧把它过滤掉。
    """
    paths = sorted(
        glob.glob("/dev/video*"),
        key=lambda p: int(p.rsplit("video", 1)[1]),
    )
    nodes = []
    for path in paths:
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        got_frame = False
        for _ in range(3):
            ret, _ = cap.read()
            if ret:
                got_frame = True
                break
        cap.release()
        if got_frame:
            nodes.append(path)
    return nodes


def _udev_props(path):
    props = {}
    try:
        out = subprocess.check_output(
            ["udevadm", "info", "-q", "property", "-n", path],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
    except Exception:
        return props
    for line in out.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            props[key.strip()] = value.strip()
    return props


def _usb_kernels(path):
    """返回该视频设备所在 USB 接口的 KERNELS，如 1-1.2:1.0。"""
    try:
        out = subprocess.check_output(
            ["udevadm", "info", "-a", "-n", path],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
    except Exception:
        return None
    pattern = re.compile(
        r'KERNELS=="([0-9]+-[0-9]+(?:\.[0-9]+)*(?::[0-9]+\.[0-9]+)?)"'
    )
    for line in out.splitlines():
        m = pattern.search(line)
        if m:
            return m.group(1)
    return None


def describe(path, index):
    props = _udev_props(path)
    info = [f"[{index}] {path}"]

    name = (
        props.get("ID_V4L_PRODUCT")
        or props.get("ID_MODEL")
        or props.get("ID_V4L_VENDOR")
    )
    if name:
        info.append(name)

    detail = []
    vendor = props.get("ID_VENDOR_ID")
    model = props.get("ID_MODEL_ID")
    if vendor and model:
        detail.append(f"USB {vendor}:{model}")
    serial = props.get("ID_SERIAL_SHORT") or props.get("ID_SERIAL")
    if serial:
        detail.append(f"SN {serial}")
    port = props.get("ID_PATH")
    if port:
        detail.append(f"路径 {port}")
    if detail:
        info.append(" | ".join(detail))
    return "  ".join(info)


def preview_assign(path):
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"无法打开 {path}，跳过")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    window = f"Assign: {path}"
    print(f"正在预览 {path}（Q=扫码  D=物块检测  S=跳过  ESC=退出）...")

    try:
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.putText(
                    frame,
                    "Q=QR scanner  D=detection  S=skip  ESC=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
                cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                return "qr"
            if key in (ord("d"), ord("D")):
                return "detection"
            if key in (ord("s"), ord("S")):
                return None
            if key == 27:
                raise KeyboardInterrupt
    finally:
        cap.release()
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass


def write_roles(roles):
    ROLES_FILE.write_text(
        json.dumps(roles, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def generate_udev_rules(roles):
    """为已指定角色的摄像头生成固定符号链接规则（/dev/video_qr、/dev/video_detect）。"""
    symlink_map = {
        "qr_camera": "video_qr",
        "detection_camera": "video_detect",
    }
    rules = ["# 机器人摄像头角色绑定（由 camera_setup.py 生成，可重跑覆盖）\n",
             "# 若符号链接指向 metadata 节点无法出图，删除对应行中的 ATTR{index}==\"0\" 再试\n"]
    new_roles = dict(roles)

    for key, symlink in symlink_map.items():
        if key not in roles:
            continue
        path = roles[key]
        kernels = _usb_kernels(path)
        if not kernels:
            print(f"警告：无法获取 {path} 的 USB 端口信息，跳过 /dev/{symlink} 规则")
            continue
        rules.append(
            f'# {path} -> /dev/{symlink}（{key}）\n'
            f'SUBSYSTEM=="video4linux", KERNELS=="{kernels}", '
            f'ATTR{{index}}=="0", SYMLINK+="{symlink}", MODE="0666"\n'
        )
        new_roles[key] = f"/dev/{symlink}"

    if len(rules) <= 2:
        print("没有生成任何 udev 规则（两个角色都未成功取得 USB 端口信息）。")
        return roles

    UDEV_RULES_FILE.write_text("".join(rules), encoding="utf-8")
    write_roles(new_roles)
    print(f"\n已生成 {UDEV_RULES_FILE.name}，角色配置已改用固定链接：")
    for key, symlink in symlink_map.items():
        if key in new_roles:
            print(f"  {key} = /dev/{symlink}")
    print("安装并生效（需要 sudo，一次即可，之后插拔/重启顺序都不会变）：")
    print(f"  sudo cp {UDEV_RULES_FILE} /etc/udev/rules.d/")
    print("  sudo udevadm control --reload-rules")
    print("  sudo udevadm trigger")
    print("  ls -l /dev/video_qr /dev/video_detect")
    return new_roles


def main():
    args = parse_args()
    nodes = list_capture_nodes()
    if not nodes:
        print("没有检测到可用的 USB 摄像头（/dev/video*），请先连接摄像头再运行。")
        return 1

    print("检测到的可用摄像头（[n] 是 OpenCV 设备编号）：")
    for index, path in enumerate(nodes):
        print("  " + describe(path, index))
    if args.list:
        print("\n共 %d 个可用摄像头。" % len(nodes))
        return 0

    print("\n下面会逐个显示摄像头画面，按 Q / D 指定角色；按 S 跳过。")
    roles = {}
    try:
        for index, path in enumerate(nodes):
            print(f"\n>>> 当前显示 [{index}] {path}")
            role = preview_assign(path)
            if role in ROLE_KEYS:
                key, label = ROLE_KEYS[role]
                roles[key] = path
                print(f"已把 {path} 设为{label}")
            else:
                print(f"跳过 {path}")

            if "qr_camera" in roles and "detection_camera" in roles:
                print("两个角色都已指定，不再预览剩余摄像头。")
                break
    except KeyboardInterrupt:
        print("\n已取消，未保存。")
        return 1

    if not roles:
        print("没有指定任何角色，未生成 camera_roles.json。")
        return 1

    if "qr_camera" not in roles or "detection_camera" not in roles:
        missing = []
        if "qr_camera" not in roles:
            missing.append("二维码扫描相机")
        if "detection_camera" not in roles:
            missing.append("物块检测相机")
        print(f"\n注意：{('、'.join(missing))} 未指定，将沿用 common_camera.py 里的默认值。")

    if args.udev:
        roles = generate_udev_rules(roles)
    else:
        write_roles(roles)
        print(f"\n已写入 {ROLES_FILE.name}：")
        for key, value in roles.items():
            print(f"  {key} = {value}")
        print("\n提示：如果以后插拔顺序导致编号互换，重跑一次本工具即可；")
        print("想彻底固定，用 python3 camera_setup.py --udev 生成 /dev/video_qr、/dev/video_detect。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
