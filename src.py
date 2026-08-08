import cv2
import numpy as np
import serial
import threading
import time
import queue
from common_camera import open_camera
from preprocessing import *
import scan_QRcode_andlist
from hikrobot_camera import *
from felling_color import (
    block_preprocessing, get_detector, reset_detector,
)
from gimbal import SerialComm, VisionToGimbal, GimbalToVision
from kalman_tracker import KalmanBlockTracker
from intercept_planner import InterceptPlanner

# ==================== 控制模式 ====================
# "stm": 等待 STM32 串口反馈 ("5"/"6"/"b"/"c") 后才切换目标
# "manual": 按键盘 n/空格 手动切换目标
CONTROL_MODE = "manual"      # <-- 改为 "manual" 则手动模式

# ==================== ROI 配置 ====================
# 若不需要 ROI 限制，设为 None；否则设为 [x, y, w, h]
# 这里改为更小的检测区域，以减少搜索范围
HIK_DETECTION_ROI = [400, 240, 560, 420]
# ==================== 全局变量 ====================
C_1 = None
C_2 = None

# 底盘回传数据（串口接收线程更新，主循环读取）
chassis_x: int = 0
chassis_y: int = 0
chassis_theta: int = 0  # 航向角（原始值）

# 共享串口对象（发送线程 + 主循环都需要用）
_serial_comm = None

# 串口接收器（独立于发送线程的 SerialComm，用于读 STM32 反馈）
_serial_reader = None
try:
    _serial_reader = serial.Serial(
        port='/dev/ttyACM0', baudrate=115200, timeout=0.1
    )
    print("串口接收器 /dev/ttyACM0 已打开")
except Exception as e:
    print(f"[警告] 串口接收器打开失败: {e}")


def receive_data():
    """非阻塞读取 STM32 反馈指令"""
    if _serial_reader is None:
        return None
    try:
        if _serial_reader.in_waiting > 0:
            return _serial_reader.readline().decode('utf-8').strip()
    except Exception:
        pass
    return None


def clamp_roi(roi, frame_shape):
    """修正 ROI 在图像范围内，避免超出边界。"""
    if roi is None:
        return None
    x, y, w, h = roi
    H, W = frame_shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return [x, y, w, h]


# ==================== 发送线程 ====================
def Sending2Gimbal(data_pack, serial_comm):
    """后台常驻：阻塞等待队列数据，收到就打包发送"""
    print("[发送线程启动]，等待数据")

    while True:
        vg = data_pack.get()
        if vg is None:
            print("[发送线程] 收到停止信号，退出")
            break

        try:
            packed = vg.pack()
            print(f"  hex: {packed.hex(' ')}")
        except Exception as e:
            print(f"  pack 失败: {e}")
            continue

        if serial_comm:
            try:
                serial_comm.send(vg)
                print("串口发送成功")
            except Exception as e:
                print(f"  串口发送失败: {e}")
        else:
            print("  [离线] 未发送")

    print("[发送线程] 已退出")


# ==================== 主程序 ====================
def main():
    global C_1, C_2, chassis_x, chassis_y, chassis_theta, _serial_comm

    cap = open_camera(camera_id=0)
    usb_camera_active = cap is not None
    if not usb_camera_active:
        return

    dev_list = enum_devices()
    if dev_list is None:
        print("未检测到海康摄像头")
    cam = create_camera_handle(dev_list, 0, width=1440, height=1080)
    start_grabbing(cam)

    q = queue.Queue()

    # 创建串口 + 启动底盘接收线程
    try:
        _serial_comm = SerialComm()
        _serial_comm.start_chassis_recv()
        print("串口已打开，底盘接收线程已启动")
    except Exception as e:
        print(f"串口打开失败: {e}")
        _serial_comm = None

    sending_thread = threading.Thread(target=Sending2Gimbal, args=(q, _serial_comm), daemon=True)
    sending_thread.start()

    # ---- 状态 ----
    last_sessions = None
    detector = get_detector()
    detector.reset()
    # 颜色检测现改为全图处理，detection_area 仅用于画面可视化
    # 放宽稳定性参数：轮廓检测的圆心会抖动，降低阈值+放宽位移容忍
    detector.stability_settings['threshold'] = 15           # 原 30，太严格
    detector.stability_settings['max_pixel_move'] = 20      # 原 10，轮廓抖动容易超
    detector.stability_settings['color_stable_threshold'] = 8  # 原 15

    # 卡尔曼滤波追踪器（替代神经网络预测）
    kf = KalmanBlockTracker(dt=1.0 / 30.0)  # 假设 30fps，会根据实际 dt 调整

    # 拦截规划器（考虑小车动力学，博弈 T 求解）
    planner = InterceptPlanner(
        car_max_speed=200.0, car_accel=100.0, car_decel=150.0,
        time_resolution=0.02, max_predict_time=5.0,
    )

    roi_clamped = False  # ROI 只修正一次

    detection_sent = False
    waiting_for_next = False    # 等待手动切换
    target_colors = []
    target_index = 0
    last_detection_time = None
    sent_time = None  # 记录发送时间
    auto_switch_timeout = 10.0  # 自动切换超时时间（秒）

    # 上一个检测到的物块信息（等待阶段重绘用）
    last_detected_center = None
    last_detected_radius = 0
    last_detected_color = (0, 255, 0)
    last_detected_label = ""

    # 图像中心容差（物块圆心距图像中心多少像素内算"对准"）
    CENTER_TOLERANCE = 5

    try:
        while True:
            if usb_camera_active:
                ret, frame = cap.read()
            else:
                ret, frame = False, None

            if not ret and not scan_QRcode_andlist.session:
                print("无法读取画面")
                break

            # ============ QR 扫描阶段 ============
            if len(scan_QRcode_andlist.session) == 0:
                Ostu_image = ostu_threshold(frame)
                scan_QRcode_andlist.scan_qrcode(Ostu_image, frame)
                sessions = scan_QRcode_andlist.session

                if sessions and sessions != last_sessions:
                    last_sessions = sessions
                    print(f"QR识别结果: {sessions}")
                    # sessions = ['1','3','2','+','3','1','2']
                    # 编码到 4 个 uint16
                    qr_ints = [int(s) if s.isdigit() else ord(s) for s in sessions[:4]]
                    while len(qr_ints) < 4:
                        qr_ints.append(0)
                    vg = VisionToGimbal(target=0, QR=qr_ints)
                    q.put(vg)

                    # 提取目标颜色序列
                    target_colors = [s for s in sessions if s.isdigit()]
                    target_index = 0
                    if target_colors:
                        print(f"目标颜色序列: {target_colors}")
                        color_names = {"1": "红色", "2": "绿色", "3": "蓝色",
                                      "4": "浅蓝", "5": "黑色", "6": "黄色"}
                        first_color_name = color_names.get(target_colors[0], "未知")
                        print(f">>> 第一个物块: {first_color_name} (代码{target_colors[0]})")

                if last_sessions is not None:
                    if cap:
                        cap.release()
                        cap = None
                    usb_camera_active = False
                    cv2.destroyAllWindows()
                    print("识别到QR，关闭USB摄像头")
                    detector.reset()
                    kf.reset()
                    detection_sent = False
                    last_detection_time = None
                    last_detected_center = None

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # ============ 物块检测阶段 ============
            else:
                # ---- 时间戳同步：先记底盘最新时间，再抓帧 ----
                chassis_data = _serial_comm.get_chassis_data() if _serial_comm else None
                if chassis_data:
                    chassis_x = chassis_data.chassis_x
                    chassis_y = chassis_data.chassis_y
                    chassis_theta = chassis_data.theta
                    chassis_t = chassis_data.timestamp  # 底盘数据时间戳

                # 抓取图像帧，记录时间
                frame_t = time.time()
                hik_fram = read_frame(cam)

                # 若图像帧落后于底盘数据，说明底盘更新了，读取最新值
                chassis_data2 = _serial_comm.get_chassis_data() if _serial_comm else None
                if chassis_data2 and chassis_data2.timestamp > frame_t:
                    # 用离 frame_t 更近的底盘数据
                    if chassis_data is None or (
                        abs(chassis_data2.timestamp - frame_t) < abs(chassis_t - frame_t)
                    ):
                        chassis_x = chassis_data2.chassis_x
                        chassis_y = chassis_data2.chassis_y
                        chassis_theta = chassis_data2.theta

                # ---- 读取串口反馈（仅 STM32 模式） ----
                if CONTROL_MODE == "stm":
                    cmd = receive_data()
                    if cmd:
                        print(f"[STM32] 收到指令: {cmd}")
                        if cmd in ("5", "6", "b", "c"):
                            if waiting_for_next:
                                print(f"[STM32] 收到 '{cmd}'，切换到下一个目标")
                                waiting_for_next = False
                                detector.reset()
                                kf.reset()
                                detection_sent = False
                                target_index += 1
                                if target_index >= len(target_colors):
                                    print("所有目标检测完毕")

                if hik_fram is not None:
                    h_img, w_img = hik_fram.shape[:2]

                    # 修正 ROI（仅首次），后续直接绘制
                    if not roi_clamped and detector.detection_area is not None:
                        detector.detection_area = clamp_roi(detector.detection_area, hik_fram.shape)
                        roi_clamped = True
                    if detector.detection_area is not None:
                        rx, ry, rw, rh = detector.detection_area
                        cv2.rectangle(hik_fram, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
                        cv2.putText(hik_fram, "ROI", (rx + 5, ry + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # 显示 QR 和状态
                    if scan_QRcode_andlist.session:
                        qr_text = "+".join(scan_QRcode_andlist.session)
                        cv2.putText(hik_fram, qr_text, (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # 底盘坐标 + 航向角（右上角）
                    cv2.putText(hik_fram, f"chassis:({chassis_x},{chassis_y}) th={chassis_theta}",
                                (w_img - 300, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (200, 200, 200), 1)

                    if target_index < len(target_colors):
                        t = f"Target: color {target_colors[target_index]}  [{target_index + 1}/{len(target_colors)}]"
                        color = (0, 200, 255) if not waiting_for_next else (0, 255, 100)
                        cv2.putText(hik_fram, t, (50, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # ---- 等待切换（根据模式） ----
                    if waiting_for_next:
                        # 在手动模式下，要求至少等待 2 秒以避免快速误触
                        min_display_time = 2.0 if CONTROL_MODE == "manual" else 0
                        elapsed_since_sent = time.time() - sent_time if sent_time else 0
                        
                        if CONTROL_MODE == "stm":
                            hint = "等待 STM32 指令..."
                        else:
                            if elapsed_since_sent < min_display_time:
                                hint = f"请至少等待 {min_display_time - elapsed_since_sent:.1f} 秒再按 n 切换"
                            else:
                                hint = "按 n 切换下一个目标"

                        # 重绘上一个已检测到的物块（避免圆消失）
                        if last_detected_center is not None:
                            cv2.circle(hik_fram, last_detected_center,
                                       last_detected_radius, last_detected_color, 2)
                            cv2.circle(hik_fram, last_detected_center, 3,
                                       last_detected_color, -1)
                            cv2.putText(hik_fram, last_detected_label,
                                        (last_detected_center[0] - 20,
                                         last_detected_center[1] - last_detected_radius - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, last_detected_color, 2)

                        cv2.putText(hik_fram, hint,
                                    (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 255), 2)

                        cv2.imshow("hik", hik_fram)
                        key = cv2.waitKey(1) & 0xFF

                        if key == ord('q'):
                            break

                        # ---- 根据模式切换目标 ----
                        should_switch = False
                        if elapsed_since_sent >= min_display_time and key in (ord('n'), ord('N'), ord(' ')):
                            print(f"[手动] 按键切换到下一个目标")
                            should_switch = True
                        elif CONTROL_MODE != "manual" and elapsed_since_sent >= auto_switch_timeout:
                            print(f"[自动] 超时切换到下一个目标")
                            should_switch = True
                        
                        if should_switch:
                            waiting_for_next = False
                            detector.reset()
                            kf.reset()
                            detection_sent = False
                            sent_time = None
                            last_detected_center = None
                            target_index += 1
                            print(f"已切换到目标 {target_index + 1}/{len(target_colors)}")
                            if target_index >= len(target_colors):
                                print("所有目标检测完毕")

                        # 等待中不执行检测，直接到循环末尾显示
                        continue

                    # ---- 正常检测模式 ----
                    current_time = cv2.getTickCount()
                    current_target = (
                        target_colors[target_index]
                        if target_index < len(target_colors) else None
                    )
                    data, current_center, current_color = block_preprocessing(
                        hik_fram, target=current_target
                    )

                    if data and current_color:
                        last_detection_time = current_time
                        is_stable = detector.update_stability(current_center, current_color)

                        # ---- 卡尔曼滤波：预测 + 更新 ----
                        kf.predict()
                        kf.update(current_center[0], current_center[1])
                        fx, fy, fvx, fvy, fax, fay = kf.get_state()

                        # 可视化
                        color_map = {
                            "1": (0, 0, 255), "2": (0, 255, 0),
                            "3": (255, 0, 0), "4": (255, 255, 0),
                            "5": (0, 0, 0), "6": (0, 255, 255),
                        }
                        draw_color = color_map.get(current_color, (255, 255, 255))
                        radius = detector.last_radius

                        # 原始测量（虚线细圈）
                        cv2.circle(hik_fram, current_center, radius, draw_color, 1)
                        cv2.circle(hik_fram, current_center, 2, (150, 150, 150), -1)

                        # 滤波后（实线粗圈）
                        filtered_center = (int(fx), int(fy))
                        cv2.circle(hik_fram, filtered_center, radius, draw_color, 2)
                        cv2.circle(hik_fram, filtered_center, 4, draw_color, -1)

                        label = {"1": "RED", "2": "GREEN", "3": "BLUE",
                                 "4": "LIGHT_BLUE", "5": "BLACK", "6": "YELLOW"}.get(current_color, "?")
                        cv2.putText(hik_fram, label,
                                    (filtered_center[0] - 20, filtered_center[1] - radius - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)
                        # 速度标注
                        speed = np.sqrt(fvx**2 + fvy**2)
                        cv2.putText(hik_fram, f"V={speed:.0f}px/s",
                                    (filtered_center[0] + radius + 5, filtered_center[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 180), 1)

                        # 保存最近一次检测结果（等待阶段重绘用）
                        last_detected_center = filtered_center
                        last_detected_radius = radius
                        last_detected_color = draw_color
                        last_detected_label = label

                        # ==== KF 预测轨迹 + 路径可视化 ====
                        h, w = hik_fram.shape[:2]
                        cx, cy = w // 2, h // 2

                        # ── 图像中心十字 ──
                        cv2.drawMarker(hik_fram, (cx, cy), (255, 255, 255),
                                       cv2.MARKER_CROSS, 12, 1)

                        # ── KF 预测轨迹（黄色，6步）──
                        future = kf.predict_future(T=2.0, steps=6)
                        prev = filtered_center
                        for i, (px, py) in enumerate(future):
                            pt = (int(px), int(py))
                            alpha = 1.0 - i * 0.12
                            color = (0, int(255 * alpha), int(255 * alpha))
                            if i == 0:
                                cv2.arrowedLine(hik_fram, prev, pt, color, 1, tipLength=0.3)
                            else:
                                cv2.line(hik_fram, prev, pt, color, 1)
                            cv2.circle(hik_fram, pt, 2, color, -1)
                            prev = pt

                        # ── 拦截点（预测轨迹上距中心最近的点）──
                        all_pts = [filtered_center] + [(int(p[0]), int(p[1])) for p in future]
                        best_d = float("inf")
                        best_pt = filtered_center
                        for pt in all_pts:
                            d = np.sqrt((pt[0] - cx)**2 + (pt[1] - cy)**2)
                            if d < best_d:
                                best_d = d
                                best_pt = pt
                        if best_pt != (cx, cy):
                            cv2.arrowedLine(hik_fram, (cx, cy), best_pt,
                                            (0, 0, 255), 2, tipLength=0.15)
                            cv2.circle(hik_fram, best_pt, 5, (0, 0, 255), -1)
                        cv2.putText(hik_fram, f"intercept:{best_d:.0f}px",
                                    (cx + 20, cy - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                        # 显示检测状态
                        status = f"KF追踪中 (稳定度: {detector.stable_count}/{detector.stability_settings['threshold']})"
                        cv2.putText(hik_fram, status, (50, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        # ---- 稳定后：闭环拦截对准 ----
                        if is_stable and not detection_sent:
                            h, w = hik_fram.shape[:2]
                            cx, cy = w // 2, h // 2

                            # 博弈求解拦截点（考虑小车动力学）
                            kf_speed = np.sqrt(fvx**2 + fvy**2)
                            intercept = planner.solve(
                                block_x=fx, block_y=fy,
                                block_vx=fvx, block_vy=fvy,
                                block_ax=fax, block_ay=fay,
                                car_x=chassis_x, car_y=chassis_y,
                                car_v=kf_speed if kf_speed > 0 else 1.0,
                            )

                            if intercept and intercept["feasible"]:
                                target_x, target_y = int(intercept["x"]), int(intercept["y"])
                                T_solve = intercept["T"]
                            else:
                                # 规划无解 → 用当前位置
                                target_x, target_y = filtered_center
                                T_solve = 0.0

                            vg = VisionToGimbal(
                                target=1,
                                x=target_x,
                                y=target_y,
                                color=int(detector.final_color),
                                radius=detector.last_radius,
                            )
                            q.put(vg)

                            cur_offset = np.sqrt((fx - cx)**2 + (fy - cy)**2)
                            if cur_offset <= CENTER_TOLERANCE:
                                print(f"物块已对准 (偏移={cur_offset:.0f}px)，"
                                      f"目标{target_index + 1}/{len(target_colors)} 完成")
                                C_1 = detector.final_color
                                detection_sent = True
                                sent_time = time.time()

                                if target_index + 1 >= len(target_colors):
                                    print("所有目标检测完毕")
                                else:
                                    print(f"等待指令切换下一个目标 "
                                          f"(当前{target_index + 1}/{len(target_colors)})")
                                    waiting_for_next = True
                            else:
                                cv2.putText(hik_fram,
                                    f"闭环对准中... 偏移{cur_offset:.0f}px T={T_solve:.2f}s→({target_x},{target_y})",
                                    (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 200, 255), 1)

                    else:
                        # 未检测到物块，显示提示
                        cv2.putText(hik_fram, f"寻找 {target_colors[target_index] if target_index < len(target_colors) else '?'} 号物块中...",
                                    (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        if last_detection_time is not None:
                            elapsed = (current_time - last_detection_time) / cv2.getTickFrequency() * 1000
                            if elapsed > detector.timeout_settings['timeout_ms']:
                                pass

                cv2.imshow("hik", hik_fram)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("用户手动终止")
    finally:
        q.put(None)
        sending_thread.join(timeout=2)

        if cap and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

        try:
            cam.MV_CC_StopGrabbing()
        except Exception:
            pass
        try:
            cam.MV_CC_CloseDevice()
        except Exception:
            pass
        try:
            cam.MV_CC_DestroyHandle()
        except Exception:
            pass

        if _serial_reader:
            try:
                _serial_reader.close()
            except Exception:
                pass
        if _serial_comm:
            try:
                _serial_comm.stop_chassis_recv()
                _serial_comm.close()
            except Exception:
                pass

        print("资源已释放")


if __name__ == "__main__":
    main()
