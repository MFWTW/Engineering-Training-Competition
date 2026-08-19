import serial
import threading
import struct
import time
import logging
from typing import List, Optional

PACK_FORMAT = "<2B B B B h h H 2B"  # 小端: head(2B) target(1B) action(1B) capture(1B) chassis_x(2B) chassis_y(2B) gripper(2B) tail(2B)
PACK_SIZE = struct.calcsize(PACK_FORMAT)  # 13 字节

logger = logging.getLogger("Gimbal")


class VisionToGimbal:
    def __init__(self, target: int = 0, action: int = 0, capture: bool = False,
                 chassis_x_mm: int = 0, chassis_y_mm: int = 0, gripper_mm: int = 0):
        self.head: bytes = b"\x53\x50"
        self.target_: int = target          # uint8_t, 0~255
        self.action_: int = int(action)     # uint8_t, 0=启动/空闲, 1=抓取, 2=放置
        self.capture_: int = 1 if capture else 0  # uint8_t, 0=跟踪中/未抓取, 1=请求抓取
        self.chassis_x_mm: int = int(chassis_x_mm)  # int16_t，底盘左右移动量，正=左，负=右
        self.chassis_y_mm: int = int(chassis_y_mm)  # int16_t，底盘前后移动量，正=前，负=后
        self.gripper_mm: int = int(gripper_mm)      # uint16_t，夹爪伸出距离
        self.tail: bytes = b"\xAA\x66"

    def pack(self) -> bytes:
        """序列化二进制：head + target + action + capture + 底盘/夹爪 + tail"""
        capture = 1 if self.capture_ else 0
        data = struct.pack(
            PACK_FORMAT,
            self.head[0], self.head[1],
            self.target_,
            self.action_,
            capture,
            self.chassis_x_mm, self.chassis_y_mm, self.gripper_mm,
            self.tail[0], self.tail[1]
        )
        logger.info(
            f"[发送包] target={self.target_} action={self.action_} capture={capture} "
            f"chassis=({self.chassis_x_mm},{self.chassis_y_mm})mm gripper={self.gripper_mm}mm "
            f"hex={data.hex(' ')} len={len(data)}"
        )
        return data

class GimbalToVision:
    """底盘→上位机 数据接收与解析"""

    RECV_FORMAT = "<2B H H h h B B B 2B"  # head(2) chassis_x(2) chassis_y(2) chassis_vx(2) chassis_vy(2) capture_ack(1) finish_capture(1) arrived(1) tail(2)
    RECV_SIZE = struct.calcsize(RECV_FORMAT)  # 15 字节

    def __init__(self, chassis_x: int = 0, chassis_y: int = 0,
                 chassis_vx: int = 0, chassis_vy: int = 0,
                 capture_ack: int = 0, finish_capture: int = 0, arrived: int = 0):
        self.head: bytes = b"\x53\x50"
        self.chassis_x: int = chassis_x      # uint16_t，底盘 X（mm）
        self.chassis_y: int = chassis_y      # uint16_t，底盘 Y（mm）
        self.chassis_vx: int = chassis_vx    # int16_t，底盘速度 X 分量（mm/s）
        self.chassis_vy: int = chassis_vy    # int16_t，底盘速度 Y 分量（mm/s）
        self.capture_ack: int = int(capture_ack)        # uint8_t, 1=下位机已收到抓取请求
        self.finish_capture: int = int(finish_capture)  # uint8_t, 1=下位机抓取完成
        self.arrived: int = int(arrived)                # uint8_t, 1=已到达指定区域
        self.tail: bytes = b"\xAA\x66"
        self.timestamp: float = 0.0          # 上位机收到时的时间戳（秒）

    @property
    def chassis_speed(self) -> float:
        """底盘实际速率（原始值单位，与 vx/vy 相同）"""
        return (self.chassis_vx ** 2 + self.chassis_vy ** 2) ** 0.5

    def pack(self) -> bytes:
        """序列化二进制（向底盘发送确认/指令时用）"""
        return struct.pack(
            self.RECV_FORMAT,
            self.head[0], self.head[1],
            self.chassis_x, self.chassis_y,
            self.chassis_vx, self.chassis_vy,
            self.capture_ack,
            self.finish_capture,
            self.arrived,
            self.tail[0], self.tail[1],
        )

    @classmethod
    def unpack(cls, data: bytes) -> "GimbalToVision | None":
        """从底盘收到的原始字节解析为 GimbalToVision 对象"""
        if len(data) < cls.RECV_SIZE:
            return None
        try:
            h0, h1, cx, cy, vx, vy, capture_ack, finish_capture, arrived, t0, t1 = struct.unpack(
                cls.RECV_FORMAT, data[:cls.RECV_SIZE]
            )
            if h0 != 0x53 or h1 != 0x50:
                return None
            if t0 != 0xAA or t1 != 0x66:
                return None
            result = cls(chassis_x=cx, chassis_y=cy,
                         chassis_vx=vx, chassis_vy=vy,
                         capture_ack=capture_ack, finish_capture=finish_capture,
                         arrived=arrived)
            result.timestamp = time.time()
            return result
        except struct.error:
            return None



class SerialComm:
    def __init__(self, port: str = "/dev/ttyACM0", baudrate: int = 115200, max_retry: int = 10, retry_delay: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.max_retry = max_retry
        self.retry_delay = retry_delay
        self.ser: Optional[serial.Serial] = None
        self.packet: VisionToGimbal = VisionToGimbal()
        self.connected_ = False
        self.quit_ = False
        self._lock = threading.Lock()

        # 底盘数据接收
        self._recv_thread: Optional[threading.Thread] = None
        self._latest_chassis: Optional[GimbalToVision] = None
        self._chassis_lock = threading.Lock()

        #初始化尝试打开串口
        self.open()
        # self.ser: serial.Serial = serial.Serial(port, baudrate, timeout=0.5)
        # self.packet: VisionToGimbal = VisionToGimbal()

    def open(self) -> bool:
        """打开串口"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.5)
            self.connected_ = True
            logger.info(f"Serial {self.port} opened successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to open serial {self.port}: {e}")
            self.connected_ = False
            return False
        
    def close(self):
        """关闭串口"""
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
                logger.info("Serial closed.")
            except Exception as e:
                logger.warning(f"Error closing serial: {e}")
        self.connected_ = False

    def clear_buffer(self):
        """清空读写缓冲区（类似 C++ queue_.clear()）"""
        if self.ser and self.ser.is_open:
            try:
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
                logger.info("Serial buffers cleared.")
            except Exception as e:
                logger.warning(f"Failed to clear buffers: {e}")

    def reconnect(self):
        """
        自动尝试重连串口，最多 max_retry 次
        失败会延迟 retry_delay 秒再尝试
        成功后清空缓冲区，确保状态同步
        """
        with self._lock:
            self.connected_ = False

            for i in range(self.max_retry):
                if self.quit_:
                    logger.warning("[Gimbal] quit_ flag set, aborting reconnect.")
                    break

            logger.warning(f"[Gimbal] Reconnecting serial, attempt {i + 1}/{self.max_retry}...")

            # 先尝试关闭
            try:
                self.close()
                time.sleep(self.retry_delay)
            except Exception:
                pass
            
            # 尝试重新打开
            try:
                self.open()
                self.connected_ = True
                self.clear_buffer()
                logger.info("[Gimbal] Reconnected serial successfully.")
                return True
            except Exception as e:
                logger.warning(f"[Gimbal] Reconnect failed: {e}")
                time.sleep(self.retry_delay)

 
    def send(self, VG: Optional[VisionToGimbal] = None) -> bool:
        """发送数据包，如果断开则自动重连"""
        if VG is None:
            VG = self.packet

        if not self.connected_ or not self.ser or not self.ser.is_open:
            logger.warning("[Gimbal] Serial not connected, attempting reconnect...")
            if not self.reconnect():
                return False

        try:
            with self._lock:
                self.ser.write(VG.pack())
            return True
        except Exception as e:
            logger.error(f"[Gimbal] Send failed: {e}")
            self.connected_ = False
            return False
        
    def read(self, size: int = GimbalToVision.RECV_SIZE) -> bytes:
        """读取数据，如果断开则自动重连"""
        if not self.connected_ or not self.ser or not self.ser.is_open:
            logger.warning("[Gimbal] Serial not connected, attempting reconnect...")
            if not self.reconnect():
                return b""

        try:
            with self._lock:
                return self.ser.read(size)
        except Exception as e:
            logger.error(f"[Gimbal] Read failed: {e}")
            self.connected_ = False
            return b""

    # ──────── 底盘数据接收线程 ────────

    def start_chassis_recv(self):
        """启动后台线程，持续接收底盘回传数据"""
        if self._recv_thread is not None and self._recv_thread.is_alive():
            return
        self.quit_ = False
        self._recv_thread = threading.Thread(target=self._chassis_recv_loop, daemon=True)
        self._recv_thread.start()
        logger.info("[Gimbal] 底盘数据接收线程已启动")

    def stop_chassis_recv(self):
        """停止接收线程"""
        self.quit_ = True
        if self._recv_thread:
            self._recv_thread.join(timeout=2)
            self._recv_thread = None

    def get_chassis_data(self) -> Optional[GimbalToVision]:
        """线程安全地获取最新底盘数据"""
        with self._chassis_lock:
            return self._latest_chassis

    def _chassis_recv_loop(self):
        """接收线程主循环：按 15 字节帧解析底盘数据"""
        buf = b""
        while not self.quit_:
            try:
                with self._lock:
                    if self.ser and self.ser.is_open and self.ser.in_waiting > 0:
                        chunk = self.ser.read(self.ser.in_waiting)
                        buf += chunk
                if len(buf) >= GimbalToVision.RECV_SIZE:
                    # 找帧头
                    idx = buf.find(b"\x53\x50")
                    if idx < 0:
                        buf = buf[-2:]  # 保留末尾，可能跨帧
                        continue
                    if idx > 0:
                        buf = buf[idx:]  # 丢弃帧头前的垃圾
                    if len(buf) >= GimbalToVision.RECV_SIZE:
                        chassis = GimbalToVision.unpack(buf[:GimbalToVision.RECV_SIZE])
                        buf = buf[GimbalToVision.RECV_SIZE:]
                        if chassis is not None:
                            with self._chassis_lock:
                                self._latest_chassis = chassis
                else:
                    time.sleep(0.01)
            except Exception as e:
                logger.warning(f"[Gimbal] recv loop error: {e}")
                time.sleep(0.1)

    def __del__(self):
        """析构时关闭串口"""
        self.close()
