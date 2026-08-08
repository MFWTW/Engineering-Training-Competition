import serial
import threading
import struct
import time
import logging
from typing import List, Optional

PACK_FORMAT = "<2B B 4H H H B H 2B"  # 小端: head(2B) target(1B) QR(8B) x(2B) y(2B) color(1B) radius(2B) tail(2B)
PACK_SIZE = struct.calcsize(PACK_FORMAT)  # 20 字节

logger = logging.getLogger("Gimbal")

class VisionToGimbal:
    def __init__(self, target: int = 0, QR: List[int] = [0, 0, 0, 0], x: int = 0, y: int = 0, color: int = 0, radius: int = 0):
        self.head: bytes = b"\x53\x50"
        self.target_: int = target          # uint8_t, 0~255
        self.QR_: List[int] = QR            # uint16_t × 4, 0~65535
        self.x_: int = x                    # uint16_t, 0~65535
        self.y_: int = y                    # uint16_t, 0~65535
        self.color_: int = color            # uint8_t, 0~255
        self.radius_: int = radius          # uint16_t, 0~65535
        self.tail: bytes = b"\xAA\x66"

    def pack(self) -> bytes:
        """序列化二进制：head + target + QR[] + x + y + color + radius + tail"""
        qr = [int(x) for x in self.QR_[:4]]
        data = struct.pack(
            PACK_FORMAT,
            self.head[0], self.head[1],
            self.target_,
            qr[0], qr[1], qr[2], qr[3],
            self.x_, self.y_,
            self.color_,
            self.radius_,
            self.tail[0], self.tail[1]
        )
        logger.info(
            f"[发送包] target={self.target_} QR={qr} "
            f"x={self.x_} y={self.y_} color={self.color_} radius={self.radius_} "
            f"hex={data.hex(' ')} len={len(data)}"
        )
        return data

class GimbalToVision:
    """底盘→上位机 数据接收与解析"""

    RECV_FORMAT = "<2B B H H h 2B"  # head(2) target(1) chassis_x(2) chassis_y(2) theta(2) tail(2)
    RECV_SIZE = struct.calcsize(RECV_FORMAT)  # 11 字节

    def __init__(self, target: int = 0, chassis_x: int = 0, chassis_y: int = 0, theta: int = 0):
        self.head: bytes = b"\x53\x50"
        self.target_: int = target          # uint8_t
        self.chassis_x: int = chassis_x      # uint16_t，底盘 X
        self.chassis_y: int = chassis_y      # uint16_t，底盘 Y
        self.theta: int = theta              # int16_t，底盘航向角（度×10 或 原始值）
        self.tail: bytes = b"\xAA\x66"
        self.timestamp: float = 0.0          # 上位机收到时的时间戳（秒）

    def pack(self) -> bytes:
        """序列化二进制（向底盘发送确认/指令时用）"""
        return struct.pack(
            self.RECV_FORMAT,
            self.head[0], self.head[1],
            self.target_,
            self.chassis_x, self.chassis_y,
            self.theta,
            self.tail[0], self.tail[1],
        )

    @classmethod
    def unpack(cls, data: bytes) -> "GimbalToVision | None":
        """从底盘收到的原始字节解析为 GimbalToVision 对象"""
        if len(data) < cls.RECV_SIZE:
            return None
        try:
            h0, h1, target, cx, cy, theta, t0, t1 = struct.unpack(cls.RECV_FORMAT, data[:cls.RECV_SIZE])
            if h0 != 0x53 or h1 != 0x50:
                return None
            if t0 != 0xAA or t1 != 0x66:
                return None
            result = cls(target=target, chassis_x=cx, chassis_y=cy, theta=theta)
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
        
    def read(self, size: int = 13) -> bytes:
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
        """接收线程主循环：按 9 字节帧解析底盘数据"""
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