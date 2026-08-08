import cv2
import sys
import numpy as np
from ctypes import *
 
from hikrobot.MvCameraControl_class import *
from hikrobot.MvCameraControl_class import PixelType_Gvsp_Mono8, PixelType_Gvsp_BayerRG8
 
 
def enum_devices():
    """
    枚举所有连接的相机设备
    """
    devicelist = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
 
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, devicelist)
    if ret != 0:
        print("枚举设备失败！ret[0x%x]" % ret)
        return None
 
    if devicelist.nDeviceNum == 0:
        print("未发现设备！")
        return None
 
    print("找到 %d 个设备：" % devicelist.nDeviceNum)
 
    for i in range(devicelist.nDeviceNum):
        mvcc_dev_info = cast(devicelist.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
 
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print("\nGigE 设备 [%d]" % i)
            model_name = ''.join(chr(c) for c in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName if c != 0)
            ip = mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp
            ip_str = '%d.%d.%d.%d' % ((ip >> 24) & 0xff, (ip >> 16) & 0xff, (ip >> 8) & 0xff, ip & 0xff)
            print("型号名称：%s" % model_name)
            print("IP地址：%s" % ip_str)
 
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("\nUSB 设备 [%d]" % i)
            model_name = ''.join(chr(c) for c in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName if c != 0)
            serial_number = ''.join(chr(c) for c in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber if c != 0)
            print("型号名称：%s" % model_name)
            print("序列号：%s" % serial_number)
 
    return devicelist
 
 #1440*1080
def create_camera_handle(device_list, index=0, width=1440, height=1080):
    """
    创建指定索引的相机句柄
    """
    if index >= device_list.nDeviceNum:
        print("选择的设备序号超出范围！")
        return None
 
    cam = MvCamera()
    dev_info = cast(device_list.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents
 
    ret = cam.MV_CC_CreateHandle(dev_info)
    if ret != 0:
        print("创建相机句柄失败！ret[0x%x]" % ret)
        return None
 
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("打开设备失败！ret[0x%x]" % ret)
        cam.MV_CC_DestroyHandle()
        return None
    
    # ====== 设置分辨率（打开设备后、采集前） ======
    if width is not None:
        ret = cam.MV_CC_SetIntValueEx("Width", width)
        if ret != 0:
            print(f"设置宽度失败！ret[0x{ret:x}]")
    if height is not None:
        ret = cam.MV_CC_SetIntValueEx("Height", height)
        if ret != 0:
            print(f"设置高度失败！ret[0x{ret:x}]")
 
    if dev_info.nTLayerType == MV_GIGE_DEVICE:
        packet_size = cam.MV_CC_GetOptimalPacketSize()
        if int(packet_size) > 0:
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
            if ret != 0:
                print("设置包大小失败！ret[0x%x]" % ret)
        else:
            print("获取最佳包大小失败！ret[0x%x]" % packet_size)
 
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        print("关闭触发模式失败！ret[0x%x]" % ret)
 
    return cam
 
 
def capture_video_stream(cam):
    """
    持续获取图像帧并显示为视频流
    """
    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
 
    ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
    if ret != 0:
        print("get payload size fail! ret[0x%x]" % ret)
        return False
 
    nPayloadSize = stParam.nCurValue
    data_buf = (c_ubyte * nPayloadSize)()
 
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("start grabbing fail! ret[0x%x]" % ret)
        return False
 
    stDeviceList = MV_FRAME_OUT_INFO_EX()
    memset(byref(stDeviceList), 0, sizeof(stDeviceList))
 
    try:
        while True:
            ret = cam.MV_CC_GetOneFrameTimeout(byref(data_buf), nPayloadSize, stDeviceList, 1000)
            if ret == 0:
                width = stDeviceList.nWidth
                height = stDeviceList.nHeight
                pixel_type = stDeviceList.enPixelType
 
                data_array = np.ctypeslib.as_array(data_buf)
 
                if pixel_type == PixelType_Gvsp_Mono8:
                    frame = data_array.reshape(height, width)
                elif pixel_type == PixelType_Gvsp_BayerRG8:
                    frame = data_array.reshape(height, width)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_RG2RGB)
                else:
                    print("Unsupported pixel format: 0x%x" % pixel_type)
                    continue
 
                frame = cv2.resize(frame, (640, 480))
                cv2.imshow("Camera Frame", frame)
 
                key = cv2.waitKey(1)
                if key != -1:
                    break
    except KeyboardInterrupt:
        print("用户中断")
 
    return True

def read_frame(cam, timeout_ms=100):
    """
    从海康相机读取一帧，返回 numpy BGR 图像 或 None
    调用前需先 StartGrabbing，见 create_and_start_camera()
    """
    stDeviceList = MV_FRAME_OUT_INFO_EX()
    memset(byref(stDeviceList), 0, sizeof(stDeviceList))

    ret = cam.MV_CC_GetOneFrameTimeout(
        byref(cam._data_buf), cam._payload_size, stDeviceList, timeout_ms
    )
    if ret != 0:
        return None

    width = stDeviceList.nWidth
    height = stDeviceList.nHeight
    pixel_type = stDeviceList.enPixelType

    data_array = np.ctypeslib.as_array(cam._data_buf)

    if pixel_type == PixelType_Gvsp_Mono8:
        return data_array.reshape(height, width)
    elif pixel_type == PixelType_Gvsp_BayerRG8:
        frame = data_array.reshape(height, width)
        return cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
    else:
        print(f"不支持的像素格式: 0x{pixel_type:x}")
        return None

def start_grabbing(cam):
    """开始取流，把 PayloadSize 和缓冲区挂到 cam 上"""
    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

    ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
    if ret != 0:
        print(f"获取 PayloadSize 失败! ret[0x{ret:x}]")
        return False

    cam._payload_size = stParam.nCurValue
    cam._data_buf = (c_ubyte * cam._payload_size)()

    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print(f"开始取流失败! ret[0x{ret:x}]")
        return False
    return True
 
if __name__ == "__main__":
    # 枚举设备
    device_list = enum_devices()
    if not device_list:
        sys.exit()
 
    # 创建相机句柄
    cam = create_camera_handle(device_list, 0)
    if not cam:
        sys.exit()
 
    # 获取视频流
    success = capture_video_stream(cam)
    if not success:
        print("视频流采集/显示失败")
 
    # 清理资源
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
 
    print("资源已释放")