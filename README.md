# 物块识别追踪系统（RoboMaster 视觉）

基于海康工业相机 + 颜色轮廓检测 + 卡尔曼滤波 + 博弈拦截规划的闭环对准系统。
仓库同时包含手写数字识别、HSV 调参工具、QR 码工具，以及旧版整机程序（遗留参考）。

---

## 项目概览

| 模块 | 入口 | 说明 |
|---|---|---|
| 物块识别追踪（主系统） | `src.py` | QR 扫序列 → 海康相机逐色检测 → KF 滤波 → 拦截规划 → 串口闭环对准 |
| 手写数字识别 | `felling_number.py` | 海康实时取流 + 轻量 CNN（MNIST）识别数字 |
| HSV 调参工具 | `hsv_tuner.py` | Trackbar 实时调节 6 色 HSV 阈值 |
| QR 工具 | `create_qr.py` | 生成测试二维码（如 `156+123+516+231`） |
| 旧版整机程序 | `example_code.py` | 旧架构整机代码（含多种定标流程），仅作参考 |

---

## 目录结构

```
├── src.py                    # 主程序：QR → 物块识别追踪闭环
├── felling_number.py         # 手写数字识别（海康实时 + TinyCNN）
├── model.py                  # TinyDigitCNN 模型定义（<25K 参数）
├── tiny_digit_cnn.pth        # 数字识别训练权重
├── hsv_tuner.py              # HSV 阈值实时调参工具
├── example_code.py           # 旧版整机程序（含物块/色环/码垛定标，遗留）
├── common_camera.py          # USB 摄像头辅助（QR 扫描阶段）
├── preprocessing.py          # 图像预处理（Otsu 二值化）
├── scan_QRcode_andlist.py    # QR 码扫描与目标序列解析
├── hikrobot_camera.py        # 海康工业相机 SDK 封装
├── hikrobot/                 # 海康 SDK 原生 Python 绑定
├── felling_color.py          # 物块颜色检测器（HSV 阈值 + 轮廓 + 稳定性判定）
├── kalman_tracker.py         # 6 维卡尔曼滤波（状态估计）
├── intercept_planner.py      # 拦截点博弈规划器（小车动力学）
├── gimbal.py                 # 串口通信（发送 + 底盘接收 + 时间戳）
├── create_qr.py              # 生成测试二维码
├── my_qrcode.png             # 生成的二维码样例
├── block/                    # 6 色物块示例图片
├── output_xgb_final/         # （无关遗留）XGBoost 信号强度预测结果
├── output_xgb_optuna/        # （无关遗留）Optuna 调参输出
├── AMD_YYDS.json             # （无关遗留）ComfyUI FLUX 工作流
├── node.tar.xz               # （无关遗留）Node.js v22 二进制包
└── conda/                    # 本地打包的 Python 3.10 环境
```

---

## 一、主程序 `src.py` —— 物块识别追踪闭环

### 工作流程（两阶段）

1. **QR 扫描阶段**：USB 摄像头取帧 → Otsu 二值化 → `pyzbar` 解码二维码
   （形如 `156+123+516+231`，4 组数字）→ 得到目标颜色序列；
2. **物块识别阶段**：关闭 USB 摄像头，切换到海康工业相机，按颜色序列逐个目标执行闭环对准。

### 闭环控制流程 (30fps)

```
┌────────────────────────────────────────────────┐
│  每帧循环（~30Hz）：                             │
│                                                 │
│ ① 海康取帧 → HSV 掩膜 → 轮廓 → minEnclosingCircle │
│ ② 时间戳同步：底盘位姿 (x,y,θ) 配对             │
│ ③ 稳定性判定（连续 N 帧圆心/颜色一致）           │
│ ④ KF.predict() → KF.update(Z) → [x,y,vx,vy,ax,ay] │
│ ⑤ KF.predict_future(T=2s) → 未来 6 步轨迹       │
│ ⑥ InterceptPlanner.solve(KF状态, 底盘位姿)      │
│    → 博弈 T → 可行拦截点                         │
│ ⑦ 串口发送 拦截点 → 小车调整                    │
│ ⑧ 物块距中心 ≤ 容差? → 完成 / 继续              │
└────────────────────────────────────────────────┘
```

### 全部可调参数速查表

#### src.py —— 主控制参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `CONTROL_MODE` | `"manual"` | 控制模式：`"manual"` 按键切换 / `"stm"` 串口指令切换 |
| `HIK_DETECTION_ROI` | `[400,240,560,420]` | ROI 区域 `[x,y,w,h]`（当前仅用于可视化，检测为全图） |
| `CENTER_TOLERANCE` | `5` | 对准容差（px），物块圆心距图像中心 ≤ 此值即完成抓取 |
| `auto_switch_timeout` | `10.0` | STM32 超时自动切换（秒），手动模式无效 |
| `min_display_time` | `2.0` | 手动模式按键冷却（秒），防连续误触 |
| 稳定度阈值 | `15` | 需连续 N 帧圆心稳定才判定 `is_stable` |
| 位移容忍 | `20` | 圆心帧间最大移动（px），超过重置稳定性计数 |
| 颜色稳定阈值 | `8` | 需 N 帧颜色一致才确认颜色识别 |

#### kalman_tracker.py —— 卡尔曼滤波参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `dt` | `1/30` | 帧间隔（秒），30fps=0.033s，实际帧率由主循环自适应 |
| `q_acc` | `5.0` | 过程噪声 (px/s²)²，越大滤波越信任测量、越不平滑 |
| `meas_std` | `8.0` | 测量标准差（px），视觉检测典型抖动，越大滤波越平滑 |
| 初始 `P` | `100*I` | 初始协方差，越大收敛越快但初期波动大 |

#### intercept_planner.py —— 拦截规划参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `car_max_speed` | `200.0` | 小车最大速度（px/s），影响能否追上物块 |
| `car_accel` | `100.0` | 小车加速度（px/s²） |
| `car_decel` | `150.0` | 小车减速度（px/s²），影响刹车距离 |
| `time_resolution` | `0.05` | T 搜索步长（秒）；`src.py` 实例化时传 `0.02` |
| `max_predict_time` | `5.0` | 最大预测时间（秒），超此范围认为无法拦截 |
| `tolerance` | `0.1` | 收敛容差（秒），\|T_car - T\| ≤ 此值即可行 |

#### felling_color.py —— 颜色检测参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `min_radius` | `50` | 最小半径（px），排除小噪点 |
| `max_radius` | `450` | 最大半径（px），排除过大非目标 |
| `param1 / param2` | `25 / 25` | 轮廓筛选辅助参数（面积门限计算） |
| `kernel_size` | `3` | 形态学核大小 |
| 各颜色 HSV 阈值 | 见文件 L8-19 | 6 种颜色的 HSV `lower/upper` 范围（红色为双区间） |
| 各颜色形态学 | 见文件 L33-41 | 每种颜色 `(erode_iter, dilate_iter)` |

#### gimbal.py —— 串口参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `port` | `"/dev/ttyACM0"` | 串口设备路径 |
| `baudrate` | `115200` | 波特率 |

### 画面可视化

| 元素 | 颜色 | 含义 |
|---|---|---|
| 绿色矩形 + "ROI" | 🟢 | ROI 检测区域（可视化） |
| 灰色细圈 + 小灰点 | ⬜ | 原始视觉测量（带噪声） |
| 彩色粗圈 + 实心点 | 随颜色 | KF 滤波后的平滑位置 |
| 彩色标签 | 随颜色 | RED / GREEN / BLUE / LIGHT_BLUE / BLACK / YELLOW |
| 白色 `＋` | ⬜ | 图像中心（对准目标） |
| 黄色箭头轨迹 | 🟡 | KF 运动学预测未来 2s/6 步 |
| 红色箭头 + 红点 | 🔴 | 拦截向量：中心 → 轨迹最近点 |
| 青色 `V=XXpx/s` | 🔵 | KF 估计速度（像素/秒） |
| 红色 `intercept:XXpx` | 🔴 | 中心到拦截点距离 |
| 灰色 `chassis:(X,Y) th=θ` | ⬜ | 右上角，底盘位姿 |
| 绿色 `KF追踪中` | 🟢 | 稳定性进度 |
| 青色 `闭环对准中... T=X.XXs` | 🔵 | 博弈 T 值 + 拦截坐标 |

### 串口协议

#### 上位机 → 云台（VisionToGimbal，20 字节）

| 字段 | 字节 | 类型 | 说明 |
|---|---|---|---|
| head | 2B | uint8×2 | `0x53 0x50` |
| target | 1B | uint8 | 0=QR, 1=物块拦截坐标 |
| QR[0..3] | 8B | uint16×4 | 颜色序列（target=0） |
| x | 2B | uint16 | 拦截点 X |
| y | 2B | uint16 | 拦截点 Y |
| color | 1B | uint8 | 颜色代码 1~6 |
| radius | 2B | uint16 | 物块半径 |
| tail | 2B | uint8×2 | `0xAA 0x66` |

#### 底盘 → 上位机（GimbalToVision，11 字节）

| 字段 | 字节 | 类型 | 说明 |
|---|---|---|---|
| head | 2B | uint8×2 | `0x53 0x50` |
| target | 1B | uint8 | 目标标识 |
| chassis_x | 2B | uint16 | 底盘 X |
| chassis_y | 2B | uint16 | 底盘 Y |
| theta | 2B | int16 | 航向角 |
| tail | 2B | uint8×2 | `0xAA 0x66` |

底盘数据由 `SerialComm._chassis_recv_loop()` 后台线程持续接收，`unpack()` 时自动打时间戳。

### 运行方式

```bash
python3 src.py
```

| 按键 | 功能 |
|---|---|
| `q` | 退出 |
| `n` / 空格 | 切换下一个目标（手动模式） |

### 模块调用关系

```
src.py
 ├→ common_camera.py         USB 摄像头（QR 阶段）
 ├→ preprocessing.py         Otsu 二值化（QR 阶段）
 ├→ scan_QRcode_andlist.py   QR 解码 → 颜色序列
 ├→ felling_color.py         视觉检测（每帧）
 ├→ kalman_tracker.py        KF 滤波（每帧 predict+update）
 ├→ intercept_planner.py     拦截规划（稳定后每帧 solve）
 ├→ gimbal.py                串口收发（发送线程 + 底盘接收线程）
 └→ hikrobot_camera.py       海康取帧（物块阶段）
```

---

## 二、手写数字识别 `felling_number.py`

海康工业相机实时采集 + 轻量 CNN（MNIST）数字识别，模型定义在 `model.py`
（`TinyDigitCNN`，深度可分离卷积思路，参数量 < 25K，训练权重 `tiny_digit_cnn.pth`）。

预处理流水线：

```
灰度帧 → 高斯模糊 → OTSU 反二值化（白字黑底）→ 形态学闭运算
→ 最大轮廓定位 → 保持宽高比缩放至 20×20 → 28×28 画布居中
→ MNIST 标准化 (x/255 - 0.1307) / 0.3081 → 推理
```

### 运行方式

```bash
python3 felling_number.py
```

| 按键 | 功能 |
|---|---|
| `q` | 退出 |
| `s` | 保存当前灰度帧截图 |

窗口实时显示：原始灰度、识别结果与置信度、28×28 模型输入（放大 10 倍）、数字 ROI。

---

## 三、调试与辅助工具

### hsv_tuner.py —— HSV 阈值实时调参

调用海康工业相机实时取流（无海康时自动回退 USB 摄像头），Trackbar 实时调节各颜色
HSV 上下界，并排显示 原始+ROI / 掩码 / 掩码叠加。

```bash
python3 hsv_tuner.py
```

| 按键 | 功能 |
|---|---|
| `s` | 打印当前所有颜色的 `color_thresholds` 配置（可直接粘贴回 `felling_color.py`） |
| `r` | 开关 ROI 框显示 |
| `q` | 退出 |

### create_qr.py —— 生成测试二维码

用 `qrcode` 库生成形如 `156+123+516+231` 的二维码并保存为 `my_qrcode.png`，
用于 USB 摄像头 QR 扫描阶段的离屏测试。

```bash
python3 create_qr.py
```

---

## 四、旧版整机程序 `example_code.py`（遗留）

旧架构的单文件整机程序（约 1900 行），采用 `SerialInterruptHandler` 串口中断主循环
（`/dev/ttyUSB0`，9600 波特），功能包括：

- QR 检测（`run_erweima`，`/dev/video_xia0`）；
- 物块圆心定标（`run_wukuaiyuanxin_1/2/A/xuanzequyu`）；
- 色环 / 码垛两次定标（`run_sehuanyuanxin2(_centered)`、`run_maduoyuanxin2(_centered)`）；
- 边界线检测（`detect_boundary_line`）；
- 串口指令：`0`/`r` 重启、`end` 结束、`5`/`6`/`b`/`c` 等任务切换。

新功能开发请以 `src.py` 模块化架构为准，此文件仅作历史参考。

---

## 五、与本项目无关的遗留文件

| 文件/目录 | 内容 | 建议 |
|---|---|---|
| `output_xgb_final/`、`output_xgb_optuna/` | XGBoost 信号强度（RSRP）预测任务的产物：逐点预测 CSV、`best_params.json`、Optuna 调参日志 | 与视觉系统无关，可移出仓库 |
| `AMD_YYDS.json` | ComfyUI FLUX 文生图工作流 | 无关文件，可删除/移出 |
| `node.tar.xz` | Node.js v22 官方二进制压缩包（约 27MB） | 无关文件，可删除/移出 |
| `conda/` | 本地打包的 Python 3.10 虚拟环境（约 5.2GB） | 应通过 `requirements` 复现，不应入库 |

---

## 依赖环境

```text
python3.10
opencv-python
numpy
pyserial
pyzbar          # QR 解码
qrcode          # 生成测试二维码
torch           # 数字识别
```

海康工业相机依赖 `hikrobot/` 目录下的 SDK 原生绑定（MvCameraControl）。
