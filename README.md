# 物块识别追踪系统（RoboMaster 视觉）

基于双 USB 免驱摄像头 + 颜色轮廓检测 + 卡尔曼滤波 + 博弈拦截规划的闭环对准系统。
仓库同时包含手写数字识别、HSV 调参工具、QR 码工具，以及旧版整机程序（遗留参考）。

---

## 项目概览

| 模块 | 入口 | 说明 |
|---|---|---|
| 物块识别追踪（主系统） | `src.py` | QR 扫序列 → USB 相机逐色检测 → KF 滤波 → 拦截规划 → 串口闭环对准 |
| 坐标/夹爪决策 | `transformer.py` | 相机坐标→世界坐标，按夹爪最远/最近距离决策底盘和夹爪动作 |
| 相机标定 | `calibrate_camera.py` | 棋盘格采集 + 内参标定，输出 `camera_calibration.json` |
| px_per_mm 标定 | `calibrate_px_per_mm.py` | 标尺法测量下位机 mm → 图像 px 的换算倍率 |
| 坐标换算验证 | `validate_transform.py` | 点击物块计算底盘/夹爪指令，并与实测距离对比 |
| 手写数字识别 | `felling_number.py` | USB 实时取流 + 轻量 CNN（MNIST）识别数字 |
| HSV 调参工具 | `hsv_tuner.py` | Trackbar 实时调节 6 色 HSV 阈值 |
| QR 工具 | `create_qr.py` | 生成测试二维码（如 `156+123+516+231`） |
| 旧版整机程序 | `example_code.py` | 旧架构整机代码（含多种定标流程），仅作参考 |

---

## 目录结构

```
├── src.py                    # 主程序：QR → 物块识别追踪闭环
├── felling_number.py         # 手写数字识别（USB 实时 + TinyCNN）
├── model.py                  # TinyDigitCNN 模型定义（<25K 参数）
├── tiny_digit_cnn.pth        # 数字识别训练权重
├── hsv_tuner.py              # HSV 阈值实时调参工具
├── example_code.py           # 旧版整机程序（含物块/色环/码垛定标，遗留）
├── common_camera.py          # 双 USB 摄像头统一配置与打开
├── preprocessing.py          # 图像预处理（Otsu 二值化）
├── scan_QRcode_andlist.py    # QR 码扫描与目标序列解析
├── felling_color.py          # 物块颜色检测器（HSV 阈值 + 轮廓 + 稳定性判定）
├── config.yaml               # 颜色阈值 / 形态学 / 稳定性 / 单位换算等统一参数（YAML）
├── kalman_tracker.py         # 6 维卡尔曼滤波（状态估计）
├── intercept_planner.py      # 拦截点博弈规划器（小车动力学）
├── transformer.py            # 相机→世界坐标变换 + 底盘/夹爪协同决策
├── calibrate_camera.py       # 棋盘格相机内参标定
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

### 工作流程（两轮抓取 + 放置）

1. **QR 扫描阶段**：USB 摄像头取帧 → Otsu 二值化 → `pyzbar` 解码二维码
   （形如 `156+123+516+231`，4 组数字）→ 解析为两轮任务：
   - 第 1 组：第 1 轮抓取颜色序列；
   - 第 2 组：第 1 轮放置编号；
   - 第 3 组：第 2 轮抓取颜色序列；
   - 第 4 组：第 2 轮放置编号。
2. **抓取阶段**：关闭二维码 USB 摄像头，切换到另一路物块检测 USB 摄像头；严格按抓取序列顺序，
   每帧只检测当前目标颜色（仅构建该颜色的 HSV 掩膜，减少计算量），
   识别到后跟踪并抓取，抓到的物块按“颜色 → 槽位”映射放入对应槽位；
3. **放置阶段**：本轮物块全部抓完后，下位机移动到放置区并回传 `arrived=1`；
   上位机识别同心圆环最内层数字，数字 n → 按“槽位 → 圆环编号”映射取出对应物块放置到该位置；
   三个位置全部放完后进入下一轮，全部轮次完成则退出。

### 代码流程图

```mermaid
flowchart TD
    START["程序启动"] --> QR["USB 扫描二维码"]
    QR --> R1G["第1轮抓取：按第1组颜色顺序抓完"]
    R1G --> R1M["下位机移动到放置区"]
    R1M --> R1P["第1轮放置：识别圆环数字，逐位放置"]
    R1P --> R2M["下位机移动到抓取区"]
    R2M --> R2G["第2轮抓取：按第3组颜色顺序抓完"]
    R2G --> R2M2["下位机移动到放置区"]
    R2M2 --> R2P["第2轮放置：识别圆环数字，逐位放置"]
    R2P --> DONE["全部完成，退出"]
```

> 串口发送由独立后台线程 `Sending2Gimbal` 完成：主循环把 `VisionToGimbal` 数据包放入队列，
> 发送线程串行打包并写串口，失败时自动重连，退出时写入 `None` 停止线程。
> 发送队列上限为 1 包，满时丢弃旧包，避免断线积压后重发过期坐标。
> 普通跟踪/对准包（capture=0）按 `TRACKING_SEND_INTERVAL` 节流，并且只有当
> 底盘/夹爪指令相对上次已发送值的变化超过死区（`CHASSIS_SEND_DEADBAND_MM` /
> `GRIPPER_DEADBAND_MM`）或到达心跳间隔时才发送，避免下位机按“增量移动量”
> 解析时被 100→90 这类微小变化反复打断；capture=1、阶段切换、区域移动和
> 重发包仍立即发送。底盘指令先乘以 `CHASSIS_P_GAIN` 再限幅，并按
> `CHASSIS_RAMP_STEP_MM` 做斜率限幅：指令每个发送周期只变化一小步，
> 起步/减速连续爬升，不会“动一下停一下”。

### 闭环控制流程 (30fps)

```
┌────────────────────────────────────────────────┐
│  每帧循环（~30Hz）：                             │
│                                                 │
│ ① USB 物块相机取帧 → 仅当前目标颜色的 HSV 掩膜 → 轮廓 → minEnclosingCircle │
│ ② 时间戳同步：底盘位姿 (x,y) + 速度 配对         │
│ ③ 稳定性判定（连续 N 帧圆心/颜色一致）           │
│ ④ KF.predict() → KF.update(Z) → [x,y,vx,vy,ax,ay] │
│ ⑤ KF.predict_future(T=2s) → 未来 6 步轨迹       │
│ ⑥ InterceptPlanner.solve(KF状态, 底盘位姿+速度)  │
│    → 博弈 T → 可行拦截点                         │
│ ⑦ 稳定后串口发送 拦截点(capture=0, 限幅+节流+死区) → 小车调整 │
│ ⑧ 物块距中心 ≤ 容差 → 发送 capture=1 请求抓取    │
│ ⑨ 下位机回传 finish_capture=1 → 切换下一目标     │
└────────────────────────────────────────────────┘
```

### 全部可调参数速查表

#### src.py —— 主控制参数

src.py 的可调参数已全部迁移到 [config.yaml](config.yaml) 对应分段，程序启动时自动读取：

| config.yaml 配置项 | 默认值 | 说明 |
|---|---|---|
| `control.mode` | `"manual"` | 切换模式：manual=手动（按 n/空格）；其他值=自动 |
| `control.auto_switch_timeout` | `10.0` | 非 manual 模式等待 `finish_capture` 的超时兜底切换（秒） |
| `control.center_tolerance_px` | `5` | x 轴（左右）对准容差（px）：\|目标x - 图像中心x\| ≤ 该值即请求抓取/放置 |
| `tracking.capture_resend_interval` | `1.0` | 未收到 `capture_ack` 时重发 `capture=1` 的间隔（秒） |
| `tracking.send_interval` | `0.1` | 普通跟踪/对准指令（capture=0）的最小发送间隔（秒）；capture=1 与阶段切换等事件包立即发送 |
| `tracking.chassis_p_gain` | `0.9` | 底盘比例增益：目标偏移 × 该系数后再下发，越靠近移动量越小，避免 0 附近过冲摆动 |
| `tracking.chassis_send_deadband_mm` | `1` | 底盘指令变化死区（mm）：目标偏移变化小于该值不重发，避免下位机增量执行被打断 |
| `tracking.gripper_deadband_mm` | `5` | 夹爪指令死区（mm） |
| `tracking.chassis_ramp_step_mm` | `4` | 平滑跟踪：每个发送周期底盘指令变化量上限（mm），按 `send_interval` 标定 |
| `tracking.send_heartbeat` | `5.0` | 普通跟踪包心跳间隔（秒），应大于 `tracking.send_interval`；`null` 禁用 |
| `display.max_width` / `max_height` | `800` / `540` | 显示窗口最大尺寸（px），宽或高超过时按同一比例缩小，仅影响显示 |
| `display.serial_overlay.enabled` | `true` | 在画面左下角叠加显示串口收发信息（只用英文/数字，避免中文乱码） |
| `display.serial_overlay.max_lines` | `4` | TX / RX 各保留并显示最近 N 条 |
| `protocol.idle_action` / `grab_action` / `place_action` | `0` / `1` / `2` | 串口 action 动作码（与下位机协议约定，一般不要改） |
| `safety.max_chassis_cmd_mm` | `2000` | 底盘单轴指令合理范围上限（mm），越界沿用上一帧有效指令 |
| `safety.max_gripper_mm` | `400` | 夹爪伸长量合理范围上限（mm） |
| `safety.max_chassis_step_mm` | `30` | 普通跟踪包单次底盘移动量上限（mm），防止一次给全量偏移过冲 |
| `logging.command_print_interval` | `0.5` | 指令打印最小间隔（秒），数值变化或超过间隔才打印 |
| `logging.warn_interval_s` | `1.0` | 坐标无效 / 命令全 0 警告打印最小间隔（秒） |
| `planner.car_max_speed_px_per_s` | `200.0` | 小车最大速度（px/s），影响能否追上物块 |
| `planner.car_accel_px_per_s2` | `100.0` | 小车加速度（px/s²） |
| `planner.car_decel_px_per_s2` | `150.0` | 小车减速度（px/s²），影响刹车距离 |
| `planner.time_resolution_s` | `0.02` | T 搜索步长（秒） |
| `planner.max_predict_time_s` | `5.0` | 最大预测时间（秒），超过视为无法拦截 |
| `planner.tolerance_s` | `0.1` | 收敛容差（秒）：\|T_car - T\| ≤ 该值认为可行 |

#### kalman_tracker.py —— 卡尔曼滤波参数

卡尔曼滤波参数统一放在 `config.yaml` 的 `kalman` 段，运行时自动读取，无需改代码。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `enabled` | `false` | 卡尔曼总开关：true 使用 6 维 KF；false 直接用当前帧检测中心，速度/加速度视为 0 |
| `dt` | `1/30` | 帧间隔（秒），30fps=0.033s |
| `q_acc` | `20.0` | 过程噪声 (px/s²)²，越大滤波越信任测量、越不平滑 |
| `meas_std` | `5.0` | 测量标准差（px），视觉检测典型抖动，越大滤波越平滑 |
| `initial_p` | `100.0` | 初始协方差，越大收敛越快但初期波动大 |
| `history_len` | `100` | 调试用历史记录条数 |
| `predict.horizon_s` | `2.0` | 未来轨迹预测时长（秒） |
| `predict.steps` | `6` | 预测轨迹点数（可视化步数） |
| `visualize.enabled` | `true` | 卡尔曼调试绘制总开关；`false` 时不再画原始/滤波圆、轨迹、拦截点等 |
| `visualize.draw_raw` | `true` | 是否绘制原始测量圈（细圈+灰点） |
| `visualize.draw_filtered` | `true` | 是否绘制滤波后位置（粗圈+实心点） |
| `visualize.draw_trajectory` | `true` | 是否绘制未来预测轨迹（青色折线） |
| `visualize.draw_intercept` | `true` | 是否绘制拦截点（红色箭头+点） |
| `visualize.draw_speed` | `true` | 是否绘制滤波速度 `V=xx px/s` |
| `visualize.draw_history` | `true` | 是否绘制滤波历史轨迹线（橙黄色） |
| `visualize.history_trail_len` | `60` | 历史轨迹保留最近 N 帧（≤ `history_len`） |

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

颜色阈值、颜色编号、形态学参数，以及圆检测 / 稳定性 / 超时等需要人工调参的变量，
统一放在 [config.yaml](config.yaml) 中，`felling_color.py` 运行时自动加载，
无需改代码。主要参数：

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `colors.<name>.lower/upper` | 见 YAML | 6 种颜色的 HSV 范围（红色为双区间 `lower1/upper1/lower2/upper2`） |
| `colors.<name>.code` | `"1"`~`"6"` | 发送给下位机的颜色代码 |
| `colors.<name>.morph` | `erode/dilate` | 每种颜色的形态学迭代次数 |
| `detection.circle.min_radius / max_radius` | `50 / 450` | 圆检测半径范围（px） |
| `detection.kernel_size` | `3` | 形态学核大小 |
| `detection.blur` | `5 / 2 / 2` | 高斯模糊核与 σ |
| `detection.stability.*` | 见 YAML | 物块位置稳定帧数 / 位移容忍 / 颜色连续稳定帧数 |
| `detection.timeout_ms` | `100` | 检测超时（ms） |
| `detection.detection_area` | `null` | 检测区域 `[x, y, w, h]`（原 `DETECTION_ROI`），`null` 为关闭 ROI（全图）；当前仅用于可视化，检测为全图 |

性能说明：调用 `detect(frame, target_code=...)` 时只构建目标颜色的 HSV 掩膜
（普通颜色 1 次 `inRange`，红色双区间 2 次），不再每帧构建 6 色掩膜；
主程序抓取阶段严格按序列逐色检测，减少计算量、提升处理帧率。

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
| 灰色 `chassis:(X,Y)mm V=XXmm/s` | ⬜ | 右上角，底盘位姿（mm）+ 实际速度（mm/s） |
| 绿色 `KF追踪中` | 🟢 | 稳定性进度 |
| 青色 `闭环对准中... T=X.XXs` | 🔵 | 博弈 T 值 + 拦截坐标 |

### 串口协议

#### 上位机 → 云台（VisionToGimbal，13 字节）

| 字段 | 字节 | 类型 | 说明 |
|---|---|---|---|
| head | 2B | uint8×2 | `0x53 0x50` |
| target | 1B | uint8 | 0=任务开始/区域移动, 1~3=物块槽位号 |
| action | 1B | uint8 | 0=启动/空闲, 1=抓取, 2=放置 |
| capture | 1B | uint8 | 0=跟踪/移动中, 1=执行动作（抓取或松爪放置） |
| chassis_x | 2B | int16 | 底盘左右移动量（mm，正=左，负=右） |
| chassis_y | 2B | int16 | 底盘前后移动量（mm，正=前，负=后） |
| gripper | 2B | uint16 | 夹爪伸出距离（mm） |
| tail | 2B | uint8×2 | `0xAA 0x66` |

`action=0` 为项目启动/空闲状态，`action=1` 为抓取阶段，`action=2` 为放置阶段；
`target=1~3` 表示物块槽位号
（抓取时是“这个颜色放到几号槽”，放置时是“从几号槽取物块”）。
上位机按 `TRACKING_SEND_INTERVAL` 连续发送当前计算出的底盘/夹爪运动量，
指令先经 `CHASSIS_RAMP_STEP_MM` 斜率限幅平滑（每个发送周期只变化一小步），
再按变化死区 / `CHASSIS_SEND_HEARTBEAT` 节流，避免下位机在增量移动执行过程中
被微小变化反复打断，同时底盘不会“动一下停一下”；`capture=1` 及阶段切换等事件包仍立即发送；
抓取阶段颜色稳定（连续 `color_stable_threshold` 帧同色）后即开始发送跟踪包移动底盘；
位置稳定（连续 `threshold` 帧圆心一致）且对准后才发送 `capture=1`。
即使偏移量很大，单次下发的底盘移动量也会被限制在 `MAX_CHASSIS_STEP_MM`
（默认 30mm）以内，并且先乘以 `CHASSIS_P_GAIN`（默认 0.9），让车逐次逼近、
越靠近单步越小，避免一次执行全量偏移导致过冲发散。
`capture=1` 表示执行当前动作（抓取或松爪放置）。若某帧像素坐标无法换算成有效运动量，
上位机会沿用上一帧有效指令，避免误发全 0 导致下位机误判为停止。

#### 底盘 → 上位机（GimbalToVision，15 字节）

| 字段 | 字节 | 类型 | 说明 |
|---|---|---|---|
| head | 2B | uint8×2 | `0x53 0x50` |
| chassis_x | 2B | uint16 | 底盘 X（mm） |
| chassis_y | 2B | uint16 | 底盘 Y（mm） |
| chassis_vx | 2B | int16 | 底盘速度 X 分量（mm/s） |
| chassis_vy | 2B | int16 | 底盘速度 Y 分量（mm/s） |
| capture_ack | 1B | uint8 | 1=已收到抓取请求（正在执行） |
| finish_capture | 1B | uint8 | 1=抓取完成，上位机切换下一目标 |
| arrived | 1B | uint8 | 1=已到达指定区域（抓取区/放置区） |
| tail | 2B | uint8×2 | `0xAA 0x66` |

底盘数据由 `SerialComm._chassis_recv_loop()` 后台线程持续接收，`unpack()` 时自动打时间戳；
回传坐标为 mm（位置）、mm/s（速度）；拦截规划在图像像素空间进行，按 `config.yaml`
中 `chassis.px_per_mm` 换算成 px / px/s（需按实际标定填写）；
上位机发送 `capture=1` 后进入等待状态；若未收到 `capture_ack=1`，按 `CAPTURE_RESEND_INTERVAL`
重发抓取请求，收到确认后停止重发。`finish_capture` 按 0→1 上升沿触发，且只在等待状态下消费，
同一完成信号只会切换一次目标；下位机完成一次抓取后应将 `finish_capture` 拉回 0，
以便下一目标产生新的上升沿。
`arrived` 按 0→1 上升沿触发：上位机发送区域移动指令后，下位机到达抓取区或放置区时回 1，
上位机才进入对应阶段的识别与执行。

### 运行方式

```bash
python3 src.py
```

| 操作 | 功能 |
|---|---|
| 下位机 `finish_capture=1` | 当前动作（抓取/放置）完成，自动进入下一步 |
| 下位机 `arrived=1` | 已到达抓取区/放置区，开始当前阶段识别 |
| `q` | 退出 |

### 模块调用关系

```
src.py
 ├→ common_camera.py         USB 摄像头（QR 阶段）
 ├→ preprocessing.py         Otsu 二值化（QR 阶段）
 ├→ scan_QRcode_andlist.py   QR 解码 → 颜色序列
 ├→ felling_color.py         视觉检测（每帧）
 ├→ kalman_tracker.py        KF 滤波（每帧 predict+update）
 ├→ intercept_planner.py     拦截规划（稳定后每帧 solve）
 ├→ transformer.py           相机坐标→夹爪/底盘指令
 ├→ gimbal.py                串口收发（发送线程 + 底盘接收线程）
 └→ common_camera.py         双 USB 摄像头（QR + 物块阶段）
```

---

## 二、手写数字识别 `felling_number.py`

USB 物块检测相机实时采集 + 轻量 CNN（MNIST）数字识别，模型定义在 `model.py`
（`TinyDigitCNN`，深度可分离卷积思路，参数量 < 25K，训练权重 `tiny_digit_cnn.pth`）。

预处理流水线：

```
灰度帧 → 高斯模糊 → OTSU 反二值化（白字黑底）→ 形态学闭运算
→ 最大轮廓定位 → 保持宽高比缩放至 20×20 → 28×28 画布居中
→ MNIST 标准化 (x/255 - 0.1307) / 0.3081 → 推理
```

### 代码流程图

```mermaid
flowchart TD
    A["加载模型 TinyCNN<br/>tiny_digit_cnn.pth"] --> B["打开物块检测 USB 相机<br/>open_camera"]
    B --> C["读取一帧 read_frame"]
    C --> D["转灰度 → 高斯模糊 → OTSU反二值<br/>闭运算 → 最大轮廓定位"]
    D --> E["20×20 等比缩放 → 28×28 居中<br/>MNIST标准化"]
    E --> F["TinyCNN 推理<br/>输出数字 + 置信度"]
    F --> G["显示: 灰度 / 识别结果 /<br/>28×28输入 / ROI"]
    G --> H{"按键?"}
    H -- "q" --> I["停止取流, 释放相机资源"]
    H -- "s" --> J["保存灰度截图"] --> C
    H -- "其他" --> C
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

调用物块检测 USB 摄像头实时取流，Trackbar 实时调节各颜色 HSV 上下界，
并排显示 原始+ROI / 掩码 / 掩码叠加。

```bash
python3 hsv_tuner.py
```

| 按键 | 功能 |
|---|---|
| `s` | 保存当前所有颜色阈值到 `config.yaml`（检测代码下次运行即生效） |
| `r` | 开关 ROI 框显示 |
| `q` | 退出 |

调参流程：`python3 hsv_tuner.py` → 拖动 Trackbar 调好某颜色阈值 → 按 `s` 写回
`config.yaml` → 重新运行主程序即可生效，无需再改 Python 代码。

### calibrate_camera.py —— 相机内参标定

用棋盘格标定 `fx/fy/cx/cy`，供 `transformer.py` 的像素→相机坐标转换使用。

```bash
python3 calibrate_camera.py --cols 9 --rows 6 --square-mm 25 --min-images 10 --update-transformer
```

| 按键 | 功能 |
|---|---|
| `s` / 空格 | 保存当前检测到棋盘格的画面 |
| `d` | 删除最近保存的一张 |
| `c` | 开始标定 |
| `q` | 退出 |

标定结果保存在 `camera_calibration.json`；加 `--update-transformer` 会把 `fx/fy/cx/cy`
自动写回 `transformer.py`。

> `transformer.py` 内参基准为 `640x480`（检测相机实际出图尺寸，已与 `camera_calibration.json` 一致）；
> 运行时如果实际画面分辨率不同，
> 会自动按宽高比例缩放内参（近似）。要求更高精度时，请用实际出图分辨率重新标定。
> 如果相机是向下俯拍安装的，还需要把 `transformer.py` 里的 `CAMERA_PITCH_DEG`
> 设为实际俯仰角（正=向下；光轴垂直向下时填 90）。相机装在车中心前方，因此
> 图像主点以下的点（相机前方距离为负）只要仍在车中心前方就是有效的，
> 只有换算后已到车中心后方的点才判无效。当前相机为垂直向下安装，
> `CAMERA_PITCH_DEG` 已设为 `90.0`。

> 双 USB 摄像头的设备编号/路径统一在 `common_camera.py` 顶部配置：
> `QR_CAMERA_SOURCE = 0`（二维码相机）、`DETECTION_CAMERA_SOURCE = 1`（物块检测相机）。
> 物块检测相机默认请求 `640x480`（`DETECTION_FRAME_WIDTH/HEIGHT`），若相机支持更高分辨率，
> 改这里并用同一分辨率重新标定。
> 如果系统里摄像头编号不是 0/1，用 `ls /dev/video*` 或
> `v4l2-ctl --list-devices` 查看后修改，也可直接填 `/dev/videoX` 路径。

### calibrate_px_per_mm.py —— px_per_mm 标定

`chassis.px_per_mm` 是下位机回传的 mm / mm/s 换算到图像 px / px/s 的倍率，
用标尺法测量：

```bash
python3 calibrate_px_per_mm.py --distance-mm 100
```

把尺子（或已知直径的物块）放在物块实际工作距离处，鼠标点两个端点，
脚本自动按 `像素距离 / 实际距离(mm)` 计算并写回 `config.yaml`。

| 按键 | 功能 |
|---|---|
| 鼠标左键 | 依次点两个端点，第二点落下后自动记为一组 |
| `R` | 清除当前第 1 点 |
| `U` | 撤销最后一组 |
| `S` | 保存平均值到 `config.yaml` 并退出 |
| `Q` | 退出（不保存） |

> 提示：这个单一倍率是近似值。左右移动对应“画面水平方向”的比例，
> 前后移动对应“画面竖直方向”的比例（透视下两者不完全相同）。
> 若拦截主要靠左右移动，让两点尽量水平；若前后移动更重要，把尺子沿前后方向放、
> 让两点尽量垂直，再按实际效果微调。

### validate_transform.py —— 坐标换算与指令验证

在真机上点击画面中的物块中心，脚本会打印相机坐标、车中心坐标、
底盘移动量和夹爪伸长量；再输入实际测量的距离，就能判断误差是否在 5cm 内。

```bash
python3 validate_transform.py
```

| 按键 | 功能 |
|---|---|
| 鼠标左键 | 点击物块/圆环中心，打印坐标与底盘/夹爪指令 |
| `M` | 输入实测“车中心到物块”距离(cm)，对比计算误差 |
| `P` | 输入“相机正下方到物块”的水平前方距离(cm)，自动反推 `CAMERA_PITCH_DEG` |
| `Q` | 退出 |

如果已经量好距离，也可以直接传参：

```bash
python3 validate_transform.py --actual-distance-cm 42
```

相机有俯仰角时，先用 `--pitch-deg` 试出误差最小的角度：

```bash
python3 validate_transform.py --pitch-deg 40
```

也可以按 `P`，输入一次实测的“相机正下方地面点到物块的水平前方距离(cm)”，
脚本会直接算出建议的 `CAMERA_PITCH_DEG`，再把这个值写回 `transformer.py`。

### create_qr.py —— 生成测试二维码

用 `qrcode` 库生成形如 `156+123+516+231` 的二维码并保存为 `my_qrcode.png`，
用于 USB 摄像头 QR 扫描阶段的离屏测试。

```bash
python3 create_qr.py
```

---

## 四、旧版整机程序 `example_code.py`（遗留）

旧架构的单文件整机程序（约 1900 行，仅作历史参考），功能包括：

- QR 检测（`run_erweima`，`/dev/video_xia0`）；
- 物块圆心定标（`run_wukuaiyuanxin_1/2/A/xuanzequyu`）；
- 色环 / 码垛两次定标（`run_sehuanyuanxin2(_centered)`、`run_maduoyuanxin2(_centered)`）；
- 边界线检测（`detect_boundary_line`）；

旧文本指令入口已移除；目标切换统一以 `src.py` 的 `finish_capture` 为准。
新功能开发请以 `src.py` 模块化架构为准。

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
pyyaml          # 读取/保存 config.yaml
pyserial
pyzbar          # QR 解码
qrcode          # 生成测试二维码
torch           # 数字识别
```

两台均为普通 USB 免驱摄像头，由 OpenCV 直接读取，不再依赖海康 SDK；
