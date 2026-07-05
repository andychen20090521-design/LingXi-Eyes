# Lingxi AI Backend

基于 FastAPI 的灵犀 AI 导盲后端服务，用于连接 ESP32 摄像头、YOLO 目标检测、豆包视觉大模型和语音播报能力，为盲人或弱视用户提供前方环境识别、红绿灯判断、障碍物提示、语音问答和自动巡检功能。

## 功能特性

- ESP32 摄像头实时取图与网页预览
- YOLOv8 快速目标检测，支持障碍物、车辆、行人、红绿灯等识别
- 豆包视觉大模型环境描述和图片问答
- 面向导盲场景的安全提示生成，默认采用保守策略
- ESP32 语音播报接口联动
- ESP32 麦克风录音、Whisper ASR 语音指令识别
- Web 控制台、健康检查页和最近操作日志

## 项目结构

```text
.
|-- main.py              # FastAPI 主服务
|-- requirements.txt     # Python 基础依赖
|-- yolov8n.pt           # YOLO 快速模型
|-- yolov8s.pt           # YOLO 较高精度模型
`-- .env                 # 本地环境变量，不建议上传 GitHub
```

## 环境要求

- Python 3.10+
- 可访问 ESP32 摄像头模块的局域网环境
- 火山方舟/豆包 API Key
- 可选：支持浏览器语音识别的 Chrome 浏览器

ESP32 端需要提供以下 HTTP 接口：

| 接口 | 用途 |
| --- | --- |
| `GET /capture.jpg` | 返回当前摄像头 JPEG 图片 |
| `GET /speak?text=...` | 播报指定文本 |
| `GET /record.wav?seconds=3` | 录制并返回 WAV 音频 |

## 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果需要使用 YOLO 检测和 ESP32 麦克风语音指令，还需要安装：

```bash
pip install ultralytics opencv-python numpy faster-whisper
```

> 如果使用 CPU 运行 ASR，默认配置 `ASR_DEVICE=cpu` 和 `ASR_COMPUTE_TYPE=int8` 更省资源。

## 配置环境变量

在项目根目录创建 `.env` 文件：

```env
AI_API_KEY=your_api_key_here
AI_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
AI_MODEL=doubao-seed-2-0-mini-260428
ESP32_BASE_URL=http://192.168.43.188

YOLO_FAST_MODEL_PATH=yolov8n.pt
YOLO_ACCURATE_MODEL_PATH=yolov8s.pt
YOLO_FAST_IMGSZ=320
YOLO_DISPLAY_IMGSZ=512
YOLO_TRAFFIC_IMGSZ=512
YOLO_CONF=0.20

ASR_MODEL=base
ASR_DEVICE=cpu
ASR_COMPUTE_TYPE=int8
```

不要把真实 `.env` 或 API Key 上传到 GitHub。建议在仓库中只保留 `.env.example` 这类示例文件。

## 启动服务

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问：

- 首页状态：`http://localhost:8000/`
- 控制台：`http://localhost:8000/control`
- 实时画面：`http://localhost:8000/live`
- 健康检查页：`http://localhost:8000/health_page`

## 主要接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/` | 服务状态与接口列表 |
| `GET` | `/health` | 检查 ESP32、YOLO、豆包、ASR 状态 |
| `GET` | `/logs` | 查看最近操作日志 |
| `GET` | `/proxy_capture.jpg` | 代理获取 ESP32 摄像头图片 |
| `POST` | `/analyze_upload` | 上传图片并调用豆包视觉分析 |
| `POST` | `/analyze_esp32` | 从 ESP32 拍照并分析 |
| `POST` | `/analyze_and_speak` | 从 ESP32 拍照、分析并播报 |
| `POST` | `/ask_esp32` | 基于当前画面进行文字问答，可选择播报 |
| `POST` | `/yolo_detect_esp32` | YOLO 展示模式识别，生成标注图 |
| `POST` | `/guide_once` | YOLO 快速导盲判断并播报 |
| `POST` | `/traffic_light_once` | 红绿灯专项判断并播报 |
| `POST` | `/yolo_speak_once` | 兼容旧接口，等价于快速导盲播报 |
| `POST` | `/asr_upload` | 上传 WAV 音频并转写中文文本 |
| `POST` | `/esp32_mic_command` | 使用 ESP32 麦克风录音并执行语音指令 |
| `GET` | `/yolo_annotated.jpg` | 获取最近一次 YOLO 标注图 |

## 调用示例

健康检查：

```bash
curl "http://localhost:8000/health?esp32_url=http://192.168.43.188"
```

执行一次快速导盲判断并播报：

```bash
curl -X POST "http://localhost:8000/guide_once?esp32_url=http://192.168.43.188&force=true&show_image=false"
```

进行红绿灯专项判断：

```bash
curl -X POST "http://localhost:8000/traffic_light_once?esp32_url=http://192.168.43.188&force=true"
```

对当前画面提问：

```bash
curl -X POST "http://localhost:8000/ask_esp32?esp32_url=http://192.168.43.188&question=前方有什么障碍物&speak=true"
```

## Web 控制台

访问 `/control` 可以在浏览器中完成：

- 开启或停止实时画面
- 执行 YOLO 单次识别
- 执行快速导盲播报
- 执行红绿灯专项判断
- 向当前画面提问并播报
- 启动自动巡检
- 使用浏览器语音指令或 ESP32 麦克风指令

## 导盲策略说明

系统提示词和后处理逻辑会尽量避免把判断责任交给用户。当识别结果不确定、画面不清晰、出现近处障碍物或红灯时，系统会倾向于提示停止、等待、绕行或缓慢前进。

该项目适合作为导盲辅助原型，不应作为唯一安全依据。真实户外使用前，请进行充分测试，并结合硬件稳定性、网络延迟、摄像头视角和语音播报可靠性综合评估。

## 常见问题

### YOLO 不可用

确认已经安装 `ultralytics`、`opencv-python`、`numpy`，并且 `yolov8n.pt`、`yolov8s.pt` 文件路径正确。

### 豆包接口不可用

确认 `.env` 中 `AI_API_KEY`、`AI_BASE_URL` 和 `AI_MODEL` 配置正确，并且当前网络可以访问火山方舟 API。

### ESP32 连接失败

确认电脑和 ESP32 在同一网络下，`ESP32_BASE_URL` 地址正确，并能在浏览器中直接访问 `http://ESP32地址/capture.jpg`。

### ASR 不可用

确认安装了 `faster-whisper`，首次加载模型需要下载模型文件。如果设备性能有限，可以将 `ASR_MODEL` 改为 `tiny`。

## GitHub 上传建议

建议上传：

- `main.py`
- `requirements.txt`
- `README.md`
- YOLO 模型文件，或在 README 中说明下载方式

建议不要上传：

- `.env`
- `__pycache__/`
- 任何真实 API Key、局域网敏感配置或临时文件
