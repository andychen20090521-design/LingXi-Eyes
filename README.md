# Lingxi Eye ESP32 Camera + TTS

基于 Seeed Studio XIAO ESP32S3 Sense 的摄像头、麦克风录音和语音播报测试项目。ESP32 负责提供网页调试界面、拍照接口、PDM 麦克风录音接口，并通过 MAX98357A I2S 功放播放电脑端 TTS 服务生成的 PCM 音频。

## 功能

- XIAO ESP32S3 Sense 摄像头拍照，输出 JPEG 图片
- 通过网页查看摄像头画面并手动刷新
- 通过 MAX98357A I2S 功放播放 TTS 语音
- 通过板载 PDM 麦克风录制 WAV 音频
- ESP32 内置 Web Server，提供调试页面和 HTTP API
- Python FastAPI 后端将文字转为 24 kHz / 16-bit / mono PCM，供 ESP32 播放

## 硬件

- Seeed Studio XIAO ESP32S3 Sense
- MAX98357A I2S 数字功放模块
- 扬声器
- 电脑一台，用于运行 Python TTS 后端

### MAX98357A 接线

| MAX98357A | XIAO ESP32S3 |
| --- | --- |
| BCLK | D1 / GPIO2 |
| LRC | D2 / GPIO3 |
| DIN | D3 / GPIO4 |
| SD | D0 / GPIO1，或直接接 3.3V |
| VIN | 3.3V 或 5V |
| GND | GND |

### 麦克风引脚

项目使用 XIAO ESP32S3 Sense 的 PDM 麦克风：

| 信号 | GPIO |
| --- | --- |
| PDM DATA | GPIO41 |
| PDM CLK | GPIO42 |

## 软件环境

### ESP32 固件

建议使用 PlatformIO。

`platformio.ini` 当前配置：

```ini
[env:seeed_xiao_esp32s3]
platform = espressif32
board = seeed_xiao_esp32s3
framework = arduino
monitor_speed = 115200
board_build.partitions = huge_app.csv
board_build.arduino.memory_type = qio_opi

build_flags =
  -DARDUINO_USB_MODE=1
  -DARDUINO_USB_CDC_ON_BOOT=1
  -DBOARD_HAS_PSRAM
```

### Python TTS 后端

电脑端需要安装 Python 依赖：

```bash
pip install fastapi uvicorn pyttsx3
```

在项目根目录启动 TTS 服务：

```bash
uvicorn server:app --host 0.0.0.0 --port 8001
```

启动后可以访问：

- `http://电脑IP:8001/`：查看服务状态
- `http://电脑IP:8001/voices`：查看系统可用语音
- `http://电脑IP:8001/preview.wav?text=你好`：预览 WAV
- `http://电脑IP:8001/tts.pcm?text=你好`：输出 ESP32 播放用 PCM

## 配置

烧录前请修改 `src/main.cpp` 中的 WiFi 和 TTS 后端地址：

```cpp
const char* wifi_ssid = "你的WiFi名称";
const char* wifi_password = "你的WiFi密码";

const char* TTS_HOST = "电脑的局域网IP";
const int TTS_PORT = 8001;
```

ESP32 和运行 `server.py` 的电脑必须在同一个局域网内。电脑防火墙需要允许 8001 端口被局域网访问。

> 注意：当前代码里 WiFi 名称和密码是明文写在源码中的。上传到 GitHub 前，请改成占位符，或改为从单独的本地配置文件读取，并把本地配置文件加入 `.gitignore`。

## 使用方法

1. 启动电脑端 TTS 服务：

   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8001
   ```

2. 在 `src/main.cpp` 中配置 WiFi 和电脑 IP。

3. 使用 PlatformIO 编译并烧录：

   ```bash
   pio run -t upload
   ```

4. 打开串口监视器：

   ```bash
   pio device monitor
   ```

5. 串口会输出 ESP32 的局域网 IP，在浏览器中打开：

   ```text
   http://ESP32_IP/
   ```

## ESP32 Web 接口

| 接口 | 方法 | 说明 |
| --- | --- | --- |
| `/` | GET | 网页调试首页 |
| `/capture.jpg` | GET | 拍摄并返回 JPEG 图片 |
| `/speak?text=你好` | GET | 播放指定文字的 TTS 语音 |
| `/test_tts` | GET | 播放默认测试语音 |
| `/record.wav?seconds=3` | GET | 录制 WAV 音频，支持 1-5 秒 |
| `/status` | GET | 返回 WiFi、TTS、摄像头和音频状态 |

## 串口命令

在串口监视器中可以发送单字符命令：

| 命令 | 说明 |
| --- | --- |
| `r` 或 `R` | 录制 1 秒麦克风音频，用于测试麦克风 |
| `w` 或 `W` | 触发 WiFi 重新连接 |

## 音频参数

TTS 播放参数：

- PCM 采样率：24 kHz
- 位深：16-bit
- 声道：mono
- ESP32 I2S：`I2S_NUM_1`

麦克风录音参数：

- 采样率：16 kHz
- 位深：16-bit
- 声道：mono
- ESP32 I2S：`I2S_NUM_0`

## 常见问题

### 网页能打开，但 TTS 不出声

- 确认 `server.py` 已启动
- 确认 `TTS_HOST` 是电脑的局域网 IP，不是 `127.0.0.1`
- 确认电脑防火墙允许 8001 端口访问
- 在浏览器打开 `http://电脑IP:8001/tts.pcm?text=你好` 测试后端是否有响应
- 检查 MAX98357A 的 BCLK、LRC、DIN、SD、VIN、GND 接线

### 摄像头初始化失败

- 确认使用的是 XIAO ESP32S3 Sense
- 确认摄像头排线连接正常
- 确认 PlatformIO 配置中启用了 PSRAM：`-DBOARD_HAS_PSRAM`

### 录音失败

- 确认当前没有正在播放 TTS
- `/record.wav?seconds=3` 的录音时长被限制在 1-5 秒
- 录音和播放不能同时进行，程序中使用 `isRecording` 和 `isSpeaking` 做了互斥保护

## 项目结构

```text
.
├── platformio.ini
├── server.py
└── src
    └── main.cpp
```

## License

如需开源发布，建议在仓库中补充明确的开源协议文件，例如 MIT License。
