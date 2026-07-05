import os
import re
import time
import base64
import requests
import threading
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import HTMLResponse, Response

try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
except Exception:
    cv2 = None
    np = None
    YOLO = None


# ==================== 读取配置 ====================

load_dotenv()

AI_API_KEY = os.getenv("AI_API_KEY")
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
AI_MODEL = os.getenv("AI_MODEL", "doubao-seed-2-0-mini-260428")
ESP32_BASE_URL = os.getenv("ESP32_BASE_URL", "http://192.168.43.188")

YOLO_FAST_MODEL_PATH = os.getenv("YOLO_FAST_MODEL_PATH", "yolov8n.pt")
YOLO_ACCURATE_MODEL_PATH = os.getenv("YOLO_ACCURATE_MODEL_PATH", "yolov8s.pt")
YOLO_MODEL_PATH = YOLO_FAST_MODEL_PATH

YOLO_FAST_IMGSZ = int(os.getenv("YOLO_FAST_IMGSZ", "320"))
YOLO_DISPLAY_IMGSZ = int(os.getenv("YOLO_DISPLAY_IMGSZ", "512"))
YOLO_TRAFFIC_IMGSZ = int(os.getenv("YOLO_TRAFFIC_IMGSZ", "512"))
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.20"))

ASR_MODEL = os.getenv("ASR_MODEL", "base")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cpu")
ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "int8")

app = FastAPI()


# ==================== 全局状态 ====================

_yolo_models = {}
_yolo_lock = threading.Lock()

_asr_model = None
_asr_model_name = None
_asr_lock = threading.Lock()

last_speak_text = ""
last_speak_time = 0

latest_annotated_jpg = None
latest_annotated_time = 0
latest_yolo_result = None
cache_lock = threading.Lock()

MAX_OPERATION_LOGS = 20
operation_logs = []
operation_log_lock = threading.Lock()


def add_operation_log(kind: str, ok: bool = True, detail: str = "", extra=None):
    item = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kind": kind,
        "ok": ok,
        "detail": detail,
    }

    if extra is not None:
        item["extra"] = extra

    with operation_log_lock:
        operation_logs.insert(0, item)
        del operation_logs[MAX_OPERATION_LOGS:]

    return item


# ==================== 豆包提示词 ====================

VISION_PROMPT = """
你是一个真正服务盲人用户的智能导盲眼镜视觉助手。

你的任务不是让用户自己观察，而是直接给出可执行的行动提示。

请根据图片识别前方环境，重点判断：
人、椅子、桌子、书包、门、墙、台阶、车辆、斑马线、红绿灯、地面障碍物。

非常重要：
1. 禁止说“请注意观察”。
2. 禁止说“自行判断”。
3. 禁止说“看清后再走”。
4. 禁止把判断责任交给盲人用户。
5. 不确定是否安全时，必须保守处理，提示用户停止等待或缓慢前进。
6. 看到红灯，必须提示停止等待。
7. 看到绿灯，也不能直接说安全通过，要结合车辆和障碍物判断。
8. 无法判断红绿灯颜色时，必须提示停止等待。
9. 看到近处障碍物，必须提示停止、绕行或避让。
10. 回答要短，适合语音播报。

必须严格按照这个格式输出：

场景描述：我看到……
行动建议：请……
"""


def build_question_prompt(question: str) -> str:
    return f"""
你是一个真正服务盲人用户的智能导盲眼镜视觉助手。

用户的问题是：{question}

请根据图片回答用户的问题，并给出明确行动建议。

要求：
1. 直接回答用户问题。
2. 不要说“请注意观察”。
3. 不要说“自行判断”。
4. 不要说“看清后再走”。
5. 不要把判断责任交给盲人用户。
6. 不确定是否安全时，默认让用户停止等待或缓慢前进。
7. 红灯停止等待，黄灯停止等待，绿灯也要确认没有车辆和近处障碍后才缓慢通行。
8. 看到近处障碍时，告诉用户停止、绕行或避让。
9. 回答不要太长，最多两句话，适合直接语音播报。

输出格式：
回答：……
行动建议：请……
"""


# ==================== YOLO 标签 ====================

LABEL_CN = {
    "person": "人",
    "bicycle": "自行车",
    "car": "汽车",
    "motorcycle": "摩托车",
    "bus": "公交车",
    "truck": "卡车",
    "traffic light": "红绿灯",
    "stop sign": "停止标志",
    "bench": "长椅",
    "chair": "椅子",
    "couch": "沙发",
    "bed": "床",
    "dining table": "桌子",
    "laptop": "电脑",
    "cell phone": "手机",
    "book": "书",
    "backpack": "书包",
    "handbag": "手提包",
    "suitcase": "行李箱",
    "bottle": "瓶子",
    "cup": "杯子",
    "dog": "狗",
}

HAZARD_LABELS = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "chair",
    "bench",
    "couch",
    "dining table",
    "backpack",
    "handbag",
    "suitcase",
    "bottle",
    "cup",
    "dog",
    "traffic light",
}

VEHICLE_LABELS = {
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
}

OBSTACLE_LABELS = {
    "chair",
    "bench",
    "couch",
    "dining table",
    "backpack",
    "handbag",
    "suitcase",
    "bottle",
    "cup",
    "dog",
    "person",
}

# COCO 类别 ID，只保留导盲相关目标
GUIDE_CLASS_IDS = [
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    9,   # traffic light
    13,  # bench
    24,  # backpack
    26,  # handbag
    28,  # suitcase
    39,  # bottle
    41,  # cup
    56,  # chair
    57,  # couch
    60,  # dining table
]

TRAFFIC_CLASS_IDS = [
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    9,   # traffic light
]

TRAFFIC_LIGHT_CN = {
    "red": "红灯",
    "yellow": "黄灯",
    "green": "绿灯",
    "unknown": "颜色不确定的红绿灯",
}


# ==================== 基础工具 ====================

def check_config():
    if not AI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="缺少 AI_API_KEY，请检查 .env 文件"
        )


def sanitize_guidance_language(text: str) -> str:
    """
    把不适合盲人用户的表达改掉。
    """

    if not text:
        return ""

    replacements = {
        "请注意观察后前进": "请缓慢前进，并等待下一次系统提示",
        "注意观察后前进": "请缓慢前进，并等待下一次系统提示",
        "请注意观察再前进": "请缓慢前进，并等待下一次系统提示",
        "注意观察再前进": "请缓慢前进，并等待下一次系统提示",
        "请注意观察": "请小心慢行",
        "注意观察": "请小心慢行",
        "自行判断": "等待系统进一步提示",
        "自己判断": "等待系统进一步提示",
        "看清后": "等待系统进一步提示后",
        "观察后": "等待系统进一步提示后",
        "可以通过，但请小心观察": "可以缓慢通过，请注意车辆声音",
        "可以通行，但请注意观察": "可以缓慢通行，请注意车辆声音",
        "安全通过": "缓慢通过",
        "放心通过": "缓慢通过",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace("不确定时请观察", "不确定时请停止等待")
    text = text.replace("无法判断时请观察", "无法判断时请停止等待")

    text = re.sub(r"\s+", " ", text).strip()

    return text


def image_bytes_to_data_url(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def fetch_esp32_image(esp32_url: str) -> bytes:
    esp32_url = esp32_url.rstrip("/")
    capture_url = f"{esp32_url}/capture.jpg"

    try:
        response = requests.get(capture_url, timeout=15)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"无法连接 ESP32 摄像头：{e}"
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"ESP32 拍照失败：{response.status_code}，{response.text}"
        )

    if not response.content:
        raise HTTPException(
            status_code=500,
            detail="ESP32 返回图片为空"
        )

    return response.content


def send_to_esp32_speak(esp32_url: str, text: str) -> dict:
    esp32_url = esp32_url.rstrip("/")
    text = sanitize_guidance_language(text)

    try:
        response = requests.get(
            f"{esp32_url}/speak",
            params={"text": text},
            timeout=90
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"调用 ESP32 播报失败：{e}"
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"ESP32 播报接口错误：{response.status_code}，{response.text}"
        )

    return {
        "status_code": response.status_code,
        "text": response.text
    }


# ==================== 豆包识别 ====================

def extract_text_from_response(data: dict) -> str:
    """
    只提取最终回答，不提取 reasoning。
    兼容 output_text / output message / choices 三种结构。
    """

    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = data.get("output", [])
    final_texts = []

    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue

            item_type = item.get("type", "")

            if item_type == "reasoning":
                continue

            if item_type in ["message", "output_message"]:
                content = item.get("content", [])

                if isinstance(content, list):
                    for c in content:
                        if not isinstance(c, dict):
                            continue

                        c_type = c.get("type", "")

                        if c_type in ["output_text", "text"]:
                            text = c.get("text", "")
                            if isinstance(text, str) and text.strip():
                                final_texts.append(text.strip())

                elif isinstance(content, str) and content.strip():
                    final_texts.append(content.strip())

            if item_type == "output_text":
                text = item.get("text", "")
                if isinstance(text, str) and text.strip():
                    final_texts.append(text.strip())

    if final_texts:
        return "\n".join(final_texts).strip()

    choices = data.get("choices", [])
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, str) and content.strip():
            return content.strip()

    return ""


def call_doubao_with_prompt(image_data_url: str, prompt: str, max_tokens: int = 600) -> str:
    """
    调用豆包视觉模型。

    这版做了稳定性增强：
    1. 第一次如果没有返回正式 output_text，会自动二次请求。
    2. 二次请求仍失败时，不让页面报错，而是返回保守导盲提示。
    """

    check_config()

    fallback_text = (
        "\u56de\u7b54\uff1a\u6682\u65f6\u6ca1\u6709\u83b7\u5f97\u5b8c\u6574\u8bc6\u522b\u7ed3\u679c\u3002"
        "\u884c\u52a8\u5efa\u8bae\uff1a\u8bf7\u5148\u505c\u6b62\u7b49\u5f85\uff0c\u7a0d\u540e\u91cd\u65b0\u8bc6\u522b\u3002"
    )

    url = AI_BASE_URL.rstrip("/") + "/responses"

    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": AI_MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": image_data_url
                    },
                    {
                        "type": "input_text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_output_tokens": max_tokens
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=90
        )
    except Exception as e:
        print(f"Doubao API request failed, using fallback guidance: {e}")
        return sanitize_guidance_language(fallback_text)

    if response.status_code != 200:
        print(
            "Doubao API returned non-200, using fallback guidance: "
            f"{response.status_code}, {response.text}"
        )
        return sanitize_guidance_language(fallback_text)

    data = response.json()
    description = extract_text_from_response(data)

    # 第一次没有正式结果，就再试一次，用更直接的短提示
    if not description:
        simple_payload = {
            "model": AI_MODEL,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": image_data_url
                        },
                        {
                            "type": "input_text",
                            "text": "请直接用中文回答图片中前方有什么，并给出一句给盲人用户的行动建议。不要输出推理过程。不要说请注意观察。最多两句话。"
                        }
                    ]
                }
            ],
            "max_output_tokens": 600
        }

        try:
            retry_response = requests.post(
                url,
                headers=headers,
                json=simple_payload,
                timeout=90
            )
        except Exception as e:
            print(f"Doubao retry failed, using fallback guidance: {e}")
            return sanitize_guidance_language(fallback_text)

        if retry_response.status_code != 200:
            print(
                "Doubao retry returned non-200, using fallback guidance: "
                f"{retry_response.status_code}, {retry_response.text}"
            )
            return sanitize_guidance_language(fallback_text)

        retry_data = retry_response.json()
        description = extract_text_from_response(retry_data)

    # 如果两次都没有结果，不能让系统崩掉，给保守提示
    if not description:
        description = fallback_text

    return sanitize_guidance_language(description.strip())


def call_doubao_vision(image_data_url: str) -> str:
    return call_doubao_with_prompt(image_data_url, VISION_PROMPT, max_tokens=600)


def call_doubao_question(image_data_url: str, question: str) -> str:
    prompt = build_question_prompt(question)

    # 提问模式不要给太少 tokens，否则模型可能只返回 reasoning，没有正式回答
    return call_doubao_with_prompt(image_data_url, prompt, max_tokens=600)


def remove_useless_words(text: str) -> str:
    if not text:
        return ""

    useless_phrases = [
        "画面较模糊，",
        "画面有些模糊，",
        "画面比较模糊，",
        "无法清晰辨认，",
        "无法准确判断，",
        "看起来不对，",
        "不对，",
        "疑似",
        "可能是",
        "可能有",
        "请重新拍摄，",
        "建议重新拍摄，",
        "请调整摄像头，",
        "需要调整摄像头，"
    ]

    for p in useless_phrases:
        text = text.replace(p, "")

    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()

    return sanitize_guidance_language(text)


def extract_part(description: str, start_keys: list, end_keys: list = None) -> str:
    if not description:
        return ""

    text = description.strip()

    for key in start_keys:
        if key in text:
            part = text.split(key, 1)[1]

            if end_keys:
                for end_key in end_keys:
                    if end_key in part:
                        part = part.split(end_key, 1)[0]

            return remove_useless_words(part).strip()

    return ""


def clean_speak_text(description: str) -> str:
    if not description:
        return "没有获取到识别结果，请停止等待。"

    description = remove_useless_words(description)

    scene = extract_part(
        description,
        start_keys=["场景描述：", "场景描述:", "回答：", "回答:"],
        end_keys=["行动建议：", "行动建议:", "建议：", "建议:"]
    )

    advice = extract_part(
        description,
        start_keys=["行动建议：", "行动建议:", "建议：", "建议:"],
        end_keys=None
    )

    if not scene and not advice:
        text = sanitize_guidance_language(description.strip())
        if len(text) > 110:
            text = text[:110] + "。"
        return text

    if scene:
        scene = scene.strip("。；;，, ")
        if len(scene) > 58:
            scene = scene[:58] + "。"

    if advice:
        advice = advice.strip("。；;，, ")
        if len(advice) > 48:
            advice = advice[:48] + "。"

    speak_parts = []

    if scene:
        if scene.startswith("我看到"):
            speak_parts.append(scene + "。")
        else:
            speak_parts.append("我看到" + scene + "。")

    if advice:
        if advice.startswith("请") or advice.startswith("停止") or advice.startswith("小心") or advice.startswith("缓慢"):
            speak_parts.append(advice + "。")
        else:
            speak_parts.append("请" + advice + "。")

    speak_text = "".join(speak_parts)

    if len(speak_text) > 125:
        speak_text = speak_text[:125] + "。"

    return sanitize_guidance_language(speak_text)


# ==================== 图像处理 ====================

def image_bytes_to_cv2(image_bytes: bytes):
    if cv2 is None or np is None:
        raise HTTPException(
            status_code=500,
            detail="opencv-python 或 numpy 没有安装"
        )

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=500,
            detail="图片解码失败"
        )

    return image


def enhance_image_for_detection(image_bgr):
    if cv2 is None or np is None:
        return image_bgr

    img = image_bgr.copy()

    img = cv2.convertScaleAbs(img, alpha=1.08, beta=4)

    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=1.6,
            tileGridSize=(8, 8)
        )

        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        img = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    except Exception:
        pass

    try:
        blur = cv2.GaussianBlur(img, (0, 0), 1.0)
        img = cv2.addWeighted(img, 1.18, blur, -0.18, 0)
    except Exception:
        pass

    return img


def encode_jpeg(image_bgr, quality: int = 82) -> bytes:
    ok, buffer = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )

    if not ok:
        raise HTTPException(
            status_code=500,
            detail="JPEG 编码失败"
        )

    return buffer.tobytes()


# ==================== 红绿灯颜色判断 ====================

def clamp_box(box, width: int, height: int):
    x1, y1, x2, y2 = [int(v) for v in box]

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)

    return x1, y1, x2, y2


def detect_traffic_light_color(image_bgr, box):
    if cv2 is None or np is None:
        return {
            "state": "unknown",
            "state_cn": TRAFFIC_LIGHT_CN["unknown"],
            "red_score": 0,
            "yellow_score": 0,
            "green_score": 0,
            "reason": "opencv not available"
        }

    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, w, h)

    pad_x = max(2, int((x2 - x1) * 0.12))
    pad_y = max(2, int((y2 - y1) * 0.12))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)

    crop = image_bgr[y1:y2, x1:x2]

    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 5:
        return {
            "state": "unknown",
            "state_cn": TRAFFIC_LIGHT_CN["unknown"],
            "red_score": 0,
            "yellow_score": 0,
            "green_score": 0,
            "reason": "traffic light crop too small"
        }

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    red_mask1 = cv2.inRange(hsv, np.array([0, 70, 80]), np.array([12, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([168, 70, 80]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    yellow_mask = cv2.inRange(hsv, np.array([15, 60, 90]), np.array([38, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([40, 50, 70]), np.array([95, 255, 255]))

    v = hsv[:, :, 2]
    bright_mask = cv2.inRange(v, 90, 255)

    red_mask = cv2.bitwise_and(red_mask, bright_mask)
    yellow_mask = cv2.bitwise_and(yellow_mask, bright_mask)
    green_mask = cv2.bitwise_and(green_mask, bright_mask)

    red_score = int(cv2.countNonZero(red_mask))
    yellow_score = int(cv2.countNonZero(yellow_mask))
    green_score = int(cv2.countNonZero(green_mask))

    area = crop.shape[0] * crop.shape[1]
    min_pixels = max(4, int(area * 0.004))

    scores = {
        "red": red_score,
        "yellow": yellow_score,
        "green": green_score,
    }

    best_state = max(scores, key=scores.get)
    best_score = scores[best_state]
    second_score = sorted(scores.values(), reverse=True)[1]

    if best_score < min_pixels:
        state = "unknown"
        reason = "color pixels too few"
    elif second_score > 0 and best_score < second_score * 1.25:
        state = "unknown"
        reason = "color scores too close"
    else:
        state = best_state
        reason = "color detected"

    return {
        "state": state,
        "state_cn": TRAFFIC_LIGHT_CN[state],
        "red_score": red_score,
        "yellow_score": yellow_score,
        "green_score": green_score,
        "reason": reason
    }


# ==================== YOLO ====================

def load_cached_yolo_model(model_path: str):
    if model_path in _yolo_models:
        return _yolo_models[model_path]

    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    _yolo_models[model_path] = model
    print(f"YOLO model loaded: {model_path}")
    return model


def get_yolo_model(model_kind: str):
    if YOLO is None or cv2 is None or np is None:
        raise HTTPException(
            status_code=500,
            detail="YOLO 依赖没有安装，请运行：pip install ultralytics opencv-python numpy"
        )

    if model_kind == "accurate":
        if os.path.exists(YOLO_ACCURATE_MODEL_PATH):
            try:
                with _yolo_lock:
                    model = load_cached_yolo_model(YOLO_ACCURATE_MODEL_PATH)
                return model, YOLO_ACCURATE_MODEL_PATH
            except Exception as e:
                print(
                    "Failed to load accurate YOLO model, falling back to fast model: "
                    f"{YOLO_ACCURATE_MODEL_PATH}, error: {e}"
                )
        else:
            print(
                "Accurate YOLO model not found, falling back to fast model: "
                f"{YOLO_ACCURATE_MODEL_PATH}"
            )

    with _yolo_lock:
        model = load_cached_yolo_model(YOLO_FAST_MODEL_PATH)
    return model, YOLO_FAST_MODEL_PATH


def position_from_box(box, width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    area = (x2 - x1) * (y2 - y1)
    frame_area = width * height

    if cx < width * 0.33:
        side = "左侧"
    elif cx > width * 0.66:
        side = "右侧"
    else:
        side = "前方"

    if area > frame_area * 0.18:
        distance = "近处"
    elif area < frame_area * 0.04:
        distance = "远处"
    else:
        distance = ""

    if distance:
        return side + distance

    return side


def yolo_detect_image(
    image_for_yolo,
    image_for_color=None,
    imgsz: int = YOLO_DISPLAY_IMGSZ,
    classes=None,
    model_kind: str = "fast"
):
    model, actual_model_path = get_yolo_model(model_kind)

    if image_for_color is None:
        image_for_color = image_for_yolo

    height, width = image_for_yolo.shape[:2]

    results = model.predict(
        image_for_yolo,
        imgsz=imgsz,
        conf=YOLO_CONF,
        classes=classes if classes is not None else GUIDE_CLASS_IDS,
        verbose=False
    )

    result = results[0]
    detections = []

    if result.boxes is None:
        return detections, actual_model_path

    names = model.names

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()

        label_en = names.get(cls_id, str(cls_id))
        label_cn = LABEL_CN.get(label_en, label_en)

        x1, y1, x2, y2 = [float(v) for v in xyxy]
        box_list = [round(x1), round(y1), round(x2), round(y2)]

        detection = {
            "label_en": label_en,
            "label_cn": label_cn,
            "confidence": round(conf, 3),
            "box": box_list,
            "position": position_from_box([x1, y1, x2, y2], width, height),
            "is_hazard": label_en in HAZARD_LABELS
        }

        if label_en == "traffic light":
            color_info = detect_traffic_light_color(image_for_color, box_list)
            detection["traffic_light_state"] = color_info["state"]
            detection["traffic_light_state_cn"] = color_info["state_cn"]
            detection["traffic_light_color_debug"] = color_info

        detections.append(detection)

    return detections, actual_model_path


def choose_bypass_direction(front_hazards: list) -> str:
    has_left = any("左侧" in d.get("position", "") for d in front_hazards)
    has_right = any("右侧" in d.get("position", "") for d in front_hazards)
    has_front = any("前方" in d.get("position", "") for d in front_hazards)

    if has_left and not has_right:
        return "请向右侧小幅绕行。"
    if has_right and not has_left:
        return "请向左侧小幅绕行。"
    if has_front and not has_left and not has_right:
        return "请先停下，然后小幅绕开前方障碍。"

    return "前方通行空间不明确，请先停止等待。"


def summarize_items(detections: list, limit: int = 3) -> str:
    if not detections:
        return ""

    sorted_items = sorted(
        detections,
        key=lambda d: d.get("confidence", 0),
        reverse=True
    )[:limit]

    parts = []

    for d in sorted_items:
        if d.get("label_en") == "traffic light":
            state_cn = d.get("traffic_light_state_cn", "红绿灯")
            parts.append(f"{d.get('position', '前方')}有{state_cn}")
        else:
            parts.append(f"{d.get('position', '前方')}有{d.get('label_cn', '物体')}")

    return "，".join(parts)


def build_blind_guidance(detections: list) -> dict:
    if not detections:
        return {
            "risk_level": "caution",
            "safety_text": "暂时没有识别到明确障碍，但系统可能漏检。请保持慢速直行，不要快速前进。",
            "decision": "slow_forward"
        }

    traffic_lights = [d for d in detections if d.get("label_en") == "traffic light"]
    vehicles = [d for d in detections if d.get("label_en") in VEHICLE_LABELS]
    obstacles = [d for d in detections if d.get("label_en") in OBSTACLE_LABELS]

    near_or_front_vehicles = [
        d for d in vehicles
        if "前方" in d.get("position", "") or "近处" in d.get("position", "")
    ]

    near_or_front_obstacles = [
        d for d in obstacles
        if "前方" in d.get("position", "") or "近处" in d.get("position", "")
    ]

    if near_or_front_vehicles:
        scene = summarize_items(near_or_front_vehicles, 2)
        return {
            "risk_level": "stop",
            "safety_text": f"我看到{scene}。请立即停止等待，不要进入前方区域。",
            "decision": "stop_for_vehicle"
        }

    if traffic_lights:
        traffic_lights_sorted = sorted(
            traffic_lights,
            key=lambda d: d.get("confidence", 0),
            reverse=True
        )

        main_light = traffic_lights_sorted[0]
        state = main_light.get("traffic_light_state", "unknown")

        if state == "red":
            return {
                "risk_level": "stop",
                "safety_text": "前方检测到红灯。请停止等待，不要进入路口。",
                "decision": "stop_red_light"
            }

        if state == "yellow":
            return {
                "risk_level": "stop",
                "safety_text": "前方检测到黄灯。请停止等待，不要抢行。",
                "decision": "stop_yellow_light"
            }

        if state == "unknown":
            return {
                "risk_level": "stop",
                "safety_text": "前方检测到红绿灯，但暂时无法判断颜色。请停止等待，不要进入路口。",
                "decision": "stop_unknown_light"
            }

        if state == "green":
            if near_or_front_obstacles:
                scene = summarize_items(near_or_front_obstacles, 2)
                return {
                    "risk_level": "stop",
                    "safety_text": f"前方检测到绿灯，但同时检测到{scene}。请先停止，避开障碍后再继续。",
                    "decision": "green_but_obstacle"
                }

            return {
                "risk_level": "caution",
                "safety_text": "前方检测到绿灯，当前没有识别到近处车辆或障碍。可以缓慢直行通过，保持直线行走。",
                "decision": "green_light_slow_pass"
            }

    if near_or_front_obstacles:
        scene = summarize_items(near_or_front_obstacles, 3)
        bypass = choose_bypass_direction(near_or_front_obstacles)

        return {
            "risk_level": "stop",
            "safety_text": f"我看到{scene}。{bypass}",
            "decision": "avoid_obstacle"
        }

    side_hazards = [d for d in detections if d.get("is_hazard")]
    if side_hazards:
        scene = summarize_items(side_hazards, 3)
        return {
            "risk_level": "caution",
            "safety_text": f"我看到{scene}。请保持慢速前进，不要靠近障碍物一侧。",
            "decision": "side_hazard_slow"
        }

    scene = summarize_items(detections, 3)
    return {
        "risk_level": "caution",
        "safety_text": f"我看到{scene}。请缓慢前进，等待下一次系统提示。",
        "decision": "slow_forward_with_objects"
    }


def build_traffic_light_text(detections: list) -> str:
    traffic_lights = [d for d in detections if d.get("label_en") == "traffic light"]
    vehicles = [d for d in detections if d.get("label_en") in VEHICLE_LABELS]
    obstacles = [d for d in detections if d.get("label_en") in OBSTACLE_LABELS]

    near_or_front_vehicles = [
        d for d in vehicles
        if "前方" in d.get("position", "") or "近处" in d.get("position", "")
    ]

    near_or_front_obstacles = [
        d for d in obstacles
        if "前方" in d.get("position", "") or "近处" in d.get("position", "")
    ]

    if not traffic_lights:
        return "暂时没有检测到红绿灯。请停止等待，不要进入路口。"

    main_light = sorted(
        traffic_lights,
        key=lambda d: d.get("confidence", 0),
        reverse=True
    )[0]

    state = main_light.get("traffic_light_state", "unknown")

    if state == "red":
        return "前方是红灯。请停止等待，不要进入路口。"

    if state == "yellow":
        return "前方是黄灯。请停止等待，不要抢行。"

    if state == "unknown":
        return "检测到红绿灯，但暂时无法判断颜色。请停止等待。"

    if state == "green":
        if near_or_front_vehicles:
            scene = summarize_items(near_or_front_vehicles, 2)
            return f"前方是绿灯，但检测到{scene}。请停止等待，不要立刻通过。"

        if near_or_front_obstacles:
            scene = summarize_items(near_or_front_obstacles, 2)
            return f"前方是绿灯，但检测到{scene}。请先避开障碍，再缓慢通过。"

        return "前方是绿灯，当前没有识别到近处车辆或障碍。可以缓慢直行通过，保持直线行走。"

    return "红绿灯状态不确定。请停止等待。"


def annotate_image(image_bgr, detections: list):
    annotated = image_bgr.copy()

    for d in detections:
        x1, y1, x2, y2 = d["box"]

        if d.get("label_en") == "traffic light":
            state = d.get("traffic_light_state", "unknown")
            label = f"traffic light {state} {d['confidence']}"
        else:
            label = f"{d['label_en']} {d['confidence']}"

        if d.get("label_en") == "traffic light":
            state = d.get("traffic_light_state", "unknown")
            if state == "red":
                color = (0, 0, 255)
            elif state == "yellow":
                color = (0, 220, 255)
            elif state == "green":
                color = (0, 200, 0)
            else:
                color = (180, 180, 180)
        else:
            color = (0, 0, 255) if d["is_hazard"] else (0, 180, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return annotated


def update_yolo_cache(annotated_jpg: bytes, result_data: dict):
    global latest_annotated_jpg, latest_annotated_time, latest_yolo_result

    with cache_lock:
        latest_annotated_jpg = annotated_jpg
        latest_annotated_time = time.time()
        latest_yolo_result = result_data


def run_yolo_pipeline_from_esp32(
    esp32_url: str,
    mode: str = "display",
    make_annotated: bool = True
):
    """
    mode:
    display  -> 512，展示用，更准，有标注图
    fast     -> 320，导盲用，更快，默认不画框
    traffic  -> 512，红绿灯专项
    """

    start_time = time.time()

    image_bytes = fetch_esp32_image(esp32_url)

    raw_bgr = image_bytes_to_cv2(image_bytes)
    enhanced_bgr = enhance_image_for_detection(raw_bgr)

    if mode == "fast":
        model_kind = "fast"
        imgsz = YOLO_FAST_IMGSZ
        classes = GUIDE_CLASS_IDS
    elif mode == "traffic":
        model_kind = "accurate"
        imgsz = YOLO_TRAFFIC_IMGSZ
        classes = TRAFFIC_CLASS_IDS
    else:
        model_kind = "accurate"
        imgsz = YOLO_DISPLAY_IMGSZ
        classes = GUIDE_CLASS_IDS

    detections, actual_model_path = yolo_detect_image(
        enhanced_bgr,
        image_for_color=raw_bgr,
        imgsz=imgsz,
        classes=classes,
        model_kind=model_kind
    )

    guidance = build_blind_guidance(detections)
    safety_text = sanitize_guidance_language(guidance["safety_text"])
    traffic_text = sanitize_guidance_language(build_traffic_light_text(detections))

    elapsed_ms = round((time.time() - start_time) * 1000)

    result_data = {
        "ok": True,
        "esp32_url": esp32_url,
        "mode": mode,
        "detections": detections,
        "risk_level": guidance["risk_level"],
        "decision": guidance["decision"],
        "safety_text": safety_text,
        "traffic_light_text": traffic_text,
        "model_path": actual_model_path,
        "yolo_conf": YOLO_CONF,
        "yolo_imgsz": imgsz,
        "elapsed_ms": elapsed_ms,
        "cache_time": time.time()
    }

    if make_annotated:
        annotated = annotate_image(enhanced_bgr, detections)
        annotated_jpg = encode_jpeg(annotated, quality=82)
        update_yolo_cache(annotated_jpg, result_data)

    return result_data


def get_asr_model():
    global _asr_model, _asr_model_name

    with _asr_lock:
        if _asr_model is not None:
            return _asr_model, _asr_model_name

        try:
            from faster_whisper import WhisperModel
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"faster-whisper 没有安装或无法导入：{e}"
            )

        candidates = [ASR_MODEL]
        if ASR_MODEL != "tiny":
            candidates.append("tiny")

        last_error = None

        for model_name in candidates:
            try:
                print(
                    "Loading ASR model: "
                    f"{model_name}, device={ASR_DEVICE}, compute_type={ASR_COMPUTE_TYPE}"
                )
                _asr_model = WhisperModel(
                    model_name,
                    device=ASR_DEVICE,
                    compute_type=ASR_COMPUTE_TYPE
                )
                _asr_model_name = model_name
                print(f"ASR model loaded: {model_name}")
                return _asr_model, _asr_model_name
            except Exception as e:
                last_error = e
                print(f"ASR model load failed: {model_name}, error: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"ASR 模型加载失败，已尝试 {candidates}：{last_error}"
        )


def transcribe_wav_bytes(wav_bytes: bytes) -> str:
    if not wav_bytes:
        return ""

    model, _ = get_asr_model()
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(wav_bytes)
            temp_path = f.name

        segments, _ = model.transcribe(
            temp_path,
            language="zh",
            vad_filter=True
        )

        texts = []
        for segment in segments:
            text = getattr(segment, "text", "")
            if text and text.strip():
                texts.append(text.strip())

        return "".join(texts).strip()
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except Exception:
                pass


def fetch_esp32_record_wav(esp32_url: str, seconds: int) -> bytes:
    esp32_url = esp32_url.rstrip("/")
    seconds = max(1, min(5, int(seconds)))
    record_url = f"{esp32_url}/record.wav?seconds={seconds}"

    try:
        response = requests.get(record_url, timeout=seconds + 20)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"无法从 ESP32 获取录音：{e}"
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"ESP32 录音失败：{response.status_code}，{response.text}"
        )

    if not response.content:
        raise HTTPException(
            status_code=500,
            detail="ESP32 返回的录音为空"
        )

    return response.content


def contains_any(text: str, keywords: list) -> bool:
    return any(keyword in text for keyword in keywords)


def run_text_command_from_esp32(text: str, esp32_url: str) -> tuple:
    if contains_any(text, ["停止巡检", "关闭巡检"]):
        return "stop_auto", None

    if contains_any(text, ["开启巡检", "开始巡检", "自动巡检"]):
        return "start_auto", None

    if contains_any(text, ["红绿灯", "红灯", "绿灯", "交通灯"]):
        return "traffic_light", traffic_light_once(
            esp32_url=esp32_url,
            force=True,
            show_image=False
        )

    if contains_any(text, ["详细", "描述", "周围"]):
        return "describe", analyze_and_speak(esp32_url=esp32_url)

    if contains_any(text, ["识别", "检测", "看一下"]):
        return "guide", guide_once(
            esp32_url=esp32_url,
            force=True,
            show_image=False
        )

    return "question", ask_esp32(
        esp32_url=esp32_url,
        question=text,
        speak=True
    )


def execute_esp32_mic_command(esp32_url: str, seconds: int) -> dict:
    retry_message = "没有听清，请再说一遍。"
    seconds = max(1, min(5, int(seconds)))

    try:
        wav_bytes = fetch_esp32_record_wav(esp32_url, seconds)
        text = transcribe_wav_bytes(wav_bytes)
    except HTTPException as e:
        response_data = {
            "ok": False,
            "command_type": "error",
            "text": "",
            "error": e.detail
        }
        add_operation_log(
            "esp32_mic_command",
            ok=False,
            detail=f"esp32_url={esp32_url}, error={e.detail}"
        )
        return response_data
    except Exception as e:
        response_data = {
            "ok": False,
            "command_type": "error",
            "text": "",
            "error": f"ESP32 mic command failed: {e}"
        }
        add_operation_log(
            "esp32_mic_command",
            ok=False,
            detail=f"esp32_url={esp32_url}, error={e}"
        )
        return response_data

    if not text:
        speak_result = None
        try:
            speak_result = send_to_esp32_speak(
                esp32_url=esp32_url,
                text=retry_message
            )
        except Exception as e:
            speak_result = {"ok": False, "error": str(e)}

        response_data = {
            "ok": False,
            "command_type": "no_speech",
            "text": "",
            "message": retry_message,
            "speak_result": speak_result
        }
        add_operation_log(
            "esp32_mic_command",
            ok=False,
            detail=f"esp32_url={esp32_url}, no_speech"
        )
        return response_data

    try:
        command_type, result = run_text_command_from_esp32(text, esp32_url)
    except HTTPException as e:
        response_data = {
            "ok": False,
            "command_type": "error",
            "text": text,
            "error": e.detail
        }
        add_operation_log(
            "esp32_mic_command",
            ok=False,
            detail=f"text={text[:80]}, error={e.detail}"
        )
        return response_data
    except Exception as e:
        response_data = {
            "ok": False,
            "command_type": "error",
            "text": text,
            "error": f"Voice command execution failed: {e}"
        }
        add_operation_log(
            "esp32_mic_command",
            ok=False,
            detail=f"text={text[:80]}, error={e}"
        )
        return response_data

    response_data = {
        "ok": True,
        "text": text,
        "command_type": command_type,
        "result": result
    }
    add_operation_log(
        "esp32_mic_command",
        ok=True,
        detail=f"text={text[:80]}, command_type={command_type}"
    )
    return response_data


def should_speak_now(text: str, min_repeat_interval: int = 8) -> bool:
    global last_speak_text, last_speak_time

    now = time.time()

    if text == last_speak_text and now - last_speak_time < min_repeat_interval:
        return False

    last_speak_text = text
    last_speak_time = now
    return True


def probe_http_endpoint(url: str, timeout: int = 3, expect_binary: bool = False) -> dict:
    start_time = time.time()

    try:
        response = requests.get(url, timeout=timeout)
    except Exception as e:
        result = {
            "ok": False,
            "url": url,
            "elapsed_ms": round((time.time() - start_time) * 1000),
            "error": str(e)
        }
        return result

    result = {
        "ok": response.status_code == 200,
        "url": url,
        "status_code": response.status_code,
        "elapsed_ms": round((time.time() - start_time) * 1000),
        "content_type": response.headers.get("content-type", "")
    }

    if response.status_code == 200:
        if expect_binary:
            result["bytes"] = len(response.content)
        else:
            try:
                result["data"] = response.json()
            except Exception:
                result["preview"] = response.text[:200]
    else:
        result["error"] = response.text[:200]

    return result


def run_health_check(esp32_url: str = ESP32_BASE_URL) -> dict:
    esp32_url = esp32_url.rstrip("/")

    with cache_lock:
        cached_result = None
        if latest_yolo_result:
            cached_result = {
                "mode": latest_yolo_result.get("mode"),
                "model_path": latest_yolo_result.get("model_path"),
                "elapsed_ms": latest_yolo_result.get("elapsed_ms"),
                "cache_time": latest_yolo_result.get("cache_time")
            }

    esp32_status = probe_http_endpoint(f"{esp32_url}/status", timeout=3)
    if esp32_status.get("ok"):
        esp32_capture = probe_http_endpoint(
            f"{esp32_url}/capture.jpg",
            timeout=5,
            expect_binary=True
        )
    else:
        esp32_capture = {
            "ok": False,
            "url": f"{esp32_url}/capture.jpg",
            "skipped": True,
            "reason": "esp32_status_failed"
        }

    health = {
        "ok": True,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": "running",
        "esp32_base_url": esp32_url,
        "checks": {
            "esp32_status": esp32_status,
            "esp32_capture": esp32_capture,
            "yolo": {
                "dependencies_ready": YOLO is not None and cv2 is not None and np is not None,
                "fast_model_path": YOLO_FAST_MODEL_PATH,
                "fast_model_exists": os.path.exists(YOLO_FAST_MODEL_PATH),
                "accurate_model_path": YOLO_ACCURATE_MODEL_PATH,
                "accurate_model_exists": os.path.exists(YOLO_ACCURATE_MODEL_PATH),
                "loaded_models": sorted(_yolo_models.keys())
            },
            "doubao": {
                "api_key_configured": bool(AI_API_KEY),
                "base_url": AI_BASE_URL,
                "model": AI_MODEL
            },
            "asr": {
                "configured_model": ASR_MODEL,
                "device": ASR_DEVICE,
                "compute_type": ASR_COMPUTE_TYPE,
                "loaded_model": _asr_model_name,
                "module_available": True
            },
            "latest_yolo_result": cached_result
        }
    }

    try:
        __import__("faster_whisper")
    except Exception as e:
        health["checks"]["asr"]["module_available"] = False
        health["checks"]["asr"]["error"] = str(e)

    health["ok"] = (
        health["checks"]["esp32_status"].get("ok", False)
        and health["checks"]["esp32_capture"].get("ok", False)
        and health["checks"]["yolo"]["dependencies_ready"]
    )

    return health


# ==================== API ====================

@app.get("/")
def index():
    return {
        "ok": True,
        "message": "Lingxi AI backend is running",
        "model": AI_MODEL,
        "esp32_base_url": ESP32_BASE_URL,
        "yolo_model": YOLO_FAST_MODEL_PATH,
        "yolo_fast_model": YOLO_FAST_MODEL_PATH,
        "yolo_accurate_model": YOLO_ACCURATE_MODEL_PATH,
        "yolo_conf": YOLO_CONF,
        "yolo_display_imgsz": YOLO_DISPLAY_IMGSZ,
        "yolo_fast_imgsz": YOLO_FAST_IMGSZ,
        "yolo_traffic_imgsz": YOLO_TRAFFIC_IMGSZ,
        "asr_model": ASR_MODEL,
        "asr_device": ASR_DEVICE,
        "asr_compute_type": ASR_COMPUTE_TYPE,
        "endpoints": [
            "/control",
            "/health",
            "/health_page",
            "/live",
            "/logs",
            "/proxy_capture.jpg",
            "/analyze_esp32",
            "/analyze_and_speak",
            "/ask_esp32",
            "/asr_upload",
            "/esp32_mic_command",
            "/yolo_detect_esp32",
            "/guide_once",
            "/traffic_light_once",
            "/yolo_speak_once",
            "/yolo_annotated.jpg"
        ]
    }


@app.get("/health")
def health(
    esp32_url: str = Query(ESP32_BASE_URL)
):
    return run_health_check(esp32_url)


@app.get("/logs")
def logs():
    with operation_log_lock:
        items = list(operation_logs)

    return {
        "ok": True,
        "items": items
    }


@app.get("/proxy_capture.jpg")
def proxy_capture_jpg(
    esp32_url: str = Query(ESP32_BASE_URL)
):
    image_bytes = fetch_esp32_image(esp32_url)

    return Response(
        image_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"
        }
    )


@app.post("/analyze_upload")
async def analyze_upload(file: UploadFile = File(...)):
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="上传图片为空")

    image_data_url = image_bytes_to_data_url(image_bytes)
    description = call_doubao_vision(image_data_url)
    speak_text = clean_speak_text(description)

    return {
        "ok": True,
        "description": description,
        "speak_text": speak_text
    }


@app.post("/analyze_esp32")
def analyze_esp32(
    esp32_url: str = Query(ESP32_BASE_URL)
):
    image_bytes = fetch_esp32_image(esp32_url)
    image_data_url = image_bytes_to_data_url(image_bytes)

    description = call_doubao_vision(image_data_url)
    speak_text = clean_speak_text(description)

    return {
        "ok": True,
        "esp32_url": esp32_url,
        "description": description,
        "speak_text": speak_text
    }


@app.post("/analyze")
def analyze(
    image_url: str = Query(...)
):
    description = call_doubao_vision(image_url)
    speak_text = clean_speak_text(description)

    return {
        "ok": True,
        "description": description,
        "speak_text": speak_text
    }


@app.post("/analyze_and_speak")
def analyze_and_speak(
    esp32_url: str = Query(ESP32_BASE_URL)
):
    image_bytes = fetch_esp32_image(esp32_url)
    image_data_url = image_bytes_to_data_url(image_bytes)

    description = call_doubao_vision(image_data_url)
    speak_text = clean_speak_text(description)

    speak_result = send_to_esp32_speak(
        esp32_url=esp32_url,
        text=speak_text
    )

    result = {
        "ok": True,
        "esp32_url": esp32_url,
        "description": description,
        "speak_text": speak_text,
        "speak_result": speak_result
    }
    add_operation_log(
        "analyze_and_speak",
        ok=True,
        detail=f"esp32_url={esp32_url}, speak_len={len(speak_text)}"
    )
    return result


@app.post("/ask_esp32")
def ask_esp32(
    esp32_url: str = Query(ESP32_BASE_URL),
    question: str = Query("前方有什么？"),
    speak: bool = Query(True)
):
    image_bytes = fetch_esp32_image(esp32_url)
    image_data_url = image_bytes_to_data_url(image_bytes)

    answer = call_doubao_question(image_data_url, question)
    speak_text = clean_speak_text(answer)

    speak_result = None
    if speak:
        speak_result = send_to_esp32_speak(
            esp32_url=esp32_url,
            text=speak_text
        )

    result = {
        "ok": True,
        "esp32_url": esp32_url,
        "question": question,
        "answer": answer,
        "speak_text": speak_text,
        "speak_result": speak_result
    }
    add_operation_log(
        "ask_esp32",
        ok=True,
        detail=f"esp32_url={esp32_url}, question={question[:80]}"
    )
    return result


@app.post("/yolo_detect_esp32")
def yolo_detect_esp32(
    esp32_url: str = Query(ESP32_BASE_URL)
):
    """
    展示用：512 + 标注图。
    """
    result_data = run_yolo_pipeline_from_esp32(
        esp32_url=esp32_url,
        mode="display",
        make_annotated=True
    )
    add_operation_log(
        "yolo_detect_esp32",
        ok=True,
        detail=(
            f"mode={result_data['mode']}, model={result_data['model_path']}, "
            f"imgsz={result_data['yolo_imgsz']}, elapsed_ms={result_data['elapsed_ms']}"
        )
    )
    return result_data


@app.post("/guide_once")
def guide_once(
    esp32_url: str = Query(ESP32_BASE_URL),
    force: bool = Query(True),
    show_image: bool = Query(False)
):
    """
    真正导盲用：320 极速模式。
    默认不生成标注图，速度更快。
    """

    result_data = run_yolo_pipeline_from_esp32(
        esp32_url=esp32_url,
        mode="fast",
        make_annotated=show_image
    )

    safety_text = result_data["safety_text"]

    spoken = False
    speak_result = None

    if force or should_speak_now(safety_text):
        speak_result = send_to_esp32_speak(
            esp32_url=esp32_url,
            text=safety_text
        )
        spoken = True

    result_data["spoken"] = spoken
    result_data["speak_result"] = speak_result

    add_operation_log(
        "guide_once",
        ok=True,
        detail=(
            f"mode={result_data['mode']}, model={result_data['model_path']}, "
            f"imgsz={result_data['yolo_imgsz']}, spoken={spoken}"
        )
    )
    return result_data


@app.post("/traffic_light_once")
def traffic_light_once(
    esp32_url: str = Query(ESP32_BASE_URL),
    force: bool = Query(True),
    show_image: bool = Query(False)
):
    """
    红绿灯专项：512，兼顾速度和红绿灯识别率。
    """

    result_data = run_yolo_pipeline_from_esp32(
        esp32_url=esp32_url,
        mode="traffic",
        make_annotated=show_image
    )

    traffic_text = sanitize_guidance_language(result_data["traffic_light_text"])

    spoken = False
    speak_result = None

    if force or should_speak_now(traffic_text):
        speak_result = send_to_esp32_speak(
            esp32_url=esp32_url,
            text=traffic_text
        )
        spoken = True

    result_data["traffic_text"] = traffic_text
    result_data["spoken"] = spoken
    result_data["speak_result"] = speak_result

    add_operation_log(
        "traffic_light_once",
        ok=True,
        detail=(
            f"mode={result_data['mode']}, model={result_data['model_path']}, "
            f"imgsz={result_data['yolo_imgsz']}, spoken={spoken}"
        )
    )
    return result_data


# 兼容旧按钮
@app.post("/yolo_speak_once")
def yolo_speak_once(
    esp32_url: str = Query(ESP32_BASE_URL),
    force: bool = Query(False)
):
    return guide_once(
        esp32_url=esp32_url,
        force=force,
        show_image=False
    )


@app.post("/asr_upload")
async def asr_upload(file: UploadFile = File(...)):
    wav_bytes = await file.read()
    text = transcribe_wav_bytes(wav_bytes)

    result = {
        "ok": True,
        "text": text
    }
    add_operation_log(
        "asr_upload",
        ok=True,
        detail=f"text={text[:80]}"
    )
    return result


@app.post("/esp32_mic_command")
def esp32_mic_command(
    esp32_url: str = Query(ESP32_BASE_URL),
    seconds: int = Query(3)
):
    return execute_esp32_mic_command(esp32_url, seconds)

    seconds = max(1, min(5, int(seconds)))

    try:
        wav_bytes = fetch_esp32_record_wav(esp32_url, seconds)
        text = transcribe_wav_bytes(wav_bytes)
    except HTTPException as e:
        result = {
            "ok": False,
            "command_type": "error",
            "text": "",
            "error": e.detail
        }
        add_operation_log(
            "esp32_mic_command",
            ok=False,
            detail=f"esp32_url={esp32_url}, error={e.detail}"
        )
        return result
    except Exception as e:
        return {
            "ok": False,
            "command_type": "error",
            "text": "",
            "error": f"ESP32 麦克风识别失败：{e}"
        }

    if not text:
        speak_result = None
        try:
            speak_result = send_to_esp32_speak(
                esp32_url=esp32_url,
                text="没有听清，请再说一遍。"
            )
        except Exception as e:
            speak_result = {"ok": False, "error": str(e)}

        return {
            "ok": False,
            "command_type": "no_speech",
            "text": "",
            "message": "没有听清，请再说一遍。",
            "speak_result": speak_result
        }

    try:
        command_type, result = run_text_command_from_esp32(text, esp32_url)
    except HTTPException as e:
        return {
            "ok": False,
            "command_type": "error",
            "text": text,
            "error": e.detail
        }
    except Exception as e:
        return {
            "ok": False,
            "command_type": "error",
            "text": text,
            "error": f"语音命令执行失败：{e}"
        }

    return {
        "ok": True,
        "text": text,
        "command_type": command_type,
        "result": result
    }


@app.get("/yolo_annotated.jpg")
def yolo_annotated_jpg(
    esp32_url: str = Query(ESP32_BASE_URL),
    use_cache: bool = Query(True)
):
    global latest_annotated_jpg

    if use_cache:
        with cache_lock:
            if latest_annotated_jpg is not None:
                return Response(
                    latest_annotated_jpg,
                    media_type="image/jpeg",
                    headers={
                        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"
                    }
                )

    run_yolo_pipeline_from_esp32(
        esp32_url=esp32_url,
        mode="display",
        make_annotated=True
    )

    with cache_lock:
        if latest_annotated_jpg is None:
            raise HTTPException(
                status_code=500,
                detail="没有可用的 YOLO 标注图"
            )

        return Response(
            latest_annotated_jpg,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"
            }
        )


# ==================== 控制页 ====================

@app.get("/health_page", response_class=HTMLResponse)
def health_page():
    html = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>Lingxi Health</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 24px; background: #f5f6f8; color: #111; }
        .card { max-width: 980px; background: #fff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
        input { width: 360px; padding: 8px; }
        button { padding: 8px 14px; margin-left: 8px; cursor: pointer; }
        pre { background: #111; color: #eaeaea; padding: 14px; border-radius: 8px; min-height: 260px; white-space: pre-wrap; word-break: break-word; }
    </style>
</head>
<body>
<div class="card">
    <h1>Health Check</h1>
    <p>
        <input id="esp32_url" value="http://192.168.43.188">
        <button onclick="refreshHealth()">Refresh</button>
    </p>
    <pre id="health_result">Waiting...</pre>
</div>
<script>
async function refreshHealth() {
    const resultBox = document.getElementById("health_result");
    const esp32Url = document.getElementById("esp32_url").value.trim();
    resultBox.innerText = "Running health check...";

    try {
        const response = await fetch("/health?esp32_url=" + encodeURIComponent(esp32Url));
        const data = await response.json();
        resultBox.innerText = JSON.stringify(data, null, 2);
    } catch (e) {
        resultBox.innerText = "Health check failed: " + e;
    }
}

refreshHealth();
</script>
</body>
</html>
"""
    return html


@app.get("/control", response_class=HTMLResponse)
def control_page():
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>灵犀之眼 一键识别播报</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 30px; background: #f6f6f6; }
        .card { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.12); max-width: 900px; }
        input { width: 90%; padding: 10px; font-size: 16px; margin-bottom: 12px; }
        button { padding: 12px 18px; font-size: 18px; cursor: pointer; margin-right: 10px; margin-bottom: 8px; }
        pre { background: #222; color: #eee; padding: 14px; border-radius: 8px; white-space: pre-wrap; word-break: break-word; min-height: 150px; }
        a { display: inline-block; margin-top: 12px; }
    </style>
</head>
<body>
    <div class="card">
        <h1>灵犀之眼 一键识别并播报</h1>

        <p>ESP32 地址：</p>
        <input id="esp32_url" value="http://192.168.43.188">

        <p>交互问题：</p>
        <input id="question" value="前方有什么障碍？">

        <br>
        <button onclick="runAnalyze()">豆包完整识别</button>
        <button onclick="runAnalyzeSpeak()">豆包识别并播报</button>
        <button onclick="askSpeak()">文字提问并播报</button>

        <p><a href="/live" target="_blank">打开交互式导盲页面</a></p>

        <h3>结果：</h3>
        <pre id="result">等待操作...</pre>
    </div>

<script>
async function runAnalyze() {
    const resultBox = document.getElementById("result");
    const esp32Url = document.getElementById("esp32_url").value.trim();

    resultBox.innerText = "正在调用豆包完整识别，请稍等...";

    try {
        const url = "/analyze_esp32?esp32_url=" + encodeURIComponent(esp32Url);
        const response = await fetch(url, { method: "POST" });
        const data = await response.json();

        resultBox.innerText =
            "识别结果：\\n" + data.description +
            "\\n\\n实际播报：\\n" + data.speak_text;
    } catch (e) {
        resultBox.innerText = "请求失败：" + e;
    }
}

async function runAnalyzeSpeak() {
    const resultBox = document.getElementById("result");
    const esp32Url = document.getElementById("esp32_url").value.trim();

    resultBox.innerText = "正在识别并播报，请稍等...";

    try {
        const url = "/analyze_and_speak?esp32_url=" + encodeURIComponent(esp32Url);
        const response = await fetch(url, { method: "POST" });
        const data = await response.json();

        resultBox.innerText =
            "识别结果：\\n" + data.description +
            "\\n\\n实际播报：\\n" + data.speak_text;
    } catch (e) {
        resultBox.innerText = "请求失败：" + e;
    }
}

async function askSpeak() {
    const resultBox = document.getElementById("result");
    const esp32Url = document.getElementById("esp32_url").value.trim();
    const question = document.getElementById("question").value.trim();

    resultBox.innerText = "正在根据你的问题识别并播报...";

    try {
        const url =
            "/ask_esp32?esp32_url=" + encodeURIComponent(esp32Url) +
            "&question=" + encodeURIComponent(question) +
            "&speak=true";

        const response = await fetch(url, { method: "POST" });
        const data = await response.json();

        resultBox.innerText =
            "问题：\\n" + data.question +
            "\\n\\n回答：\\n" + data.answer +
            "\\n\\n实际播报：\\n" + data.speak_text;
    } catch (e) {
        resultBox.innerText = "请求失败：" + e;
    }
}
</script>
</body>
</html>
"""
    return html


# ==================== 交互式导盲页面 ====================

@app.get("/live", response_class=HTMLResponse)
def live_page():
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>灵犀之眼 交互式导盲系统</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 25px; background: #f3f3f3; }
        .card { background: white; padding: 18px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.12); margin-bottom: 18px; }
        input { padding: 8px; font-size: 16px; }
        button { padding: 10px 14px; font-size: 16px; margin: 5px; cursor: pointer; }
        img { max-width: 100%; border: 1px solid #ccc; border-radius: 8px; background: #111; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
        pre { background: #222; color: #eee; padding: 12px; border-radius: 8px; min-height: 150px; white-space: pre-wrap; word-break: break-word; }
        .small { color: #666; font-size: 14px; }
        .danger { color: #b00020; font-weight: bold; }
        .ok { color: #087a2f; font-weight: bold; }
    </style>
</head>
<body>

<div class="card">
    <h1>灵犀之眼 交互式导盲系统</h1>

    <p>ESP32 地址：</p>
    <input id="esp32_url" value="http://192.168.43.188" style="width: 360px;">

    <p>交互问题：</p>
    <input id="question" value="前方有什么障碍？" style="width: 420px;">

    <p>自动巡检间隔：</p>
    <input id="interval_ms" value="2500" style="width: 100px;"> ms

    <br><br>

    <button onclick="startLive()">开始实时画面</button>
    <button onclick="stopLive()">停止实时画面</button>

    <br>

    <button onclick="yoloOnce()">展示识别一次 512</button>
    <button onclick="guideOnce(true, false)">极速导盲播报 320</button>
    <button onclick="trafficLightOnce(true, false)">红绿灯专项判断 512</button>
    <button onclick="askSpeak()">文字提问并播报</button>
    <button onclick="esp32MicCommand()">ESP32麦克风提问</button>
    <button onclick="runHealthCheck()">健康检查</button>

    <br>

    <button onclick="doubaoDescribeSpeak()">详细环境描述并播报</button>
    <button onclick="startAuto()">开始极速自动巡检</button>
    <button onclick="stopAuto()">停止自动巡检</button>
    <button onclick="startVoiceCommand()">开始语音指令</button>

    <p class="small">
        真实导盲走极速通道：320 推理，不画标注图，优先速度。展示识别走 512，有标注图，详细准确。
    </p>

    <p class="danger">
        导盲策略：不确定时默认停止等待；不再使用“请注意观察”这类无效提示。
    </p>
</div>

<div class="grid">
    <div class="card">
        <h2>实时画面</h2>
        <img id="live_img" src="" width="520">
    </div>

    <div class="card">
        <h2>YOLO 标注画面</h2>
        <img id="yolo_img" src="" width="520">
        <p class="small">标注图只在“展示识别一次 512”后刷新，自动巡检不会刷新它。</p>
    </div>
</div>

<div class="card">
    <h2>识别结果</h2>
    <pre id="result">等待操作...</pre>
</div>

<div class="card">
    <h2>最近操作日志</h2>
    <pre id="op_logs">Loading...</pre>
</div>

<script>
let liveTimer = null;
let autoTimer = null;
let autoBusy = false;
let voiceBusy = false;
let autoPausedForVoice = false;
let resumeAutoAfterVoice = false;
let liveRunning = false;
let operationLogsTimer = null;

function getEsp32Url() {
    return document.getElementById("esp32_url").value.trim();
}

function getQuestion() {
    return document.getElementById("question").value.trim();
}

function getIntervalMs() {
    let v = parseInt(document.getElementById("interval_ms").value);

    if (isNaN(v) || v < 2000) {
        v = 2000;
        document.getElementById("interval_ms").value = "2000";
    }

    return v;
}

async function runHealthCheck() {
    const resultBox = document.getElementById("result");
    const esp32Url = getEsp32Url();

    resultBox.innerText = "正在执行健康检查...";

    try {
        const response = await fetch(
            "/health?esp32_url=" + encodeURIComponent(esp32Url)
        );
        const data = await response.json();
        resultBox.innerText = JSON.stringify(data, null, 2);
    } catch (e) {
        resultBox.innerText = "健康检查失败: " + e;
    } finally {
        refreshOperationLogs();
    }
}

async function refreshOperationLogs() {
    const logsBox = document.getElementById("op_logs");

    if (!logsBox) {
        return;
    }

    try {
        const response = await fetch("/logs");
        const data = await response.json();
        const items = Array.isArray(data.items) ? data.items : [];

        if (!items.length) {
            logsBox.innerText = "暂无最近操作日志。";
            return;
        }

        logsBox.innerText = items.map((item) => {
            const status = item.ok ? "OK" : "FAIL";
            const detail = item.detail || "";
            const extra = item.extra ? "\\n" + JSON.stringify(item.extra, null, 2) : "";
            return `[${item.time}] ${item.kind} ${status}\\n${detail}${extra}`;
        }).join("\\n\\n");
    } catch (e) {
        logsBox.innerText = "日志加载失败: " + e;
    }
}

function refreshLiveImage() {
    const esp32Url = getEsp32Url();

    document.getElementById("live_img").src =
        "/proxy_capture.jpg?esp32_url=" +
        encodeURIComponent(esp32Url) +
        "&t=" + Date.now();
}

function refreshYoloImage() {
    const esp32Url = getEsp32Url();

    document.getElementById("yolo_img").src =
        "/yolo_annotated.jpg?use_cache=true&esp32_url=" +
        encodeURIComponent(esp32Url) +
        "&t=" + Date.now();
}

function startLive() {
    stopLive();

    liveRunning = true;
    refreshLiveImage();

    liveTimer = setInterval(() => {
        if (liveRunning && !autoBusy) {
            refreshLiveImage();
        }
    }, 2500);
}

function stopLive() {
    liveRunning = false;

    if (liveTimer) {
        clearInterval(liveTimer);
        liveTimer = null;
    }
}

async function yoloOnce() {
    const resultBox = document.getElementById("result");
    const esp32Url = getEsp32Url();

    stopLive();

    resultBox.innerText = "正在进行展示识别，使用 512 模式并生成标注图...";

    try {
        const url = "/yolo_detect_esp32?esp32_url=" + encodeURIComponent(esp32Url);

        const response = await fetch(url, { method: "POST" });
        const data = await response.json();

        if (!data.ok) {
            resultBox.innerText = "识别失败：\\n" + JSON.stringify(data, null, 2);
            return;
        }

        resultBox.innerText =
            "导盲提示：\\n" + data.safety_text +
            "\\n\\n模式：" + data.mode +
            "\\n模型：" + data.model_path +
            "\\n耗时：" + data.elapsed_ms + " ms" +
            "\\n风险等级：" + data.risk_level +
            "\\n决策：" + data.decision +
            "\\n\\n红绿灯专项提示：\\n" + data.traffic_light_text +
            "\\n\\nYOLO参数：conf=" + data.yolo_conf + ", imgsz=" + data.yolo_imgsz +
            "\\n\\n检测结果：\\n" + JSON.stringify(data.detections, null, 2);

        refreshYoloImage();
    } catch (e) {
        resultBox.innerText = "识别失败：" + e;
    }
}

async function guideOnce(force=true, showImage=false) {
    const resultBox = document.getElementById("result");
    const esp32Url = getEsp32Url();

    stopLive();

    resultBox.innerText = "正在进行极速导盲判断并播报，使用 320 模式...";

    try {
        const url =
            "/guide_once?esp32_url=" +
            encodeURIComponent(esp32Url) +
            "&force=" + force +
            "&show_image=" + showImage;

        const response = await fetch(url, { method: "POST" });
        const data = await response.json();

        if (!data.ok) {
            resultBox.innerText = "导盲判断失败：\\n" + JSON.stringify(data, null, 2);
            return;
        }

        resultBox.innerText =
            "导盲提示：\\n" + data.safety_text +
            "\\n\\n模式：" + data.mode +
            "\\n模型：" + data.model_path +
            "\\n耗时：" + data.elapsed_ms + " ms" +
            "\\n风险等级：" + data.risk_level +
            "\\n决策：" + data.decision +
            "\\n是否播报：" + data.spoken +
            "\\n\\nYOLO参数：conf=" + data.yolo_conf + ", imgsz=" + data.yolo_imgsz +
            "\\n\\n检测结果：\\n" + JSON.stringify(data.detections, null, 2);

        if (showImage) {
            refreshYoloImage();
        }
    } catch (e) {
        resultBox.innerText = "导盲判断失败：" + e;
    }
}

async function trafficLightOnce(force=true, showImage=false) {
    const resultBox = document.getElementById("result");
    const esp32Url = getEsp32Url();

    stopLive();

    resultBox.innerText = "正在进行红绿灯专项判断，使用 512 模式...";

    try {
        const url =
            "/traffic_light_once?esp32_url=" +
            encodeURIComponent(esp32Url) +
            "&force=" + force +
            "&show_image=" + showImage;

        const response = await fetch(url, { method: "POST" });
        const data = await response.json();

        if (!data.ok) {
            resultBox.innerText = "红绿灯判断失败：\\n" + JSON.stringify(data, null, 2);
            return;
        }

        resultBox.innerText =
            "红绿灯提示：\\n" + data.traffic_text +
            "\\n\\n模式：" + data.mode +
            "\\n模型：" + data.model_path +
            "\\n耗时：" + data.elapsed_ms + " ms" +
            "\\n是否播报：" + data.spoken +
            "\\n\\nYOLO参数：conf=" + data.yolo_conf + ", imgsz=" + data.yolo_imgsz +
            "\\n\\n检测结果：\\n" + JSON.stringify(data.detections, null, 2);

        if (showImage) {
            refreshYoloImage();
        }
    } catch (e) {
        resultBox.innerText = "红绿灯判断失败：" + e;
    }
}

async function askSpeak() {
    const resultBox = document.getElementById("result");
    const esp32Url = getEsp32Url();
    const question = getQuestion();
    const managePriority = !voiceBusy;

    stopLive();

    if (managePriority) {
        pauseAutoForVoice();
        voiceBusy = true;
    }

    resultBox.innerText = "正在根据你的问题识别并播报，走豆包简短回答模式...";

    try {
        const url =
            "/ask_esp32?esp32_url=" + encodeURIComponent(esp32Url) +
            "&question=" + encodeURIComponent(question) +
            "&speak=true";

        const response = await fetch(url, { method: "POST" });
        const data = await response.json();

        if (!data.ok) {
            resultBox.innerText = "提问失败：\\n" + JSON.stringify(data, null, 2);
            return;
        }

        resultBox.innerText =
            "问题：\\n" + data.question +
            "\\n\\n回答：\\n" + data.answer +
            "\\n\\n实际播报：\\n" + data.speak_text;
    } catch (e) {
        resultBox.innerText = "提问失败：" + e;
    } finally {
        if (managePriority) {
            const shouldShowResume = resumeAutoAfterVoice;
            voiceBusy = false;
            resumeAutoAfterVoiceCommand();

            if (shouldShowResume) {
                resultBox.innerText += "\\n\\n语音任务完成，自动巡检已恢复。";
            }
        }
    }
}

async function doubaoDescribeSpeak() {
    const resultBox = document.getElementById("result");
    const esp32Url = getEsp32Url();
    const managePriority = !voiceBusy;

    stopLive();

    if (managePriority) {
        pauseAutoForVoice();
        voiceBusy = true;
    }

    resultBox.innerText = "正在进行详细环境描述并播报，走豆包模式...";

    try {
        const url =
            "/analyze_and_speak?esp32_url=" +
            encodeURIComponent(esp32Url);

        const response = await fetch(url, { method: "POST" });
        const data = await response.json();

        if (!data.ok) {
            resultBox.innerText = "详细描述失败：\\n" + JSON.stringify(data, null, 2);
            return;
        }

        resultBox.innerText =
            "详细识别结果：\\n" + data.description +
            "\\n\\n实际播报：\\n" + data.speak_text;
    } catch (e) {
        resultBox.innerText = "详细描述失败：" + e;
    } finally {
        if (managePriority) {
            const shouldShowResume = resumeAutoAfterVoice;
            voiceBusy = false;
            resumeAutoAfterVoiceCommand();

            if (shouldShowResume) {
                resultBox.innerText += "\\n\\n语音任务完成，自动巡检已恢复。";
            }
        }
    }
}

async function esp32MicCommand() {
    const resultBox = document.getElementById("result");
    const esp32Url = getEsp32Url();
    let commandType = "";

    stopLive();
    pauseAutoForVoice();
    voiceBusy = true;

    resultBox.innerText = "正在使用 ESP32 板载麦克风录音，请说话……";

    try {
        const url =
            "/esp32_mic_command?esp32_url=" +
            encodeURIComponent(esp32Url) +
            "&seconds=3";

        const response = await fetch(url, { method: "POST" });
        const data = await response.json();
        commandType = data.command_type || "";

        let displayText =
            "语音识别文字：\\n" + (data.text || "") +
            "\\n\\n执行的命令类型：\\n" + commandType +
            "\\n\\n执行结果：\\n" + JSON.stringify(data.result || data, null, 2);

        if (!data.ok && data.error) {
            displayText += "\\n\\n错误：\\n" + data.error;
        }

        if (commandType === "stop_auto") {
            resumeAutoAfterVoice = false;
            voiceBusy = false;
            stopAuto();
            resultBox.innerText =
                displayText +
                "\\n\\n已根据语音指令停止自动巡检。";
            return;
        }

        if (commandType === "start_auto") {
            resumeAutoAfterVoice = false;
            autoPausedForVoice = false;
            voiceBusy = false;
            startAuto();
            resultBox.innerText = displayText;
            return;
        }

        resultBox.innerText = displayText;
    } catch (e) {
        resultBox.innerText = "ESP32 麦克风提问失败：" + e;
    } finally {
        if (commandType !== "stop_auto" && commandType !== "start_auto") {
            const shouldShowResume = resumeAutoAfterVoice;
            voiceBusy = false;
            resumeAutoAfterVoiceCommand();

            if (shouldShowResume) {
                resultBox.innerText += "\\n\\n语音任务完成，自动巡检已恢复。";
            }
        }
    }
}

function startAuto() {
    stopAuto();
    stopLive();

    const resultBox = document.getElementById("result");
    resultBox.innerText = "极速自动巡检已启动。系统会使用 320 模式快速判断风险，不刷新标注图。";

    autoTimer = setInterval(async () => {
        if (autoBusy || voiceBusy || autoPausedForVoice) return;

        autoBusy = true;

        try {
            await guideOnce(false, false);
        } finally {
            autoBusy = false;
        }
    }, getIntervalMs());
}

function stopAuto() {
    if (autoTimer) {
        clearInterval(autoTimer);
        autoTimer = null;
    }

    autoBusy = false;
    autoPausedForVoice = false;
    document.getElementById("result").innerText = "自动安全巡检已停止。";
}

function isStopAutoCommand(command) {
    const text = command.trim();
    return text.includes("停止巡检") || text.includes("关闭巡检");
}

function pauseAutoForVoice() {
    if (autoTimer) {
        resumeAutoAfterVoice = true;
        autoPausedForVoice = true;
    }
}

function resumeAutoAfterVoiceCommand() {
    if (resumeAutoAfterVoice) {
        autoPausedForVoice = false;
    }

    resumeAutoAfterVoice = false;
}

function startVoiceCommand() {
    const resultBox = document.getElementById("result");

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        resultBox.innerText = "当前浏览器不支持语音指令。请使用 Chrome 浏览器。";
        return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "zh-CN";
    recognition.continuous = false;
    recognition.interimResults = false;

    pauseAutoForVoice();
    voiceBusy = true;
    resultBox.innerText = "语音优先模式已启动，自动巡检已临时暂停，请说指令。";

    recognition.onresult = async function(event) {
        const command = event.results[0][0].transcript;
        const stopAutoCommand = isStopAutoCommand(command);

        resultBox.innerText = "识别到语音指令：\\n" + command;

        try {
            await handleVoiceCommand(command);
        } finally {
            voiceBusy = false;

            if (!stopAutoCommand) {
                const shouldShowResume = resumeAutoAfterVoice;
                resumeAutoAfterVoiceCommand();

                if (shouldShowResume) {
                    resultBox.innerText = "语音任务完成，自动巡检已恢复。";
                }
            }
        }
    };

    recognition.onerror = function(event) {
        voiceBusy = false;
        resumeAutoAfterVoiceCommand();
        resultBox.innerText = "语音识别失败：" + event.error;
    };

    recognition.start();
}

async function handleVoiceCommand(command) {
    const text = command.trim();

    if (isStopAutoCommand(text)) {
        stopAuto();
        resumeAutoAfterVoice = false;
        autoPausedForVoice = false;
        document.getElementById("result").innerText = "已根据语音指令停止自动巡检。";
        return;
    }

    if (text.includes("开启巡检") || text.includes("开始巡检") || text.includes("自动巡检")) {
        startAuto();
        return;
    }

    if (text.includes("红绿灯") || text.includes("红灯") || text.includes("绿灯") || text.includes("交通灯")) {
        await trafficLightOnce(true, false);
        return;
    }

    if (text.includes("详细") || text.includes("描述") || text.includes("周围")) {
        await doubaoDescribeSpeak();
        return;
    }

    if (text.includes("识别") || text.includes("检测") || text.includes("看一下")) {
        await guideOnce(true, false);
        return;
    }

    document.getElementById("question").value = text;
    await askSpeak();
}

refreshOperationLogs();
if (operationLogsTimer) {
    clearInterval(operationLogsTimer);
}
operationLogsTimer = setInterval(refreshOperationLogs, 5000);
startLive();
</script>

</body>
</html>
"""
    return html
