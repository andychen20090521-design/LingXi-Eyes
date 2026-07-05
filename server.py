from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import Response
import pyttsx3
import tempfile
import os
import uuid
import wave
import time
import threading
import struct

app = FastAPI()

# ==================== TTS 设置 ====================
# 必须和 ESP32 main.cpp 里的 PCM_SAMPLE_RATE 保持一致
TARGET_SAMPLE_RATE = 24000

# 语速：越大越快
TTS_RATE = 190

# 音量：0.0 到 1.0
TTS_VOLUME = 1.0

tts_lock = threading.Lock()


# ==================== 选择女声 ====================
def get_voice_text(voice) -> str:
    parts = []

    try:
        parts.append(str(voice.id))
    except:
        pass

    try:
        parts.append(str(voice.name))
    except:
        pass

    try:
        parts.append(str(voice.languages))
    except:
        pass

    try:
        parts.append(str(voice.gender))
    except:
        pass

    return " ".join(parts).lower()


def choose_best_female_voice(engine):
    voices = engine.getProperty("voices")

    if not voices:
        return None

    candidates = []

    for voice in voices:
        text = get_voice_text(voice)
        score = 0

        # 中文优先
        if "zh" in text:
            score += 50
        if "chinese" in text:
            score += 50
        if "china" in text:
            score += 30

        # 常见中文女声
        if "xiaoxiao" in text:
            score += 120
        if "huihui" in text:
            score += 110
        if "yaoyao" in text:
            score += 100
        if "hanhan" in text:
            score += 90

        # 女声优先
        if "female" in text:
            score += 40
        if "zira" in text:
            score += 20

        # 男声降低优先级
        if "male" in text:
            score -= 40
        if "yunxi" in text:
            score -= 30
        if "kangkang" in text:
            score -= 30

        candidates.append((score, voice))

    candidates.sort(key=lambda x: x[0], reverse=True)

    best_score, best_voice = candidates[0]

    print("Selected voice:")
    print("  score:", best_score)
    print("  id:", best_voice.id)
    print("  name:", best_voice.name)

    return best_voice.id


# ==================== 生成 WAV ====================
def synth_to_wav(text: str) -> str:
    wav_path = os.path.join(
        tempfile.gettempdir(),
        f"tts_{uuid.uuid4().hex}.wav"
    )

    with tts_lock:
        engine = pyttsx3.init()

        voice_id = choose_best_female_voice(engine)
        if voice_id:
            engine.setProperty("voice", voice_id)

        engine.setProperty("rate", TTS_RATE)
        engine.setProperty("volume", TTS_VOLUME)

        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        engine.stop()

    # 等待文件生成完成
    for _ in range(30):
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            break
        time.sleep(0.1)

    if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        raise HTTPException(status_code=500, detail="TTS 生成失败")

    return wav_path


# ==================== WAV 转 16-bit 单声道采样数组 ====================
def wav_to_int16_samples(wav_path: str):
    with wave.open(wav_path, "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    samples = []

    frame_size = sample_width * channels
    total_frames = len(frames) // frame_size

    for frame_index in range(total_frames):
        base = frame_index * frame_size
        channel_values = []

        for ch in range(channels):
            offset = base + ch * sample_width
            raw = frames[offset:offset + sample_width]

            if sample_width == 1:
                # 8-bit WAV 通常是无符号，转成 signed 16-bit
                value = (raw[0] - 128) << 8

            elif sample_width == 2:
                value = int.from_bytes(raw, byteorder="little", signed=True)

            elif sample_width == 3:
                # 24-bit signed little-endian
                sign = 0xFF if raw[2] & 0x80 else 0x00
                raw4 = raw + bytes([sign])
                value32 = int.from_bytes(raw4, byteorder="little", signed=True)
                value = value32 >> 8

            elif sample_width == 4:
                value32 = int.from_bytes(raw, byteorder="little", signed=True)
                value = value32 >> 16

            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"不支持的采样宽度：{sample_width} bytes"
                )

            channel_values.append(value)

        # 转单声道：多声道取平均
        mono = int(sum(channel_values) / len(channel_values))

        if mono > 32767:
            mono = 32767
        if mono < -32768:
            mono = -32768

        samples.append(mono)

    return samples, sample_rate


# ==================== 纯 Python 重采样到 24000Hz ====================
def resample_linear(samples, source_rate: int, target_rate: int):
    if source_rate == target_rate:
        return samples

    if not samples:
        return samples

    target_len = int(len(samples) * target_rate / source_rate)

    if target_len <= 1:
        return samples

    result = []

    ratio = source_rate / target_rate

    for i in range(target_len):
        src_pos = i * ratio
        idx = int(src_pos)
        frac = src_pos - idx

        if idx >= len(samples) - 1:
            value = samples[-1]
        else:
            s1 = samples[idx]
            s2 = samples[idx + 1]
            value = int(s1 + (s2 - s1) * frac)

        if value > 32767:
            value = 32767
        if value < -32768:
            value = -32768

        result.append(value)

    return result


def samples_to_pcm16(samples) -> bytes:
    out = bytearray()

    for s in samples:
        if s > 32767:
            s = 32767
        if s < -32768:
            s = -32768

        out += struct.pack("<h", int(s))

    return bytes(out)


def wav_to_pcm_for_esp32(wav_path: str) -> bytes:
    samples, source_rate = wav_to_int16_samples(wav_path)

    samples = resample_linear(
        samples,
        source_rate=source_rate,
        target_rate=TARGET_SAMPLE_RATE
    )

    return samples_to_pcm16(samples)


# ==================== API ====================
@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "TTS server is running",
        "voice": "auto female chinese voice",
        "rate": TTS_RATE,
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "channels": 1,
        "bits": 16
    }


@app.get("/voices")
def list_voices():
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")

    result = []

    for index, voice in enumerate(voices):
        result.append({
            "index": index,
            "id": voice.id,
            "name": voice.name,
            "languages": str(getattr(voice, "languages", "")),
            "gender": str(getattr(voice, "gender", "")),
        })

    engine.stop()

    return {
        "count": len(result),
        "voices": result
    }


@app.get("/preview.wav")
def preview_wav(
    text: str = Query(
        "你好，语音测试成功。前方有椅子，请小心慢行。",
        max_length=120
    )
):
    wav_path = synth_to_wav(text)

    try:
        with open(wav_path, "rb") as f:
            data = f.read()

        return Response(
            data,
            media_type="audio/wav"
        )

    finally:
        try:
            os.remove(wav_path)
        except:
            pass


@app.get("/tts.pcm")
def tts_pcm(text: str = Query(..., max_length=120)):
    wav_path = synth_to_wav(text)

    try:
        pcm = wav_to_pcm_for_esp32(wav_path)

        return Response(
            pcm,
            media_type="application/octet-stream",
            headers={
                "X-Sample-Rate": str(TARGET_SAMPLE_RATE),
                "X-Channels": "1",
                "X-Bits": "16"
            }
        )

    finally:
        try:
            os.remove(wav_path)
        except:
            pass