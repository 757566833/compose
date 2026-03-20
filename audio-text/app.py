from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, HttpUrl
import torch
import tempfile
import shutil
import os
import httpx  # 用于下载远程文件
from pathlib import Path
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import logging
from typing import Optional, Dict
import struct
import re

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 文件验证函数
def validate_audio_file(file_path: str) -> bool:
    """通过文件头验证音频/视频文件类型"""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(12)  # 读取前12字节

        # 常见音频/视频文件头检查
        # MP3: ID3 tag或FF FB/FF F3等
        # WAV: "RIFF"
        # FLAC: "fLaC"
        # OGG: "OggS"
        # MP4/M4A: "ftyp" at offset 4
        # AVI: "RIFF"

        if len(header) < 4:
            return False

        # 检查常见格式
        if header.startswith(b'RIFF'):  # WAV, AVI
            return True
        elif header.startswith(b'fLaC'):  # FLAC
            return True
        elif header.startswith(b'OggS'):  # OGG
            return True
        elif header.startswith(b'ID3'):  # MP3 with ID3 tag
            return True
        elif len(header) >= 8 and header[4:8] == b'ftyp':  # MP4, M4A
            return True
        elif len(header) >= 2 and header[:2] in (b'\xFF\xFB', b'\xFF\xF3', b'\xFF\xFA'):  # MP3 frame
            return True

        return False
    except Exception:
        return False

def get_safe_filename(filename: Optional[str]) -> str:
    """安全处理文件名"""
    if not filename:
        return "file"

    # 移除路径遍历攻击
    # 移除目录路径，只保留文件名
    name = filename.split("/")[-1].split("\\")[-1]
    # 只允许字母、数字、点、下划线、连字符
    name = re.sub(r'[^\w\.\-]', '_', name)
    # 确保不以点开头（隐藏文件）
    if name.startswith('.'):
        name = '_' + name[1:]
    # 如果过滤后为空，返回默认值
    return name if name.strip() else "file"

app = FastAPI(title="Audio Transcription Service")

MODEL_BASE = os.getenv("MODEL_PATH", "/models")
# --- 模型加载与初始化 ---
# 自动检测设备：CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model_dir = f"{MODEL_BASE}/SenseVoiceSmall"
try:
    # 这里的 device 参数会自动处理 MPS 或 CUDA 的分配
    model = AutoModel(
        model=model_dir,
        vad_model=f"{MODEL_BASE}/fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
        hub="hf",
        trust_remote_code=True,     # 必须开启，以运行模型目录下的 Python 代码
        disable_update=True         # 离线运行必备
    )
except Exception as e:
    logger.error(f"Error loading Whisper model: {e}")

SUPPORTED_TYPES = ('audio/', 'video/', 'application/octet-stream')

@app.post("/transcribe/audio/file")
async def transcribe_audio(file: UploadFile = File(...)):  # 无文件大小限制
    """
    接收音频文件，支持长音频转录。
    """
    logger.info(f"Received audio file upload: {file.filename}, size: {file.size}, type: {file.content_type}")

    # 1. 基础内容类型检查
    if not file.content_type.startswith(SUPPORTED_TYPES):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Please upload an audio or video file."
        )

    # 2. 安全文件名处理
    safe_filename = get_safe_filename(file.filename)
    suffix = Path(safe_filename).suffix if safe_filename != "file" else ".mp3"

    # 3. 保存到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        try:
            shutil.copyfileobj(file.file, tmp)
            tmp.flush()  # 确保数据写入
        except Exception as e:
            logger.error(f"Failed to save audio file: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise HTTPException(status_code=500, detail="Failed to save audio file locally.")

    # 4. 文件头验证
    try:
        if not validate_audio_file(tmp_path):
            logger.warning(f"File header validation failed for: {file.filename}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise HTTPException(
                status_code=400,
                detail="Invalid audio/video file format. Please upload a valid audio or video file."
            )
    except Exception as e:
        logger.error(f"File validation error: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail="File validation error.")

    logger.info(f"File validated successfully: {file.filename}")
    return await process_transcription(tmp_path, safe_filename)
class TranscribeRequest(BaseModel):
    url: HttpUrl
    headers: Optional[Dict[str, str]] = None
@app.post("/transcribe/audio/url")
async def transcribe_url(request_data: TranscribeRequest):
    """
    接收 JSON Body 中的 url，下载后进行转录。
    """
    url_str = str(request_data.url)
    logger.info(f"Processing audio URL transcription: {url_str}")

    # 1. 安全处理路径和文件名
    url_path = request_data.url.path or ""
    safe_filename = get_safe_filename(url_path.split("/")[-1] if "/" in url_path else url_path)
    suffix = Path(safe_filename).suffix if safe_filename != "file" else ".mp3"

    tmp_path = None

    try:
        # 2. 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name

        # 3. 使用 httpx 流式下载（无大小限制）
        download_headers = request_data.headers or {}
        async with httpx.AsyncClient(timeout=600.0) as client:
            async with client.stream("GET", url_str, follow_redirects=True, headers=download_headers) as response:
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download file from URL. HTTP {response.status_code}"
                    )

                with open(tmp_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)

        logger.info(f"File downloaded successfully: {downloaded_size} bytes")

        # 4. 文件头验证
        if not validate_audio_file(tmp_path):
            logger.warning(f"File header validation failed for URL: {url_str}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise HTTPException(
                status_code=400,
                detail="Invalid audio/video file format from URL. Please provide a valid audio or video file URL."
            )

        logger.info(f"URL file validated successfully: {url_str}")
        return await process_transcription(tmp_path, url_str)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading audio from URL {url_str}: {e}")
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Error downloading audio: {str(e)}")

# 建议在 process_transcription 内部最后执行 os.remove(tmp_path)
async def process_transcription(tmp_path: str, original_source: str):
    logger.info(f"Processing transcription for source: {original_source}")
    try:
        # 调用 Whisper
        # transcription_result = asr_pipeline(
        #     tmp_path,
        #     chunk_length_s=30,      # 每个分块的时长
        #     stride_length_s=5,      # 分块间的重叠时长，增加上下文衔接的准确性
        #     batch_size=8,           # 如果显存允许，增加 batch_size 可以提升速度
        #     return_timestamps=True,
        #     generate_kwargs={
        #         "language": "chinese",
        #         "task": "transcribe"
        #     }
        # )
        res = model.generate(
            input=tmp_path,
            cache={},
            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,  #
            merge_length_s=15,
        )

        logger.info(f"Transcription completed successfully for {original_source}")
        return {
            "source": original_source,
            "transcribed_text": rich_transcription_postprocess(res[0]["text"]),
            "device_used": device,
        }

    except Exception as e:
        logger.error(f"Transcription Error for {original_source}: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription processing error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/health")
def health_check():
    return {"status": "ok", "model": model_dir, "device": str(device)}