from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, HttpUrl
import torch
from transformers import pipeline, CLIPProcessor, CLIPModel
import tempfile
import shutil
import os
import cv2
import numpy as np
from PIL import Image
import httpx  # 用于异步下载视频

app = FastAPI(title="Video Visual Feature Service (8003)")

# --- 模型加载与初始化 ---
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# 1. 语音识别 (Whisper small)
try:
    asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)
    print(f"Whisper Model loaded on device: {device}")
except Exception as e:
    print(f"ERROR: Could not load Whisper model on {device}, falling back to CPU. Detail: {e}")
    asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small", device="cpu")

# 2. CLIP 模型
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print(f"CLIP Model loaded on device: {device}")


# --- 数据模型 ---

class TranscribeRequest(BaseModel):
    url: HttpUrl


# --- 辅助函数 ---

def extract_frames(video_path: str, fps: int = 1) -> list[Image.Image]:
    """从视频中提取帧"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 25  # 默认兜底帧率
        
    frame_interval = max(1, int(video_fps / fps))
    current_frame = 0
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        current_frame += frame_interval
        
    cap.release()
    return frames

async def download_video(url: str) -> str:
    """下载远程视频到临时文件"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise HTTPException(status_code=400, detail=f"Failed to download video from {url}")
                for chunk in response.iter_bytes():
                    tmp.write(chunk)
        return tmp.name
    except Exception as e:
        if os.path.exists(tmp.name): os.remove(tmp.name)
        raise HTTPException(status_code=500, detail=f"Download error: {str(e)}")
    finally:
        tmp.close()

def internal_process_logic(video_path: str, sampling_fps: int, filename: str):
    """通用的核心处理逻辑：Whisper + CLIP"""
    # 1. 听觉处理
    print(f"Processing audio for: {filename}")
    transcription_result = asr_pipeline(video_path)
    transcribed_text = transcription_result["text"].strip()

    # 2. 视觉处理
    print(f"Extracting frames at {sampling_fps} FPS...")
    frames = extract_frames(video_path, fps=sampling_fps)
    
    visual_vector = None
    frames_processed = 0

    if frames:
        clip_inputs = clip_processor(images=frames, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**clip_inputs)
        
        # 归一化并取平均
        normalized_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        visual_vector = normalized_features.mean(dim=0).cpu().numpy().tolist()
        frames_processed = len(frames)

    return {
        "filename": filename,
        "transcribed_text": transcribed_text,
        "visual_vector": visual_vector,
        "vector_dimension": 512 if visual_vector else 0,
        "frames_processed": frames_processed,
    }


# --- 接口 1：文件上传 (原有功能) ---

@app.post("/video/file")
async def video_file(file: UploadFile = File(...), sampling_fps: int = 1):
    """接收上传的视频文件进行处理"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = internal_process_logic(tmp_path, sampling_fps, file.filename)
        result["modality_source"] = "file_upload"
        return result
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)


# --- 接口 2：URL 触发 (新增功能) ---

@app.post("/video/url")
async def video_url(request: TranscribeRequest, sampling_fps: int = 1):
    """
    接收视频 URL，下载并执行音频转文本和视觉向量提取。
    """
    url_str = str(request.url)
    print(f"Request received for URL: {url_str}")
    
    # 1. 下载文件
    tmp_path = await download_video(url_str)
    
    try:
        # 2. 调用核心逻辑
        filename = os.path.basename(url_str.split('?')[0]) # 提取文件名，去除 URL 参数
        result = internal_process_logic(tmp_path, sampling_fps, filename)
        result["modality_source"] = "url_download"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL processing failed: {str(e)}")
    finally:
        # 3. 清理临时文件
        if os.path.exists(tmp_path): os.remove(tmp_path)


# 健康检查
@app.get("/health")
def health_check():
    return {"status": "ok", "device": str(device)}