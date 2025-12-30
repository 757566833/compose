from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, HttpUrl
import torch
from transformers import AutoModel
import tempfile
import shutil
import os
import cv2
import numpy as np
from PIL import Image
import httpx  # 用于异步下载视频
from funasr import AutoModel as FunASRAutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

app = FastAPI(title="Video Visual Feature Service (8003)")

# --- 模型加载与初始化 ---
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

model_dir = "FunAudioLLM/SenseVoiceSmall"
# 1. 语音识别 (Whisper small)
try:
    audio_model = FunASRAutoModel(
        model=model_dir,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
        hub="hf",
    )
except Exception as e:
    print(f"ERROR: Could not load Whisper model on {device}, falling back to CPU. Detail: {e}")

# 2. CLIP 模型
try:
    # 加载 CLIP 模型和处理器
    image_model = AutoModel.from_pretrained(
        "jinaai/jina-clip-v2", 
        trust_remote_code=True,
    ).to(device)
    print(f"Model loaded successfully on device: {device}")
except Exception as e:
    print(f"Error loading model: {e}")
truncate_dim = 512


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
    """使用 jina-clip-v2 的核心处理逻辑"""
    
    # 1. 听觉处理 (保持不变)
    print(f"Processing audio for: {filename}")
    # transcription_result = asr_pipeline(video_path)
    res = audio_model.generate(
            input=video_path,
            cache={},
            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,  #
            merge_length_s=15,
        )
    transcribed_text = rich_transcription_postprocess(res[0]["text"])

    # 2. 视觉处理
    print(f"Extracting frames at {sampling_fps} FPS...")
    frames = extract_frames(video_path, fps=sampling_fps) # 确保返回的是 PIL 图像列表
    
    visual_vector = None
    frames_processed = 0
    truncate_dim = 512

    if frames:
        # Jina CLIP v2 直接支持传入 PIL 图像列表
        # model.encode_image 内部处理了预处理、特征提取和 L2 归一化
        with torch.no_grad():
            # 返回形状为 (num_frames, truncate_dim) 的 numpy 数组
            image_embeddings = image_model.encode_image(frames, truncate_dim=truncate_dim)
        
        # 将多帧向量取平均，得到代表视频的全局视觉向量
        # jina-clip 默认已做归一化，均值后建议再次归一化以保持单位长度
        avg_vector = image_embeddings.mean(axis=0)
        # 再次归一化 (可选，但推荐用于检索)
        
        visual_vector = (avg_vector / np.linalg.norm(avg_vector)).tolist()
        
        frames_processed = len(frames)

    return {
        "filename": filename,
        "transcribed_text": transcribed_text,
        "visual_vector": visual_vector,
        "vector_dimension": truncate_dim if visual_vector else 0,
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