from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
from transformers import pipeline, CLIPProcessor, CLIPModel
import tempfile
import shutil
import os
import cv2
import numpy as np
from PIL import Image

app = FastAPI(title="Video Visual Feature Service (8003)")

# --- 模型加载与初始化 ---
# 优先使用 GPU 或 Mac M4 的 MPS，否则使用 CPU
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# 1. 语音识别 (Whisper small)
try:
    asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)
    print(f"Whisper Model loaded on device: {device}")
except Exception as e:
    print(f"ERROR: Could not load Whisper model. Detail: {e}")
    # 失败时尝试在 CPU 上重新加载
    asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small", device="cpu")


# 2. CLIP 模型
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print(f"CLIP Model loaded on device: {device}")


# --- 辅助函数：处理文件和提取帧 ---

def save_upload_file_to_temp(upload_file: UploadFile) -> str:
    """将上传文件保存到临时文件并返回路径"""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            return tmp.name
    except Exception as e:
        print(f"File saving error: {e}")
        raise HTTPException(status_code=500, detail="Failed to save video file temporarily.")

def extract_frames(video_path: str, fps: int = 1) -> list[Image.Image]:
    """从视频中每隔固定秒数提取一帧，返回 PIL Image 列表"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        raise ValueError("Cannot determine video frame rate.")
        
    frame_interval = max(1, int(video_fps / fps))
    
    current_frame = 0
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # OpenCV BGR -> PIL Image (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        
        current_frame += frame_interval
        
    cap.release()
    return frames

# --- 核心接口：整合后的处理接口 ---

@app.post("/video/process")
async def video_process(file: UploadFile = File(...), sampling_fps: int = 1):
    """
    接收视频文件，同时执行音频转文本和视觉向量提取，返回整合结果。
    """
    
    tmp_path = save_upload_file_to_temp(file)
            
    try:
        # 1. 听觉处理：音频转文本 (Whisper)
        print("Starting audio transcription...")
        transcription_result = asr_pipeline(tmp_path)
        transcribed_text = transcription_result["text"].strip()
        
        # 2. 视觉处理：提取纯视觉向量 (CLIP)
        print(f"Extracting frames at {sampling_fps} FPS...")
        frames = extract_frames(tmp_path, fps=sampling_fps)
        
        visual_vector = None
        frames_processed = 0

        if frames:
            # 批量计算 CLIP 视觉向量
            clip_inputs = clip_processor(images=frames, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = clip_model.get_image_features(**clip_inputs)
            
            # 平均池化：L2 规范化后，取平均
            normalized_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            visual_vector = normalized_features.mean(dim=0).cpu().numpy().tolist() # 512-dim list
            frames_processed = len(frames)
        
        # 3. 返回整合结果
        return {
            "filename": file.filename,
            "transcribed_text": transcribed_text,
            "visual_vector": visual_vector,
            "vector_dimension": 512 if visual_vector else 0,
            "frames_processed": frames_processed,
            "modality_source": "audio_text_and_visual"
        }
    
    except Exception as e:
        print(f"Processing Error: {e}")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {e}")
    finally:
        os.remove(tmp_path)

# 健康检查
@app.get("/health")
def health_check():
    return {"status": "ok", "models": ["Whisper-Small", "CLIP-ViT-B/32"], "device": str(device)}