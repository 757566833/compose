from fastapi import FastAPI, File, UploadFile, HTTPException, Query
import torch
from transformers import pipeline
import tempfile
import shutil
import os
import httpx  # 用于下载远程文件

app = FastAPI(title="Audio Transcription Service")

# --- 模型加载与初始化 ---
# 自动检测设备：CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model_id = "openai/whisper-small" 

try:
    # 这里的 device 参数会自动处理 MPS 或 CUDA 的分配
    asr_pipeline = pipeline(
        "automatic-speech-recognition", 
        model=model_id, 
        device=device
    )
    print(f"Whisper Model loaded successfully on device: {device}")
except Exception as e:
    print(f"Error loading Whisper model: {e}")

# --- API 接口定义 ---

@app.post("/transcribe/audio/file")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    接收音频文件，支持长音频转录。
    """
    if not file.content_type.startswith(('audio/', 'application/octet-stream')):
        raise HTTPException(status_code=400, detail="File must be an audio format.")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        try:
            shutil.copyfileobj(file.file, tmp)
        except Exception:
            if os.path.exists(tmp_path): os.remove(tmp_path)
            raise HTTPException(status_code=500, detail="Failed to save audio file locally.")
            
    return await process_transcription(tmp_path, file.filename)

@app.post("/transcribe/audio/url")
async def transcribe_url(url: str = Query(..., description="音频文件的公开 URL")):
    """
    接收音频 URL，下载后进行转录。
    """
    # 从 URL 提取后缀，默认为 .mp3
    path_without_query = url.split('?')[0]
    suffix = os.path.splitext(path_without_query)[1] or ".mp3"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        try:
            # 使用 httpx 流式下载，防止大文件撑爆内存
            async with httpx.AsyncClient(timeout=1200000.0) as client:
                async with client.stream("GET", url) as response:
                    if response.status_code != 200:
                        raise HTTPException(status_code=400, detail=f"Failed to download file from URL. HTTP {response.status_code}")
                    with open(tmp_path, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
        except Exception as e:
            if os.path.exists(tmp_path): os.remove(tmp_path)
            raise HTTPException(status_code=500, detail=f"Error downloading audio: {str(e)}")

    return await process_transcription(tmp_path, url)

async def process_transcription(tmp_path: str, original_source: str):
    """
    核心转录逻辑抽取，供内部复用。
    """
    try:
        # 调用 Whisper 
        transcription_result = asr_pipeline(
            tmp_path,
            chunk_length_s=30,
            return_timestamps=True,
            generate_kwargs={"language": "chinese"}
        )
        
        return {
            "source": original_source,
            "transcribed_text": transcription_result["text"].strip(),
            "device_used": device,
            "vector_ready": True 
        }

    except Exception as e:
        print(f"Transcription Error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription processing error: {str(e)}")
    finally:
        # 彻底清理临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/health")
def health_check():
    return {"status": "ok", "model": model_id, "device": str(device)}