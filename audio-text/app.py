from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
from transformers import pipeline
import tempfile
import shutil
import os

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

@app.post("/transcribe/audio")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    接收音频文件，支持长音频转录。
    """
    # 简单的格式校验
    if not file.content_type.startswith(('audio/', 'application/octet-stream')):
        raise HTTPException(status_code=400, detail="File must be an audio format.")

    # 1. 保存到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        try:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        except Exception:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)
            raise HTTPException(status_code=500, detail="Failed to save audio file temporarily.")
            
    try:
        # 2. 调用 Whisper 进行转录
        # chunk_length_s=30: 核心改动，自动切分超过30秒的音频
        # return_timestamps=True: 长音频模式必备参数
        # generate_kwargs: 强制指定中文识别，避免误判
        transcription_result = asr_pipeline(
            tmp_path,
            chunk_length_s=30,
            return_timestamps=True,
            generate_kwargs={"language": "chinese"}
        )
        
        # 3. 提取结果
        transcribed_text = transcription_result["text"]

        return {
            "filename": file.filename,
            "transcribed_text": transcribed_text.strip(),
            "device_used": device,
            "vector_ready": True 
        }

    except Exception as e:
        print(f"Transcription Error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription processing error: {str(e)}")
    finally:
        # 4. 彻底清理临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# 健康检查接口
@app.get("/health")
def health_check():
    return {"status": "ok", "model": model_name, "device": str(device)}
