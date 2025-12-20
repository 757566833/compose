from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel, HttpUrl
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
class TranscribeRequest(BaseModel):
    url: HttpUrl
@app.post("/transcribe/audio/url")
async def transcribe_url(request_data: TranscribeRequest):
    """
    接收 JSON Body 中的 url，下载后进行转录。
    """
    # 将 HttpUrl 对象转换为字符串
    url_str = str(request_data.url)

    print(f"Downloading audio from URL: {url_str}")
    
    # 提取后缀：使用 Pydantic 对象的 path 属性更优雅，它会自动排除 Query 参数
    url_path = request_data.url.path or ""
    suffix = os.path.splitext(url_path)[1] or ".mp3"
    
    tmp_path = None
    try:
        # 2. 创建临时文件
        # 注意：先关闭文件句柄，让后面的 open(tmp_path, "wb") 可以跨平台安全写入
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
        
        # 3. 使用 httpx 流式下载
        # 增加了 follow_redirects=True，因为很多云存储 URL 会重定向
        async with httpx.AsyncClient(timeout=600.0) as client:
            async with client.stream("GET", url_str, follow_redirects=True) as response:
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Failed to download file from URL. HTTP {response.status_code}"
                    )
                
                with open(tmp_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)

        # 4. 调用转录处理函数
        # 传入下载好的本地路径和原始 URL 字符串
        return await process_transcription(tmp_path, url_str)

    except Exception as e:
        # 如果出错，清理可能残留的临时文件
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            
        # 如果已经是 HTTPException 则直接抛出，否则封装为 500
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error downloading audio: {str(e)}")

# 建议在 process_transcription 内部最后执行 os.remove(tmp_path)
async def process_transcription(tmp_path: str, original_source: str):
    try:
        # 调用 Whisper
        transcription_result = asr_pipeline(
            tmp_path,
            chunk_length_s=30,      # 每个分块的时长
            stride_length_s=5,      # 分块间的重叠时长，增加上下文衔接的准确性
            batch_size=8,           # 如果显存允许，增加 batch_size 可以提升速度
            return_timestamps=True,
            generate_kwargs={
                "language": "chinese",
                "task": "transcribe"
            }
        )
        
        return {
            "source": original_source,
            "transcribed_text": transcription_result["text"].strip(),
            "device_used": device,
        }

    except Exception as e:
        print(f"Transcription Error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription processing error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/health")
def health_check():
    return {"status": "ok", "model": model_id, "device": str(device)}