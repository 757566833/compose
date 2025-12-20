from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from PIL import Image
from pydantic import BaseModel, HttpUrl
import io
import torch
import httpx
import tempfile
import os
from transformers import CLIPProcessor, CLIPModel
from typing import List
from pathlib import Path

app = FastAPI(title="Image Vectorization Service")

# --- 模型加载与初始化 ---
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model_name = "openai/clip-vit-base-patch32"

try:
    # 加载 CLIP 模型和处理器
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
    print(f"Model loaded successfully on device: {device}")
except Exception as e:
    print(f"Error loading model: {e}")

# --- 核心提取逻辑 (基于临时文件路径) ---

def get_vector_from_temp_file(file_path: str) -> List[float]:
    """
    核心方法：从磁盘临时文件路径读取并提取向量
    """
    try:
        # 使用 Pillow 打开图片并确保是 RGB 模式
        image = Image.open(file_path).convert("RGB")
        
        # 预处理
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # 计算 Embedding
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        # L2 规范化
        image_vector = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        return image_vector.squeeze().cpu().numpy().tolist()
    except Exception as e:
        raise ValueError(f"模型提取特征失败: {str(e)}")

# --- API 接口定义 ---
@app.post("/vectorize/image/file", summary="通过上传图片文件提取向量")
async def vectorize_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="文件必须是图片类型")

    tmp_path = None
    try:
        suffix = Path(file.filename).suffix if file.filename else ".tmp"
        
        # 1. 采用 delete=False，这样我们可以安全地在 close 后手动控制
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name  # 记录路径
        
        # 2. 走出 with 块，此时文件句柄已关闭，任何进程都能安全读取 tmp_path
        vector = get_vector_from_temp_file(tmp_path)
        
        return {
            "filename": file.filename,
            "vector_dimension": len(vector),
            "vector": vector
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"内部处理错误: {e}")
    finally:
        # 3. 无论成功失败，都在 finally 块中物理删除文件
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                print(f"警告：临时文件清理失败 {tmp_path}: {e}")

class TranscribeRequest(BaseModel):
    url: HttpUrl
@app.post("/vectorize/image/url")
async def vectorize_url(request_data: TranscribeRequest):
    tmp_path = None
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(str(request_data.url))
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="下载失败")

            # 准备临时文件
            suffix = Path(request_data.url.path or "").suffix or ".tmp"
            # 注意：delete=False 是关键，这样我们可以手动控制在文件关闭后再读取
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name 
            
            # 在 with 块外（即文件已关闭状态）调用推理
            vector = get_vector_from_temp_file(tmp_path)
            return {"url": str(request_data.url), "vector": vector, "vector_dimension": len(vector)}

    finally:
        # 无论成功还是报错，都在这里彻底清理
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

# --- 健康检查接口 ---

@app.get("/health", summary="服务健康状态检查")
def health_check():
    """
    返回当前模型信息和运行设备状态
    """
    return {
        "status": "active",
        "model": model_name,
        "device": str(device),
        "api_version": "1.1",
        "temp_file_support": True
    }