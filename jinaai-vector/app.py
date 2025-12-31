from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from PIL import Image
from pydantic import BaseModel, HttpUrl
import io
import torch
import httpx
import tempfile
import os
from transformers import AutoModel
from typing import List
from pathlib import Path
import shutil

app = FastAPI(title="Image Vectorization Service")

MODEL_BASE = os.getenv("MODEL_PATH", "/models")
# --- 模型加载与初始化 ---
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model_name = "jinaai/jina-clip-v2"

full_model_path = os.path.abspath(os.path.join(MODEL_BASE, model_name))

try:
    # 加载 CLIP 模型和处理器
    model = AutoModel.from_pretrained(
        full_model_path, 
        trust_remote_code=True,
    ).to(device)
    print(f"Model loaded successfully on device: {device}")
except Exception as e:
    print(f"Error loading model: {e}")
truncate_dim = 512
# --- 核心提取逻辑 (基于临时文件路径) ---

def get_vector_from_temp_file(image_urls: List[str]) -> List[List[float]]:
    """
    核心方法：从磁盘临时文件路径读取并提取向量
    """
    try:
        image_embeddings = model.encode_image(
            image_urls, truncate_dim=truncate_dim
        )  # also accepts PIL.Image.Image, local filenames, dataURI

        
        return image_embeddings
    except Exception as e:
        raise ValueError(f"模型提取特征失败: {str(e)}")

# --- API 接口定义 ---
@app.post("/vectorize/image/files", summary="通过上传图片文件提取向量")
async def vectorize_image(files: List[UploadFile] = File(...)):
    print(f"Received {len(files)} files for vectorization.")
    # 1. 过滤并留下 image 类型的文件
    valid_images = [f for f in files if f.content_type and f.content_type.startswith('image/')]

    # 2. 如果没有 image 则报错
    if not valid_images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="未检测到有效的图片文件。"
        )

    temp_file_paths = []
    print(f"2")
    try:
        # 3. 组成数组并持久化到临时文件
        for file in valid_images:
            print(f"Processing file: {file.filename}, content_type: {file.content_type}")
            # 使用你提供的后缀获取逻辑
            suffix = Path(file.filename).suffix if file.filename else ".tmp"
            
            # 创建临时文件
            # delete=False 是关键，否则文件在 close 时会被自动删除，导致后面函数读不到
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            try:
                # 写入内容
                shutil.copyfileobj(file.file, tmp_file)
                temp_file_paths.append(tmp_file.name)
            finally:
                tmp_file.close()

        # 4. 调用函数处理这些持久化后的文件路径
        print(f"Temporary files created: {temp_file_paths}")
        results = get_vector_from_temp_file(temp_file_paths)
        print(f"Vectorization completed successfully.")
        return {
            "data": results.tolist(),
        }

    except Exception as e:
        # 异常处理
        raise HTTPException(status_code=500, detail=f"向量提取失败: {str(e)}")

    finally:
        # 5. 最后结果出来了，把文件都删了
        for path in temp_file_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as cleanup_error:
                    print(f"清理临时文件失败: {path}, 错误: {cleanup_error}")

class TranscribeRequest(BaseModel):
    urls: List[str]
@app.post("/vectorize/image/urls")
async def vectorize_url(request_data: TranscribeRequest):
    vector = get_vector_from_temp_file(request_data.urls)
    return {
        "data": vector.tolist(),
    }


class TextData(BaseModel):
    """定义接口接收的数据结构"""
    texts: list[str] # 接收一个字符串列表，以便批量处理
@app.post("/vectorize/text")
def vectorize_text(data: TextData):
    """
    接收一个文本列表，返回其对应的向量（Embeddings）
    """
    if not data.texts:
        return {"vectors": []}

    try:
        # 1. 批量计算向量
        # convert_to_tensor=True 确保输出是 PyTorch Tensor
        text_embeddings = model.encode(data.texts, normalize_embeddings=True)
        
        # 2. 转换为标准的 Python list
        # embeddings 是 NumPy 数组，直接调用 tolist()
        vectors_list = text_embeddings.tolist()

        return {
            "num_texts": len(vectors_list),
            "vectors": vectors_list
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal processing error: {e}")
    
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