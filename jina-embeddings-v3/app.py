from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import os
from transformers import AutoModel
from typing import List
import logging

app = FastAPI(title="Text Embeddings Service")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_BASE = os.getenv("MODEL_PATH", "/models")
# --- 模型加载与初始化 ---
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model_name = "jinaai/jina-embeddings-v3"

# 根据设备动态设置批量处理限制（CPU 处理较慢，限制更严格）
MAX_BATCH_SIZE = 3 if device == "cpu" else 10
logger.info(f"Batch size limit set to {MAX_BATCH_SIZE} (device: {device})")

full_model_path = os.path.abspath(os.path.join(MODEL_BASE, model_name))

try:
    # 加载 Embeddings 模型
    model = AutoModel.from_pretrained(
        full_model_path,
        trust_remote_code=True,
    ).to(device)
    logger.info(f"Model loaded successfully on device: {device}")
except Exception as e:
    logger.error(f"Error loading model: {e}")

class TextData(BaseModel):
    """定义接口接收的数据结构"""
    texts: List[str] # 接收一个字符串列表，以便批量处理

@app.post("/vectorize/text")
def vectorize_text(data: TextData):
    """
    接收一个文本列表，返回其对应的向量（Embeddings）
    """
    logger.info(f"Vectorizing {len(data.texts)} text(s)")

    if not data.texts:
        logger.info("Empty text list received")
        return {"vectors": []}

    try:
        # 1. 批量计算向量
        text_embeddings = model.encode(data.texts, normalize_embeddings=True)

        # 2. 转换为标准的 Python list
        vectors_list = text_embeddings.tolist()

        logger.info(f"Text vectorization completed successfully for {len(data.texts)} text(s)")
        return {
            "num_texts": len(vectors_list),
            "vectors": vectors_list
        }

    except Exception as e:
        logger.error(f"Text vectorization error: {e}")
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
        "temp_file_support": False
    }