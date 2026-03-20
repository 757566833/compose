from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from PIL import Image
from pydantic import BaseModel, HttpUrl
import io
import torch
import httpx
import tempfile
import os
from transformers import AutoModel
from typing import List, Optional, Dict
from pathlib import Path
import shutil
import logging
import re
import base64

app = FastAPI(title="Image Vectorization Service")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 安全文件名函数
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

# 图片验证函数
def validate_image_file(file_path: str) -> bool:
    """验证图片文件格式"""
    try:
        with Image.open(file_path) as img:
            img.verify()  # 验证图片完整性
            return True
    except Exception:
        return False

MODEL_BASE = os.getenv("MODEL_PATH", "/models")
# --- 模型加载与初始化 ---
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model_name = "jinaai/jina-clip-v2"

# 根据设备动态设置批量处理限制（CPU 处理较慢，限制更严格）
MAX_BATCH_SIZE = 3 if device == "cpu" else 10
logger.info(f"Batch size limit set to {MAX_BATCH_SIZE} (device: {device})")

full_model_path = os.path.abspath(os.path.join(MODEL_BASE, model_name))

try:
    # 加载 CLIP 模型和处理器
    model = AutoModel.from_pretrained(
        full_model_path, 
        trust_remote_code=True,
    ).to(device)
    logger.info(f"Model loaded successfully on device: {device}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
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
async def vectorize_image(files: List[UploadFile] = File(...)):  # 无单个文件大小限制
    """通过上传图片文件提取向量"""
    from fastapi import status

    MAX_FILES = MAX_BATCH_SIZE

    logger.info(f"Received {len(files)} files for vectorization.")

    # 1. 检查文件数量限制
    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files. Maximum allowed is {MAX_FILES}."
        )

    # 2. 过滤并留下 image 类型的文件
    valid_images = [f for f in files if f.content_type and f.content_type.startswith('image/')]

    # 3. 如果没有 image 则报错
    if not valid_images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="未检测到有效的图片文件。"
        )

    temp_file_paths = []
    try:
        # 4. 处理每个文件（无大小限制）
        for file in valid_images:
            logger.info(f"Processing file: {file.filename}, size: {file.size}, type: {file.content_type}")

            # 安全文件名处理
            safe_filename = get_safe_filename(file.filename)
            suffix = Path(safe_filename).suffix if safe_filename != "file" else ".tmp"

            # 创建临时文件
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            try:
                # 写入内容
                shutil.copyfileobj(file.file, tmp_file)
                tmp_file.flush()
                temp_file_paths.append(tmp_file.name)
            finally:
                tmp_file.close()

            # 验证图片格式
            if not validate_image_file(tmp_file.name):
                logger.warning(f"Invalid image file: {file.filename}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File {file.filename} is not a valid image file."
                )

        logger.info(f"Temporary files created: {[os.path.basename(p) for p in temp_file_paths]}")

        # 5. 调用函数处理这些持久化后的文件路径
        results = get_vector_from_temp_file(temp_file_paths)
        logger.info(f"Vectorization completed successfully for {len(valid_images)} images.")
        return {
            "data": results.tolist(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"向量提取失败: {str(e)}")

    finally:
        # 6. 清理临时文件
        for path in temp_file_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as cleanup_error:
                    logger.error(f"清理临时文件失败: {path}, 错误: {cleanup_error}")

class TranscribeRequest(BaseModel):
    urls: List[str]
    headers: Optional[Dict[str, str]] = None


class Base64ImageRequest(BaseModel):
    """接收 base64 编码的图片数组"""
    images: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "images": [
                    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
                    "iVBORw0KGgoAAAANSUhEUgAA..."
                ]
            }
        }
@app.post("/vectorize/image/urls")
async def vectorize_url(request_data: TranscribeRequest):
    """
    接收图片 URL 列表，下载后提取向量。
    支持传入自定义 headers 用于访问私有文件。
    """
    from fastapi import status

    MAX_URLS = MAX_BATCH_SIZE

    urls = request_data.urls
    headers = request_data.headers or {}

    logger.info(f"Processing {len(urls)} image URLs for vectorization.")

    # 检查URL数量限制
    if len(urls) > MAX_URLS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many URLs. Maximum allowed is {MAX_URLS}."
        )

    temp_file_paths = []

    try:
        for idx, url in enumerate(urls):
            logger.info(f"Processing URL {idx+1}/{len(urls)}: {url}")

            # 安全处理文件名
            url_path = url.split("?")[0]  # 移除 query 参数
            filename = url_path.split("/")[-1] if "/" in url_path else url_path
            safe_filename = get_safe_filename(filename)
            suffix = Path(safe_filename).suffix if safe_filename != "file" else ".tmp"

            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = tmp.name

            # 下载文件（无大小限制）
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download file from URL: {url}, HTTP {response.status_code}"
                    )

                # 写入文件
                downloaded_size = 0
                with open(tmp_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)
                        downloaded_size += len(chunk)

            logger.info(f"Downloaded {downloaded_size} bytes from {url}")

            # 验证图片格式
            if not validate_image_file(tmp_path):
                logger.warning(f"Invalid image file from URL: {url}")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File from URL {url} is not a valid image file."
                )

            temp_file_paths.append(tmp_path)

        # 调用模型提取向量
        logger.info(f"All {len(urls)} images downloaded and validated.")
        vector = get_vector_from_temp_file(temp_file_paths)
        logger.info(f"Vectorization completed successfully for {len(urls)} images.")
        return {
            "data": vector.tolist(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing images from URLs: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")

    finally:
        # 清理临时文件
        for path in temp_file_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as cleanup_error:
                    logger.error(f"清理临时文件失败: {path}, 错误: {cleanup_error}")


@app.post("/vectorize/image/base64")
async def vectorize_base64_images(request_data: Base64ImageRequest):
    """
    接收 base64 编码的图片数组，提取向量。
    支持纯 base64 字符串或 dataURI 格式 (data:image/xxx;base64,xxx)。
    """
    from fastapi import status

    MAX_IMAGES = MAX_BATCH_SIZE
    
    images = request_data.images
    logger.info(f"Processing {len(images)} base64 images for vectorization.")

    # 检查数量限制
    if len(images) > MAX_IMAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many images. Maximum allowed is {MAX_IMAGES}."
        )

    temp_file_paths = []

    try:
        for idx, img_base64 in enumerate(images):
            logger.info(f"Processing base64 image {idx+1}/{len(images)}")
            
            # 处理 dataURI 格式: data:image/xxx;base64,xxx
            if img_base64.startswith("data:"):
                if ";base64," in img_base64:
                    img_base64 = img_base64.split(";base64,")[1]
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid dataURI format at index {idx}. Expected 'data:image/xxx;base64,xxx'"
                    )
            
            # base64 解码
            try:
                img_data = base64.b64decode(img_base64)
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid base64 encoding at index {idx}."
                )
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
                tmp_path = tmp.name
            
            # 写入文件
            with open(tmp_path, "wb") as f:
                f.write(img_data)
            
            # 验证图片格式
            if not validate_image_file(tmp_path):
                logger.warning(f"Invalid image file from base64 at index {idx}")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Data at index {idx} is not a valid image file."
                )
            
            temp_file_paths.append(tmp_path)

        # 调用模型提取向量
        logger.info(f"All {len(images)} base64 images decoded and validated.")
        vectors = get_vector_from_temp_file(temp_file_paths)
        logger.info(f"Vectorization completed successfully for {len(images)} images.")
        return {
            "data": vectors.tolist(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing base64 images: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")

    finally:
        # 清理临时文件
        for path in temp_file_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as cleanup_error:
                    logger.error(f"清理临时文件失败: {path}, 错误: {cleanup_error}")


class TextData(BaseModel):
    """定义接口接收的数据结构"""
    texts: list[str] # 接收一个字符串列表，以便批量处理
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
        "temp_file_support": True
    }