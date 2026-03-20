import os
# 禁用模型源检查，跳过网络连通性测试，加快启动速度
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import cv2
import numpy as np
import json
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Any, List, Dict, Tuple, Optional

from paddlex import create_pipeline
import logging
import re

app = FastAPI()

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

# 图片验证函数（使用cv2）
def validate_image_file(image_bytes: bytes) -> bool:
    """验证图片文件格式"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img is not None
    except Exception:
        return False

# ---------------- 初始化 ----------------

try:
    # 建议生产环境中使用具体路径或确保环境变量已配置
    ocr_model = create_pipeline(pipeline="OCR") 
    logger.info("PaddleX OCR Pipeline 初始化成功")
except Exception as e:
    logger.error(f"OCR 初始化失败: {e}")
    ocr_model = None

# ---------------- 模型与工具函数 ----------------

class TranscribeRequest(BaseModel):
    url: HttpUrl
    headers: Optional[Dict[str, str]] = None

def check_serializable(obj: Any) -> bool:
    """确保 OCR 结果可以转为 JSON"""
    try:
        json.dumps(obj)
        return True
    except:
        return False

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    大图预处理：限制长边在 2500px 以内
    """
    max_side = 2500
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

async def perform_ocr(image_bytes: bytes) -> Dict[str, Any]:
    """
    核心公共 OCR 逻辑：从字节流到识别结果
    """
    if ocr_model is None:
        raise HTTPException(status_code=503, detail="OCR 服务未就绪")

    # 1. 解码
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="图片解码失败，请检查文件格式")

    # 2. 预处理
    img = preprocess_image(img)

    # 3. 推理
    ocr_result = []
    try:
        results = ocr_model.predict(img)
        # 清洗结果，PaddleX 的结果项可能包含不可序列化的对象
        for item in results:
            clean_item = {k: v for k, v in item.items() if check_serializable(v)}
            ocr_result.append(clean_item)
    except Exception as e:
        logger.error(f"PaddleX 推理异常: {str(e)}")
        # 视业务需求决定是抛出异常还是返回空结果
        raise HTTPException(status_code=500, detail=f"推理引擎错误: {str(e)}")

    return {
        "ocr_result": ocr_result
    }

# ---------------- 接口实现 ----------------

@app.post("/ocr/file")
async def ocr_from_file(file: UploadFile = File(...)):  # 无文件大小限制
    """处理上传的文件"""
    logger.info(f"OCR file upload: {file.filename}, size: {file.size}, type: {file.content_type}")

    # 基础内容类型检查
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="请上传图片格式文件")

    contents = await file.read()

    # 图片验证
    if not validate_image_file(contents):
        logger.warning(f"Invalid image file: {file.filename}")
        raise HTTPException(status_code=400, detail="图片文件格式无效，请上传有效的图片文件")

    logger.info(f"File validated successfully: {file.filename}")
    return await perform_ocr(contents)

@app.post("/ocr/url")
async def ocr_from_url(request_data: TranscribeRequest):
    """处理图片 URL，支持自定义 headers 用于访问私有文件"""
    url = str(request_data.url)
    headers = request_data.headers or {}

    logger.info(f"OCR URL processing: {url}")

    try:
        # 使用 httpx 异步获取图片（无大小限制）
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)

            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"获取图片失败，HTTP 状态码: {response.status_code}"
                )

            # 读取内容
            content = await response.aread()

            content_type = response.headers.get("content-type", "")
            if "image" not in content_type:
                # 某些 URL 可能不带 content-type，这里根据实际情况决定是否强制拦截
                logger.warning(f"警告: 响应 Content-Type 为 {content_type}")

            # 验证图片格式
            if not validate_image_file(content):
                logger.warning(f"Invalid image file from URL: {url}")
                raise HTTPException(
                    status_code=400,
                    detail="URL返回的图片文件格式无效，请提供有效的图片URL"
                )

            logger.info(f"URL image validated successfully: {url}, size: {len(content)} bytes")
            return await perform_ocr(content)

    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="请求图片 URL 超时")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")
        raise HTTPException(status_code=500, detail=f"处理 URL 时发生意外错误: {str(e)}")

@app.get("/health", summary="服务健康状态检查")
def health_check():
    """
    返回当前模型信息和运行设备状态
    """
    return {
        "status": "active",
        "model": "PaddleX OCR",
    }