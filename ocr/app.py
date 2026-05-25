import os
# 禁用模型源检查，跳过网络连通性测试，加快启动速度
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import cv2
import numpy as np
import json
import httpx
import tempfile
import base64
import binascii
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, HttpUrl, model_validator
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

# PDF 识别（通过文件头魔数）
def is_pdf(data: bytes) -> bool:
    return len(data) >= 4 and data[:4] == b'%PDF'

# 图片验证函数（使用cv2）
def validate_image_file(image_bytes: bytes) -> bool:
    """验证图片文件格式"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img is not None
    except Exception:
        return False

def validate_ocr_file(data: bytes) -> bool:
    """验证文件是否为支持的 OCR 输入（PDF 或 cv2 可解码的图片）"""
    return is_pdf(data) or validate_image_file(data)

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
    url: Optional[HttpUrl] = None
    base64: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

    @model_validator(mode="after")
    def _check_url_or_base64(self) -> "TranscribeRequest":
        if not self.url and not self.base64:
            raise ValueError("必须提供 url 或 base64 其中之一")
        return self


def decode_base64_input(b64: str) -> bytes:
    """
    解码 base64 输入，兼容两种格式：
    - 纯 base64 字符串
    - 带 data URI 前缀（如 data:image/png;base64,xxx）
    """
    if not b64 or not isinstance(b64, str):
        raise HTTPException(status_code=400, detail="base64 内容为空")

    s = b64.strip()
    # 去除 data URI 前缀
    if s.startswith("data:"):
        comma_idx = s.find(",")
        if comma_idx == -1:
            raise HTTPException(status_code=400, detail="无效的 data URI 格式")
        s = s[comma_idx + 1:]

    # 移除可能的空白字符（换行、空格等）
    s = re.sub(r"\s+", "", s)

    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"base64 解码失败: {str(e)}")

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

def _collect_ocr_results(results) -> List[Dict[str, Any]]:
    ocr_result = []
    for item in results:
        clean_item = {k: v for k, v in item.items() if check_serializable(v)}
        ocr_result.append(clean_item)
    return ocr_result

async def perform_ocr(file_bytes: bytes) -> Dict[str, Any]:
    """
    核心公共 OCR 逻辑：支持图片（JPEG/PNG/BMP/TIFF/WebP 等 cv2 支持的格式）与 PDF
    """
    if ocr_model is None:
        raise HTTPException(status_code=503, detail="OCR 服务未就绪")

    try:
        if is_pdf(file_bytes):
            # PDF 交由 PaddleX 直接按页处理（内部使用 pypdfium2 渲染）
            with tempfile.TemporaryDirectory() as tmpdir:
                pdf_path = os.path.join(tmpdir, "input.pdf")
                with open(pdf_path, "wb") as f:
                    f.write(file_bytes)
                results = ocr_model.predict(pdf_path)
                ocr_result = _collect_ocr_results(results)
        else:
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="文件解码失败，请上传有效的图片或 PDF 文件")
            img = preprocess_image(img)
            results = ocr_model.predict(img)
            ocr_result = _collect_ocr_results(results)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PaddleX 推理异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"推理引擎错误: {str(e)}")

    return {
        "ocr_result": ocr_result
    }

# ---------------- 接口实现 ----------------

@app.post("/ocr/file")
async def ocr_from_file(file: UploadFile = File(...)):  # 无文件大小限制
    """处理上传的文件，支持图片（JPEG/PNG/BMP/TIFF/WebP 等）与 PDF"""
    logger.info(f"OCR file upload: {file.filename}, size: {file.size}, type: {file.content_type}")

    contents = await file.read()

    # 基于文件内容验证（不依赖 content-type，兼容 image/* 与 application/pdf）
    if not validate_ocr_file(contents):
        logger.warning(f"Unsupported or invalid file: {file.filename}")
        raise HTTPException(status_code=400, detail="文件格式无效，请上传有效的图片或 PDF 文件")

    logger.info(f"File validated successfully: {file.filename}")
    return await perform_ocr(contents)

@app.post("/ocr/url")
async def ocr_from_url(request_data: TranscribeRequest):
    """处理图片，支持 URL 或 base64（纯 base64 或带 data URI 前缀）输入"""
    # 优先处理 base64
    if request_data.base64:
        logger.info(f"OCR base64 processing: length={len(request_data.base64)}")
        content = decode_base64_input(request_data.base64)

        if not validate_ocr_file(content):
            logger.warning("Invalid file from base64 input")
            raise HTTPException(
                status_code=400,
                detail="base64 解码后的文件格式无效，请提供有效的图片或 PDF"
            )

        logger.info(f"Base64 file validated successfully, size: {len(content)} bytes")
        return await perform_ocr(content)

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
            if "image" not in content_type and "pdf" not in content_type:
                # 某些 URL 可能不带 content-type，仅记录警告，最终以内容验证为准
                logger.warning(f"警告: 响应 Content-Type 为 {content_type}")

            # 基于文件内容验证（支持图片与 PDF）
            if not validate_ocr_file(content):
                logger.warning(f"Invalid file from URL: {url}")
                raise HTTPException(
                    status_code=400,
                    detail="URL 返回的文件格式无效，请提供有效的图片或 PDF URL"
                )

            logger.info(f"URL file validated successfully: {url}, size: {len(content)} bytes")
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