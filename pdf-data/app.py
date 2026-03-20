import os
# 禁用模型源检查
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import fitz
import base64
import numpy as np
import cv2
import httpx  # 用于下载 URL 文件
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, HttpUrl
import json
from typing import Any, List, Dict, Optional
from paddlex import create_pipeline
import logging
import re

app = FastAPI(title="PDF OCR Service - PaddleX 3.x")

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

# PDF验证函数
def validate_pdf_file(file_bytes: bytes) -> bool:
    """验证PDF文件格式"""
    try:
        # 检查PDF文件头
        return file_bytes.startswith(b"%PDF")
    except Exception:
        return False

# --- 模型初始化 ---
try:
    ocr_model = create_pipeline(pipeline="OCR")
    logger.info("PaddleX OCR Pipeline 初始化成功")
except Exception as e:
    logger.error(f"OCR 初始化失败: {e}")
    ocr_model = None

# --- 数据模型 ---
class TranscribeRequest(BaseModel):
    url: HttpUrl
    headers: Optional[Dict[str, str]] = None

# --- 公共工具函数 ---

def check_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False

def preprocess_image(img: np.ndarray) -> np.ndarray:
    max_side = 2500
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

async def perform_ocr(image_bytes: bytes) -> Dict[str, Any]:
    if ocr_model is None:
        raise HTTPException(status_code=503, detail="OCR 服务未就绪")

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("图片解码失败，返回空OCR结果")
        return {"ocr_result": []}

    img = preprocess_image(img)
    ocr_result = []
    try:
        # PaddleX 3.x predict 返回的是结果列表
        results = ocr_model.predict(img)
        for item in results:
            # 过滤不可序列化的对象（如自定义 Result 类）
            # 注意：PaddleX 3.x 结果通常通过 .json 或 .dict 获取更方便
            # 这里的逻辑保持你原有的 check_serializable 过滤
            clean_item = {k: str(v) if not check_serializable(v) else v for k, v in item.items()}
            ocr_result.append(clean_item)
    except Exception as e:
        logger.error(f"PaddleX 推理异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"推理引擎错误: {str(e)}")

    return {"ocr_result": ocr_result}

# --- 核心业务逻辑提取 ---

async def process_pdf_content(content: bytes, filename: str) -> Dict[str, Any]:
    """
    公共 PDF 处理逻辑：判断电子版/扫描版并执行相应处理
    """
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        pages_data = []

        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            
            # 1. 尝试原生文本提取
            text = page.get_text("text").strip() or ""
            method = "native"

            # 2. 准备图片渲染 (用于展示或扫描件 OCR)
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")

            # 3. 策略：若无文本则走 OCR，若有文本则跳过 OCR
            ocr_data = []
            if not text:
                res = await perform_ocr(img_bytes)
                ocr_data = res.get("ocr_result", [])
                method = "ocr"
                            
            pages_data.append({
                "page_number": page_index + 1,
                "method": method,
                "text": text,
                "ocr_result": ocr_data,
                "image": base64.b64encode(img_bytes).decode()
            })

        doc.close()
        return {"filename": filename, "data": pages_data, "format": "image/png"}

    except Exception as e:
        logger.error(f"PDF 处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF 处理失败: {str(e)}")

# --- API 接口 ---

@app.post("/pdf/file", summary="通过上传文件识别 PDF")
async def process_by_file(file: UploadFile = File(...)):
    """处理上传的PDF文件"""
    logger.info(f"PDF file upload: {file.filename}, size: {file.size}, type: {file.content_type}")

    # 基础内容类型检查
    if not (file.content_type and 'pdf' in file.content_type.lower()):
        raise HTTPException(status_code=400, detail="请上传PDF格式文件")

    content = await file.read()

    # PDF验证
    if not validate_pdf_file(content):
        logger.warning(f"Invalid PDF file: {file.filename}")
        raise HTTPException(status_code=400, detail="PDF文件格式无效，请上传有效的PDF文件")

    # 安全文件名处理
    safe_filename = get_safe_filename(file.filename)

    logger.info(f"File validated successfully: {file.filename}")
    return await process_pdf_content(content, safe_filename)

@app.post("/pdf/url", summary="通过 URL 链接识别 PDF")
async def process_by_url(request: TranscribeRequest):
    """处理PDF URL，支持自定义headers用于访问私有文件"""
    url = str(request.url)
    headers = request.headers or {}

    logger.info(f"PDF URL processing: {url}")

    try:
        # 使用httpx异步获取PDF（无大小限制）
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)

            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"获取PDF失败，HTTP状态码: {response.status_code}"
                )

            # 读取内容
            content = await response.aread()

            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type.lower():
                # 某些URL可能不带content-type，这里根据实际情况决定是否强制拦截
                logger.warning(f"警告: 响应Content-Type为 {content_type}")

            # 验证PDF格式
            if not validate_pdf_file(content):
                logger.warning(f"Invalid PDF file from URL: {url}")
                raise HTTPException(
                    status_code=400,
                    detail="URL返回的PDF文件格式无效，请提供有效的PDF URL"
                )

            # 安全处理文件名
            url_path = request.url.path or ""
            filename = get_safe_filename(url_path.split("/")[-1] if "/" in url_path else url_path) or "downloaded.pdf"

            logger.info(f"URL PDF validated successfully: {url}, size: {len(content)} bytes")
            return await process_pdf_content(content, filename)

    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="请求PDF URL超时")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")
        raise HTTPException(status_code=500, detail=f"处理URL时发生意外错误: {str(e)}")

@app.get("/health", summary="服务健康状态检查")
def health_check():
    """
    返回当前模型信息和运行状态
    """
    return {
        "status": "active" if ocr_model else "inactive",
        "model": "PaddleX OCR",
        "api_version": "1.0"
    }
