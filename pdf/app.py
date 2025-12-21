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
from paddleocr import PaddleOCR
import json
from typing import Any, List, Dict, Optional
from paddlex import create_pipeline

app = FastAPI(title="PDF OCR Service - PaddleX 3.x")

# --- 模型初始化 ---
try:
    ocr_model = create_pipeline(pipeline="OCR") 
    print("PaddleX OCR Pipeline 初始化成功")
except Exception as e:
    print(f"OCR 初始化失败: {e}")
    ocr_model = None

# --- 数据模型 ---
class TranscribeRequest(BaseModel):
    url: HttpUrl

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
        print(f"PaddleX 推理异常: {str(e)}")
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
        raise HTTPException(status_code=500, detail=f"PDF 处理失败: {str(e)}")

# --- API 接口 ---

@app.post("/pdf/file", summary="通过上传文件识别 PDF")
async def process_by_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")
    
    content = await file.read()
    return await process_pdf_content(content, file.filename)

@app.post("/pdf/url", summary="通过 URL 链接识别 PDF")
async def process_by_url(request: TranscribeRequest):
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(str(request.url))
            response.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无法从 URL 下载文件: {str(e)}")
        
        content = response.content
        # 简单校验是否为 PDF
        if not content.startswith(b"%PDF"):
             raise HTTPException(status_code=400, detail="该 URL 指向的内容不是有效的 PDF")

        filename = os.path.basename(str(request.url.path)) or "downloaded.pdf"
        return await process_pdf_content(content, filename)
