import fitz
import base64
import numpy as np
import cv2
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from paddleocr import PaddleOCR
import json

# 屏蔽模型来源检查（解决你的 SSL/LibreSSL 警告）
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

app = FastAPI(title="PDF OCR Service - PaddleX 3.x")

# --- PaddleX 3.x 版本的初始化方式 ---
# 在 3.x 版本中，大部分参数通过模型配置处理，初始化非常简单
# 它会自动检测你的 M4 CPU 或 GPU 运行环境
try:
    ocr = PaddleOCR(lang="ch") 
except Exception as e:
    print(f"初始化警告: {e}")
    # 极简模式兜底
    ocr = PaddleOCR()
def check_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except TypeError as e:
        return False
@app.post("/process")
async def process(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF allowed")
    
    try:
        content = await file.read()
        doc = fitz.open(stream=content, filetype="pdf")
        pages_data = []

        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            
            # 1. 尝试原生提取
            text = page.get_text("text").strip() or ""
            method = "native"

            # 2. 渲染图片用于展示和 OCR
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")

            # 3. 如果原生提取失败，启动新版 OCR
            if not text:
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                # PaddleX 3.x 的调用方式
                ocr_texts = []  # 初始化为空列表
                try:
                # 尝试直接赋值
                    ocr_texts = ocr.ocr(img)
                except Exception as e:
                    ocr_texts = []  # 如果发生异常，ocr_texts 保持为空列表，或者你可以设置其他默认值
                method = "paddlex_3.x_ocr"

                ocr_result = []
                for item in ocr_texts:
                    item_dict = {}
                    for key, value in item.items():
                        if  check_serializable(value):
                            item_dict[key] = value
                            ocr_result.append(item_dict)
                            
            pages_data.append({
                "page_number": page_index + 1,
                "method": method,
                "text": text,
                "ocr_result": ocr_result,
                "image": f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}"
            })

        doc.close()
        return {"filename": file.filename, "data": pages_data}

    except Exception as e:
        return {"error": str(e)}
