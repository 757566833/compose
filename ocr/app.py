import base64
import numpy as np
import cv2
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from paddleocr import PaddleOCR
import json

# 屏蔽模型来源检查
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

app = FastAPI(title="Image OCR Service - PaddleX 3.x")

# 初始化 Magic 实例
# mime=True 表示返回 'image/jpeg' 这种格式
mime_detector = magic.Magic(mime=True)

ALLOWED_MIMES = {
    "image/jpeg",
    "image/png", 
    "image/webp",
    "image/bmp",
    "image/tiff",
    "image/x-portable-graymap",  # PGM
    "image/x-portable-pixmap",   # PPM
}
# 初始化 PaddleOCR (3.x 兼容版)
# use_angle_cls=True 可以自动旋转校正方向错误的图片
try:
    ocr = PaddleOCR(lang="ch") 
except Exception as e:
    print(f"初始化警告: {e}")
    ocr = PaddleOCR()

def check_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False

@app.post("/ocr/file")
async def process_image(file: UploadFile = File(...)):
    try:
        # 1. 读取前 2048 字节用于类型检测
        initial_content = await file.read(2048)
        
        # 2. 使用 python-magic 检测真实类型
        detected_mime = mime_detector.from_buffer(initial_content)
        
        if detected_mime not in ALLOWED_MIMES:
            raise HTTPException(
                status_code=400, 
                detail=f"非法文件格式。检测到: {detected_mime}，仅支持常见的图片格式。"
            )
        
        # 3. 回到文件开头读取完整内容
        await file.seek(0)
        full_content = await file.read()
        
        # 4. 转换为 OpenCV 格式进行 OCR
        img_array = np.frombuffer(full_content, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="图片解码失败")

        # --- 这里接入你的 PaddleOCR 逻辑 ---
        # result = ocr.ocr(img)
        
        return {
            "filename": file.filename,
            "real_mime": detected_mime,
            "status": "verified & processed"
        }

    except Exception as e:
        return {"error": str(e)}