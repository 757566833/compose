import os
# 禁用模型源检查，跳过网络连通性测试，加快启动速度
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import cv2
import numpy as np
import json
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Any, List, Dict, Tuple

from paddlex import create_pipeline

app = FastAPI()

# ---------------- 初始化 ----------------

try:
    # 建议生产环境中使用具体路径或确保环境变量已配置
    ocr_model = create_pipeline(pipeline="OCR") 
    print("PaddleX OCR Pipeline 初始化成功")
except Exception as e:
    print(f"OCR 初始化失败: {e}")
    ocr_model = None

# ---------------- 模型与工具函数 ----------------

class TranscribeRequest(BaseModel):
    url: HttpUrl

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
        print(f"PaddleX 推理异常: {str(e)}")
        # 视业务需求决定是抛出异常还是返回空结果
        raise HTTPException(status_code=500, detail=f"推理引擎错误: {str(e)}")

    # 4. 文本聚合
    text = "\n".join([str(res.get("text", "")) for res in ocr_result if res.get("text")])

    return {
        "text": text,
        "ocr_result": ocr_result
    }

# ---------------- 接口实现 ----------------

@app.post("/ocr/file")
async def ocr_from_file(file: UploadFile = File(...)):
    """处理上传的文件"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="请上传图片格式文件")
    
    contents = await file.read()
    return await perform_ocr(contents)

@app.post("/ocr/url")
async def ocr_from_url(request_data: TranscribeRequest):
    """处理图片 URL"""
    url = str(request_data.url)
    
    try:
        # 使用 httpx 异步获取图片
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(url)
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400, 
                    detail=f"获取图片失败，HTTP 状态码: {response.status_code}"
                )
            
            content_type = response.headers.get("content-type", "")
            if "image" not in content_type:
                # 某些 URL 可能不带 content-type，这里根据实际情况决定是否强制拦截
                print(f"警告: 响应 Content-Type 为 {content_type}")

            return await perform_ocr(response.content)

    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="请求图片 URL 超时")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理 URL 时发生意外错误: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)