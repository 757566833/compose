from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel
import base64

app = FastAPI(title="Image Vectorization Service")

# --- 模型加载与初始化 ---
# 优先使用 CUDA 或 Mac M4 的 MPS 加速，否则使用 CPU
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model_name = "openai/clip-vit-base-patch32"

try:
    # 首次运行会自动下载模型权重
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    print(f"Model loaded successfully on device: {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    # 生产环境中应处理模型加载失败的情况

# --- API 接口定义 ---

@app.post("/vectorize/image")
async def vectorize_image(file: UploadFile = File(...)):
    """
    接收一张图片文件，返回其 512 维的向量（Embedding）
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # 1. 读取图片文件
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        
        # 2. 预处理
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # 3. 计算向量
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        # 4. 规范化并转换为 list
        # 向量通常需要 L2 规范化
        image_vector = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        vector_list = image_vector.squeeze().cpu().numpy().tolist()

        return {
            "filename": file.filename,
            "vector_dimension": len(vector_list),
            # 为了减少响应体积，通常会对向量进行压缩或只返回重要的部分，
            # 这里返回完整的向量列表
            "vector": vector_list
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal processing error: {e}")

# (可选) 增加一个健康检查接口
@app.get("/health")
def health_check():
    return {"status": "ok", "model": model_name, "device": str(device)}