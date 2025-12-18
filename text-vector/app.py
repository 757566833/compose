from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch

app = FastAPI(title="Text Vectorization Service")

# --- 模型加载与初始化 ---
# 文本模型通常使用 CPU 运行，因为它对 GPU 的需求不如图像模型高
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model_name = "all-MiniLM-L6-v2"

try:
    # 加载 SBERT 模型
    model = SentenceTransformer(model_name, device=device)
    print(f"Text Model loaded successfully on device: {device}")
except Exception as e:
    print(f"Error loading text model: {e}")
    # 生产环境中应处理模型加载失败的情况

# --- 定义输入数据结构 ---
class TextData(BaseModel):
    """定义接口接收的数据结构"""
    texts: list[str] # 接收一个字符串列表，以便批量处理

# --- API 接口定义 ---

@app.post("/vectorize/text")
def vectorize_text(data: TextData):
    """
    接收一个文本列表，返回其对应的向量（Embeddings）
    """
    if not data.texts:
        return {"vectors": []}

    try:
        # 1. 批量计算向量
        # convert_to_tensor=True 确保输出是 PyTorch Tensor
        embeddings = model.encode(data.texts, convert_to_tensor=False)
        
        # 2. 转换为标准的 Python list
        # embeddings 是 NumPy 数组，直接调用 tolist()
        vectors_list = embeddings.tolist()

        return {
            "num_texts": len(vectors_list),
            "vector_dimension": len(vectors_list[0]) if vectors_list else 0,
            "vectors": vectors_list
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal processing error: {e}")

# 健康检查接口
@app.get("/health")
def health_check():
    return {"status": "ok", "model": model_name, "device": str(device)}