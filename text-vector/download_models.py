from sentence_transformers import SentenceTransformer
import os

def download():
    model_name = "jinaai/jina-clip-v2"
    print(f"正在下载文本向量化模型: {model_name}...")
    
    # 显式指定加载模型，这会触发下载到默认缓存路径 (~/.cache/torch/sentence_transformers)
    model = SentenceTransformer(model_name, device=device ,trust_remote_code=True, truncate_dim=truncate_dim)
    
    print("文本模型下载完成！")

if __name__ == "__main__":
    download()