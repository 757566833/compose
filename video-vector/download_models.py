import torch
from transformers import pipeline, CLIPProcessor, CLIPModel, AutoModel

def download():
    print("Pre-downloading models...")
    # 下载 Whisper 模型
    pipeline("automatic-speech-recognition", model="openai/whisper-small")
    
    model_name = "jinaai/jina-clip-v2"
    print(f"正在下载图像向量化模型: {model_name}...")
    
    # 下载模型权重
    AutoModel.from_pretrained(model_name, trust_remote_code=True)
    
    print("图像模型下载完成！")

if __name__ == "__main__":
    download()