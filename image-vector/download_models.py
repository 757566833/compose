import torch
from transformers import CLIPProcessor, CLIPModel

def download():
    model_name = "openai/clip-vit-base-patch32"
    print(f"正在下载图像向量化模型: {model_name}...")
    
    # 下载模型权重
    CLIPModel.from_pretrained(model_name)
    # 下载预处理器配置
    CLIPProcessor.from_pretrained(model_name)
    
    print("图像模型下载完成！")

if __name__ == "__main__":
    download()