import torch
from transformers import pipeline, CLIPProcessor, CLIPModel

def download():
    print("Pre-downloading models...")
    # 下载 Whisper 模型
    pipeline("automatic-speech-recognition", model="openai/whisper-small")
    
    # 下载 CLIP 模型
    CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("All models downloaded successfully!")

if __name__ == "__main__":
    download()