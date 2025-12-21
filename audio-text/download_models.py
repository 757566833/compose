import torch
from transformers import pipeline, CLIPProcessor, CLIPModel

def download():
    print("Pre-downloading models...")
    # 下载 Whisper 模型
    pipeline("automatic-speech-recognition", model="openai/whisper-small")

if __name__ == "__main__":
    download()