import torch
from transformers import AutoModel
from funasr import AutoModel as FunASRAutoModel

def download():
    print("Pre-downloading models...")
    # 下载 Whisper 模型
    audio_model = FunASRAutoModel(
        model=model_dir,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
        hub="hf",
    )
    
    model_name = "jinaai/jina-clip-v2"
    print(f"正在下载图像向量化模型: {model_name}...")
    
    # 下载模型权重
    AutoModel.from_pretrained(model_name, trust_remote_code=True)
    
    print("图像模型下载完成！")

if __name__ == "__main__":
    download()