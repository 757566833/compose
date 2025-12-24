import os
import sys
import shutil

# ==========================================
# 1. 核心防御：拦截所有 GPU 插件和环境
# ==========================================
sys.modules["flash_attn"] = None
sys.modules["xformers"] = None
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
from transformers import AutoModel, AutoTokenizer

def download():
    model_name = "jinaai/jina-clip-v2"
    target_dir = os.path.abspath("./jina-clip-v2-offline")
    
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    print(f"正在开始下载...")

    try:
        # 2. 移除 device_map，改用 torch.device 强制 CPU
        # 这样就不需要安装 accelerate 库了
        device = torch.device("cpu")
        
        # 3. 使用 dtype 代替过时的 torch_dtype
        model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            # 不使用 device_map，避免 accelerate 依赖
            torch_dtype=torch.float32, 
            low_cpu_mem_usage=True
        ).to(device) # 手动移动到 CPU
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # 4. 保存到本地
        model.save_pretrained(target_dir)
        tokenizer.save_pretrained(target_dir)
        
        print("-" * 30)
        print(f"成功！完整离线包已导出至: {target_dir}")
        print("-" * 30)
        
    except Exception as e:
        print(f"下载失败: {e}")

if __name__ == "__main__":
    download()