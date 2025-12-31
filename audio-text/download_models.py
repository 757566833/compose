import os
from huggingface_hub import snapshot_download

# --- 配置部分 ---
local_root_dir = "./models"
asr_local_dir = os.path.join(local_root_dir, "SenseVoiceSmall")
vad_local_dir = os.path.join(local_root_dir, "fsmn-vad")

# --- 修正后的 Hugging Face ID ---
asr_model_id = "FunAudioLLM/SenseVoiceSmall"
# 注意：VAD 模型在 HF 上的路径通常是这个
vad_model_id = "funasr/FSMN-VAD" 

def download_hf_model(model_id, target_dir):
    print(f"正在从 Hugging Face 下载: {model_id} ...")
    try:
        path = snapshot_download(
            repo_id=model_id, 
            local_dir=target_dir,
            # 加上这个可以跳过登录检查，直接下载公开模型
            token=False 
        )
        print(f"下载成功！存放在: {path}")
        return path
    except Exception as e:
        print(f"下载 {model_id} 失败: {e}")
        return None

if __name__ == "__main__":
    if not os.path.exists(local_root_dir):
        os.makedirs(local_root_dir)

    # 执行下载
    asr_path = download_hf_model(asr_model_id, asr_local_dir)
    vad_path = download_hf_model(vad_model_id, vad_local_dir)

    print("-" * 30)
    if asr_path and vad_path:
        print("所有模型已就绪！")
    else:
        print("部分模型下载失败，请检查 ID 是否正确。")