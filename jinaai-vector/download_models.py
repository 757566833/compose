import os
from huggingface_hub import snapshot_download

# --- 配置部分 ---
local_root_dir = "./models"
clip_model_name = "jinaai/jina-clip-v2"
embeddings_model_name = "jinaai/jina-embeddings-v3"

model_local_dir = os.path.join(local_root_dir, clip_model_name)

# --- 修正后的 Hugging Face ID ---



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
    clip_model_path = download_hf_model(clip_model_name, model_local_dir)
    embeddings_path = download_hf_model(embeddings_model_name, os.path.join(local_root_dir, embeddings_model_name))

    print("-" * 30)
    if clip_model_path and embeddings_path:
        print("所有模型已就绪！")
    else:
        print("模型下载失败，请检查 ID 是否正确。")