# --- 阶段 1: 基础环境和依赖安装 ---
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime AS builder


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*
# 1. 设置工作目录
WORKDIR /app

# 2. 拷贝依赖文件并安装 (使用 --no-cache-dir 减小镜像)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 如果在国内构建，可以取消下面这一行的注释来使用镜像站
# ENV HF_ENDPOINT=https://hf-mirror.com

# 复制下载脚本并运行
COPY download_models.py .
RUN python download_models.py

COPY ./app.py .
# 3. 暴露端口
EXPOSE 8000

# 4. 运行服务
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]