# --- 阶段 1: 基础环境和依赖安装 ---
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV HF_ENDPOINT=https://hf-mirror.com
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ARG DEBIAN_FRONTEND=noninteractive

RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*
# 1. 设置工作目录
WORKDIR /app

# 2. 拷贝依赖文件并安装 (使用 --no-cache-dir 减小镜像)
COPY requirements-locked.txt . 
RUN pip install --no-cache-dir --root-user-action=ignore --upgrade pip && \
    pip install --no-cache-dir --root-user-action=ignore -r requirements-locked.txt -i https://mirrors.aliyun.com/pypi/simple


# 2. 安装需要编译的库 (flash-attn)
RUN pip install --user --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 3. 安装 xformers
RUN pip install --user --no-cache-dir --root-user-action=ignore -U xformers --index-url https://download.pytorch.org/whl/cu124

# 如果在国内构建，可以取消下面这一行的注释来使用镜像站
ENV HF_ENDPOINT=https://hf-mirror.com

# 复制下载脚本并运行
COPY download_models.py .
RUN python download_models.py

COPY ./app.py .
# 3. 暴露端口
EXPOSE 8000

# 4. 运行服务
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]