FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV HF_ENDPOINT=https://hf-mirror.com
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ARG DEBIAN_FRONTEND=noninteractive

# 1. 安装 FFmpeg
# apt-get update & install -y ffmpeg 是安装 librosa 依赖的关键步骤
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    gcc \
    g++ \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# 2. 设置工作目录
WORKDIR /app

# 3. 拷贝依赖和应用代码
COPY requirements-locked.txt . 
# 4. 安装 Python 依赖
# 注意：这会安装 Whisper 模型和所有依赖
RUN pip install --no-cache-dir --root-user-action=ignore --upgrade pip && \
    pip install --no-cache-dir --root-user-action=ignore -r requirements-locked.txt -i https://mirrors.aliyun.com/pypi/simple
# 如果在国内构建，可以取消下面这一行的注释来使用镜像站
ENV HF_ENDPOINT=https://hf-mirror.com

# 复制下载脚本并运行
COPY download_models.py .
RUN python download_models.py

COPY ./app.py .
# 5. 暴露端口 (例如使用 8002)
EXPOSE 8000

# 6. 运行服务
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]