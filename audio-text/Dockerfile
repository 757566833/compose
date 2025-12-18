FROM python:3.10-slim

# 1. 安装 FFmpeg
# apt-get update & install -y ffmpeg 是安装 librosa 依赖的关键步骤
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    gcc \
    g++ \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# 2. 设置工作目录
WORKDIR /app

# 3. 拷贝依赖和应用代码
COPY requirements.txt .
# 4. 安装 Python 依赖
# 注意：这会安装 Whisper 模型和所有依赖
RUN pip install --no-cache-dir -r requirements.txt

# 如果在国内构建，可以取消下面这一行的注释来使用镜像站
# ENV HF_ENDPOINT=https://hf-mirror.com

# 复制下载脚本并运行
COPY download_models.py .
RUN python download_models.py

COPY ./app.py .
# 5. 暴露端口 (例如使用 8002)
EXPOSE 8000

# 6. 运行服务
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]