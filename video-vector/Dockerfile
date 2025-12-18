# 使用 python:3.10-slim 作为基础镜像
FROM python:3.10-slim

# 1. 安装 FFmpeg 和 OpenCV 的运行时依赖
# libsm6 和 libxext6 是 OpenCV 正常运行所必需的图形库依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    gcc \
    g++ \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*


# 2. 设置工作目录
WORKDIR /app

# 3. 拷贝依赖和应用代码
COPY requirements.txt .


# 4. 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 如果在国内构建，可以取消下面这一行的注释来使用镜像站
# ENV HF_ENDPOINT=https://hf-mirror.com

COPY download_models.py .
RUN python download_models.py

COPY ./app.py .
# 5. 暴露端口
EXPOSE 8000

# 6. 运行服务
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]