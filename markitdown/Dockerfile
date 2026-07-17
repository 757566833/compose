FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# curl: 供编排层 healthcheck 使用
# ffmpeg: 音频格式转换 (mp3/m4a -> wav, pydub 依赖)
# libimage-exiftool-perl: 图片/音频元数据提取 (exiftool)
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    ffmpeg \
    libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --root-user-action=ignore --upgrade pip -i https://mirrors.aliyun.com/pypi/simple && \
    pip install --no-cache-dir --root-user-action=ignore 'markitdown[all]' markitdown-mcp \
    fastapi uvicorn python-multipart -i https://mirrors.aliyun.com/pypi/simple

COPY app.py /app/app.py

WORKDIR /workdir

# 容器内监听必须绑 0.0.0.0（宿主访问走端口映射）
CMD ["markitdown-mcp", "--http", "--host", "0.0.0.0", "--port", "3001"]
