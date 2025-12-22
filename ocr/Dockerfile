FROM python:3.10-slim

ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
ENV HF_ENDPOINT=https://hf-mirror.com
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 安装系统依赖
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-locked.txt . 

# 升级 pip 并安装依赖
RUN pip install --no-cache-dir --root-user-action=ignore --upgrade pip && \
    pip install --no-cache-dir --root-user-action=ignore -r requirements-locked.txt  -i https://mirrors.aliyun.com/pypi/simple

COPY app.py .

# --- 修改后的预下载指令 ---
# 1. 移除 use_gpu 参数
# 2. 设置环境变量屏蔽联网检查警告
RUN export DISABLE_MODEL_SOURCE_CHECK=True && \
    python3 -c "from paddleocr import PaddleOCR; PaddleOCR(lang='ch')"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]