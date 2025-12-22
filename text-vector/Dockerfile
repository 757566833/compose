# 使用单阶段构建简化部署
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# 设置环境变量
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

# 2. 拷贝依赖和应用代码
COPY requirements-locked.txt . 

# 3. 安装依赖
# 注意：第一次运行时，SBERT 会自动下载模型权重
RUN pip install --no-cache-dir --root-user-action=ignore --upgrade pip && \
    pip install --no-cache-dir --root-user-action=ignore -r requirements-locked.txt -i https://mirrors.aliyun.com/pypi/simple

# 如果在国内构建，可以取消下面这一行的注释来使用镜像站
# ENV HF_ENDPOINT=https://hf-mirror.com
# 复制下载脚本并运行
COPY download_models.py .
RUN python download_models.py

COPY ./app.py .
# 4. 暴露端口 (例如使用 8001)
EXPOSE 8000

# 5. 运行服务
# 确保使用 0.0.0.0 和指定的端口
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]