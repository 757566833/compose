FROM python:3.10-slim

# 设置环境变量
ENV HF_ENDPOINT=https://hf-mirror.com
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ARG DEBIAN_FRONTEND=noninteractive

# 设置工作目录
WORKDIR /app

# 1. 安装系统依赖
# libreoffice-impress: 处理 PPT 转换
# libmagic1: python-magic 的底层库
# fonts-wqy-zenhei: 关键！中文字体，防止 PPT 转 PPTX 时文字乱码或消失
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    libreoffice-impress \
    libreoffice-java-common \
    libmagic1 \
    fonts-wqy-zenhei \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. 安装 Python 依赖
# 先复制 requirements.txt 以利用 Docker 缓存层
COPY requirements-locked.txt . 
RUN pip install --no-cache-dir --root-user-action=ignore --upgrade pip && \
    pip install --no-cache-dir --root-user-action=ignore -r requirements-locked.txt -i https://mirrors.aliyun.com/pypi/simple

# 3. 复制项目代码
COPY . .

# 4. 暴露端口
EXPOSE 8000

# 5. 启动命令
# 使用 uvicorn 启动，增加 --proxy-headers 保证在 K8S/Nginx 后获取真实 IP
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]