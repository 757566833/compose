# 使用 24.04 基础镜像
FROM nvidia/cuda:13.1.0-runtime-ubuntu24.04

ENV HF_ENDPOINT=https://hf-mirror.com
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ARG DEBIAN_FRONTEND=noninteractive


ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive
ENV RUNNING_IN_DOCKER=true

# 1. 安装系统依赖 (24.04 的包名基本一致)
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    wget \
    xz-utils \
    libgl1 \
    libgomp1 \
    libx11-6 \
    libxi6 \
    libxrender1 \
    libxkbcommon0 \
    libsm6 \
    libdbus-1-3 \
    libxfixes3 \
    ca-certificates \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/blender

# 2. 分架构处理
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        # AMD64: 直接上最新的 4.0.2
        wget https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz && \
        tar -xf blender-4.0.2-linux-x64.tar.xz --strip-components=1 -C /opt/blender && \
        rm blender-4.0.2-linux-x64.tar.xz; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
        # ARM64: 使用 Ubuntu 24.04 仓库的 4.0.x 版本
        apt-get update && apt-get install -y blender=4.0.2+dfsg-1ubuntu8 && \
        # 建立软链接保持路径一致
        ln -s /usr/bin/blender /opt/blender/blender; \
    fi

COPY requirements-locked.txt ./
# 3. 动态安装 Python 依赖
# 24.04 默认 Python 3.12，Blender 5.x 内部也是 3.11/3.12 左右，兼容性很好
RUN B_PYTHON=$(find /opt/blender -name "python3*" -type f | grep bin/python || which python3) && \
    $B_PYTHON -m pip install --root-user-action=ignore --upgrade pip --break-system-packages || true && \
    $B_PYTHON -m pip install --root-user-action=ignore --no-cache-dir \
    -i https://mirrors.aliyun.com/pypi/simple \
    -r requirements-locked.txt --break-system-packages

# 4. 环境配置
ENV PATH="/opt/blender:${PATH}"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility


WORKDIR /app
COPY app.py /app/

EXPOSE 8000

# 启动
CMD ["blender", "--background", "--python", "/app/app.py"]