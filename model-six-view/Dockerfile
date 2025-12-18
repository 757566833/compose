FROM nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    xz-utils \
    libxi6 \
    libxrender1 \
    libsm6 \
    libxkbcommon0 \
    libxxf86vm1 \
    libxfixes3 \
    python3 \
    python3-pip \
    xorg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 根据架构选择正确的 Blender 版本
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        wget https://download.blender.org/release/Blender5.0/blender-5.0.1-linux-x64.tar.xz \
        && tar -xf blender-5.0.1-linux-x64.tar.xz --strip-components=1 -C /usr/local \
        && rm blender-5.0.1-linux-x64.tar.xz; \
    elif [ "$ARCH" = "aarch64" ]; then \
        wget https://download.blender.org/release/Blender5.0/blender-5.0.1-linux-aarch64.tar.xz \
        && tar -xf blender-5.0.1-linux-aarch64.tar.xz --strip-components=1 -C /usr/local \
        && rm blender-5.0.1-linux-aarch64.tar.xz; \
    else \
        echo "Unsupported architecture: $ARCH"; \
        exit 1; \
    fi

# 安装 Python 依赖
RUN pip3 install fastapi uvicorn python-multipart

# 复制应用代码
COPY app.py /app/

# 设置 Blender 路径
ENV PATH="/usr/local/bin:${PATH}"

# 设置 NVIDIA 环境变量
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,compute,utility

# 暴露 API 端口
EXPOSE 8000

# 启动命令
CMD ["blender", "--background", "--python", "app.py"]
