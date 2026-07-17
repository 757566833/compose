CURRENT_TIME=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
echo "time is: $CURRENT_TIME"
mkdir -p docker.log

# 多架构构建（本机 arm / 生产 amd64），buildx 多平台无法 --load，需 --push 直推
docker buildx build --platform linux/amd64,linux/arm64 \
  -t harbor.fzcode.com/analyze/markitdown:${CURRENT_TIME} \
  --push . > docker.log/markitdown.build.log 2>&1

echo "pushed: harbor.fzcode.com/analyze/markitdown:${CURRENT_TIME}"
