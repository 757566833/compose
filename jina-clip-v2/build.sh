CURRENT_TIME=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
echo "time is: $CURRENT_TIME"
mkdir docker.log
docker build --platform linux/amd64 -t jina-clip-v2:${CURRENT_TIME} . >  docker.log/jina-clip-v2.build.log 2>&1

docker tag jina-clip-v2:${CURRENT_TIME} harbor.fzcode.com/analyze/jina-clip-v2:${CURRENT_TIME}
echo "docker push harbor.fzcode.com/analyze/jina-clip-v2:${CURRENT_TIME}"