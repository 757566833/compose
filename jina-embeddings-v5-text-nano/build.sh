CURRENT_TIME=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
echo "time is: $CURRENT_TIME"
mkdir docker.log
docker build --platform linux/amd64 -t jina-embeddings-v5-text-nano:${CURRENT_TIME} . >  docker.log/jina-embeddings-v5-text-nano.build.log 2>&1

docker tag jina-embeddings-v5-text-nano:${CURRENT_TIME} harbor.fzcode.com/analyze/jina-embeddings-v5-text-nano:${CURRENT_TIME}
echo "docker push harbor.fzcode.com/analyze/jina-embeddings-v5-text-nano:${CURRENT_TIME}"