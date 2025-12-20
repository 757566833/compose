CURRENT_TIME=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
echo "time is: $CURRENT_TIME"
mkdir docker.log
docker build --platform linux/amd64 -t image-vector:${CURRENT_TIME} . >  docker.log/image-vector.build.log 2>&1

docker tag image-vector:${CURRENT_TIME} harbor.fzcode.com/analyze/image-vector:${CURRENT_TIME}
echo "docker push harbor.fzcode.com/analyze/image-vector:${CURRENT_TIME}"