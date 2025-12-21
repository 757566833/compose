CURRENT_TIME=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
echo "time is: $CURRENT_TIME"
mkdir docker.log
docker build --platform linux/amd64 -t video-vector:${CURRENT_TIME} . >  docker.log/video-vector.build.log 2>&1

docker tag video-vector:${CURRENT_TIME} harbor.fzcode.com/analyze/video-vector:${CURRENT_TIME}
echo "docker push harbor.fzcode.com/analyze/video-vector:${CURRENT_TIME}"