CURRENT_TIME=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
echo "time is: $CURRENT_TIME"
mkdir docker.log
docker build --platform linux/amd64 -t model-six-view:${CURRENT_TIME} . >  docker.log/model-six-view.build.log 2>&1

docker tag model-six-view:${CURRENT_TIME} harbor.fzcode.com/analyze/model-six-view:${CURRENT_TIME}
echo "docker push harbor.fzcode.com/analyze/model-six-view:${CURRENT_TIME}"