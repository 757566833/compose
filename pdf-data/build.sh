CURRENT_TIME=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
echo "time is: $CURRENT_TIME"
mkdir docker.log
docker build --platform linux/amd64 -t pdf-data:${CURRENT_TIME} . >  docker.log/pdf-data.build.log 2>&1

docker tag pdf-data:${CURRENT_TIME} harbor.fzcode.com/analyze/pdf-data:${CURRENT_TIME}
echo "docker push harbor.fzcode.com/analyze/pdf-data:${CURRENT_TIME}"