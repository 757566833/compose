CURRENT_TIME=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
echo "time is: $CURRENT_TIME"
mkdir docker.log
docker build --platform linux/amd64 -t ppt-parse:${CURRENT_TIME} . >  docker.log/ppt-parse.build.log 2>&1

docker tag ppt-parse:${CURRENT_TIME} harbor.fzcode.com/analyze/ppt-parse:${CURRENT_TIME}
echo "docker push harbor.fzcode.com/analyze/ppt-parse:${CURRENT_TIME}"