CURRENT_TIME=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
echo "time is: $CURRENT_TIME"
mkdir docker.log
docker build --platform linux/amd64 -t audio-text:${CURRENT_TIME} . >  docker.log/audio-text.build.log 2>&1

docker tag audio-text:${CURRENT_TIME} harbor.fzcode.com/analyze/audio-text:${CURRENT_TIME}
echo "docker push harbor.fzcode.com/analyze/audio-text:${CURRENT_TIME}"