CURRENT_TIME=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
echo "time is: $CURRENT_TIME"
mkdir docker.log
docker build --platform linux/amd64 -t ocr:${CURRENT_TIME} . >  docker.log/ocr.build.log 2>&1

docker tag ocr:${CURRENT_TIME} harbor.fzcode.com/analyze/ocr:${CURRENT_TIME}
echo "docker push harbor.fzcode.com/analyze/ocr:${CURRENT_TIME}"