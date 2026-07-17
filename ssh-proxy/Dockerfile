FROM alpine:3.24

RUN apk add --no-cache openssh-client autossh

ENV AUTOSSH_GATETIME=0

# 私钥构建时拷进去
COPY id_rsa /root/.ssh/id_rsa
RUN chmod 700 /root/.ssh && chmod 600 /root/.ssh/id_rsa

CMD ["sh", "-c", "exec autossh -M 0 -N \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -L 0.0.0.0:${LOCAL_PORT}:${REMOTE_PROXY}:${REMOTE_PORT} \
    ${SSH_USER}@${SSH_HOST}"]