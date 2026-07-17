# markitdown

microsoft/markitdown 的 MCP server（Streamable HTTP）+ REST API + CLI 容器。
镜像内含 `markitdown[all]` 全量 extras + ffmpeg + exiftool。

## 启动

```bash
docker compose up -d --build
```

- MCP 端点：`http://localhost:3001/mcp`（Streamable HTTP），工具：`convert_to_markdown(uri)`
- uri 支持 `http:` / `https:` / `file:` / `data:`；本地文件放 `./files/`，
  容器内路径为 `file:///workdir/<文件名>`
- REST API：`http://localhost:3002`（见下）

## REST API（3002，文件/URL 进 → Markdown 字节流出）

```bash
# 二进制上传（multipart，字段名 file，靠文件名后缀识别格式）
curl -F 'file=@a.pdf' http://localhost:3002/convert -o a.md

# 文件 URL（服务端拉取后转换；仅允许 http/https）
curl -X POST http://localhost:3002/convert-url \
  -H 'Content-Type: application/json' -d '{"url":"https://.../a.docx"}' -o a.md
```

响应 `200` 时 body 即 Markdown（`text/markdown; charset=utf-8`）；
转换失败 `422`，URL 拉取失败 `502`，非 http/https 协议 `400`。
健康检查：`GET /health`。

## 当 CLI 用（不走 MCP）

```bash
# 转换 ./files/ 里的文件
docker exec markitdown markitdown /workdir/a.pdf

# 或 stdin/stdout 一次性运行
docker run --rm -i markitdown:latest markitdown < in.docx > out.md
```

## 说明

- PDF 默认是纯文字层抽取（pdfplumber/pdfminer），扫描件出来为空；
  高质量 PDF/OCR 需外接 Azure Document Intelligence 或 LLM Vision，见主仓文档
- 音频转录走 Google Web Speech 在线接口，需容器能出网
- `build.sh` 做多架构构建并推 harbor；本地开发直接 `docker compose up -d --build`
