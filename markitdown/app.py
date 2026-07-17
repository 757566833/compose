"""markitdown REST API：二进制/URL 进，Markdown 字节流出。"""

import io
import os
from urllib.parse import urlparse

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from markitdown import MarkItDown, StreamInfo
from markitdown._exceptions import MarkItDownException
from pydantic import BaseModel

app = FastAPI(title="markitdown-api")
md = MarkItDown(enable_plugins=False)

MARKDOWN_MEDIA_TYPE = "text/markdown; charset=utf-8"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/convert")
def convert_binary(file: UploadFile = File(...)) -> Response:
    """multipart 上传任意文件，返回 Markdown 字节流。"""
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    ext = os.path.splitext(file.filename or "")[1].lower() or None
    try:
        result = md.convert_stream(
            io.BytesIO(data),
            stream_info=StreamInfo(extension=ext, filename=file.filename),
        )
    except MarkItDownException as e:
        raise HTTPException(status_code=422, detail=f"conversion failed: {e}")
    return Response(content=result.markdown.encode("utf-8"), media_type=MARKDOWN_MEDIA_TYPE)


class ConvertUrlRequest(BaseModel):
    url: str


@app.post("/convert-url")
def convert_url(req: ConvertUrlRequest) -> Response:
    """给一个 http(s) 文件 URL，服务端拉取并转换，返回 Markdown 字节流。"""
    scheme = urlparse(req.url).scheme.lower()
    if scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="only http/https urls are allowed")
    try:
        result = md.convert(req.url)
    except MarkItDownException as e:
        raise HTTPException(status_code=422, detail=f"conversion failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"fetch failed: {e}")
    return Response(content=result.markdown.encode("utf-8"), media_type=MARKDOWN_MEDIA_TYPE)
