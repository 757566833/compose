"""
Jina Reranker M0 multimodal rerank service.

Endpoints:
- POST /rerank/text          —— text query, text documents
- POST /rerank/text-image    —— text query, image documents（含 base64/url，多模态以图搜文常用）
- POST /rerank/image-text    —— image query, text documents
- POST /rerank/image         —— image query, image documents（以图搜图二阶段重排）
- GET  /health               —— 健康检查

请求 / 响应字段对齐 Jina Cloud `/v1/rerank` 风格，方便后续切换托管服务无 schema 改动。

模型：`jinaai/jina-reranker-m0`（基于 Qwen2-VL-2B，支持文本/图像 cross-modal rerank）
"""

import base64
import io
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import httpx
import torch
from fastapi import FastAPI, HTTPException, status
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModel

# -----------------------------------------------------------------------------
# 日志 & 工具
# -----------------------------------------------------------------------------

app = FastAPI(title="Jina Reranker M0 Service")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_safe_filename(filename: Optional[str]) -> str:
    """安全处理文件名（去路径、限字符、防隐藏文件）。"""
    if not filename:
        return "file"
    name = filename.split("/")[-1].split("\\")[-1]
    name = re.sub(r"[^\w\.\-]", "_", name)
    if name.startswith("."):
        name = "_" + name[1:]
    return name if name.strip() else "file"


def validate_image_file(file_path: str) -> bool:
    """验证图片文件格式。"""
    try:
        with Image.open(file_path) as img:
            img.verify()
            return True
    except Exception:
        return False


# -----------------------------------------------------------------------------
# 模型加载
# -----------------------------------------------------------------------------

MODEL_BASE = os.getenv("MODEL_PATH", "/models")
device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
model_name = "jinaai/jina-reranker-m0"

# CPU 推理 cross-encoder 很慢，限制 batch 比 clip-v2 严
MAX_BATCH_SIZE = 5 if device == "cpu" else 32
logger.info(f"Batch size limit set to {MAX_BATCH_SIZE} (device: {device})")

full_model_path = os.path.abspath(os.path.join(MODEL_BASE, model_name))

try:
    model = AutoModel.from_pretrained(
        full_model_path,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully on device: {device}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Reranker M0 的 max_length 默认 1024，文档侧大段文本可调 2048（GPU 显存允许时）
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "1024"))
MAX_DOC_LENGTH = int(os.getenv("MAX_DOC_LENGTH", "2048"))


# -----------------------------------------------------------------------------
# 图片预处理工具：把任意 base64/dataURI/url 的"图片字符串"落成一个本地临时文件路径
# -----------------------------------------------------------------------------


async def materialize_image(
    image_str: str, idx: int, headers: Optional[Dict[str, str]] = None
) -> str:
    """
    接收 base64 / dataURI / http(s) URL 之一，落成磁盘临时文件并返回路径。
    调用者负责后续清理。
    """
    headers = headers or {}

    # 1. dataURI 或纯 base64
    if image_str.startswith("data:") or _looks_like_base64(image_str):
        b64 = image_str
        if b64.startswith("data:"):
            if ";base64," not in b64:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid dataURI format at index {idx}",
                )
            b64 = b64.split(";base64,", 1)[1]
        try:
            img_data = base64.b64decode(b64)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64 at index {idx}",
            )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
            tmp.write(img_data)
            tmp_path = tmp.name
        if not validate_image_file(tmp_path):
            os.remove(tmp_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Decoded base64 at index {idx} is not a valid image",
            )
        return tmp_path

    # 2. URL
    if image_str.startswith("http://") or image_str.startswith("https://"):
        url_path = image_str.split("?")[0]
        filename = url_path.split("/")[-1] or "file"
        suffix = Path(get_safe_filename(filename)).suffix or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            resp = await client.get(image_str, headers=headers)
            if resp.status_code != 200:
                os.remove(tmp_path)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to download {image_str}: HTTP {resp.status_code}",
                )
            with open(tmp_path, "wb") as f:
                async for chunk in resp.aiter_bytes():
                    f.write(chunk)
        if not validate_image_file(tmp_path):
            os.remove(tmp_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Downloaded file from {image_str} is not a valid image",
            )
        return tmp_path

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"image[{idx}] must be base64 / dataURI / http(s) URL",
    )


def _looks_like_base64(s: str) -> bool:
    """粗略判断是不是裸 base64 字符串（不含 dataURI 前缀）。
    HTTP URL / 短文本不会误判。
    """
    if len(s) < 32:
        return False
    if s.startswith("http://") or s.startswith("https://"):
        return False
    return bool(re.match(r"^[A-Za-z0-9+/=\s]+$", s[:200]))


def cleanup_paths(paths: List[str]) -> None:
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception as e:
                logger.error(f"清理临时文件失败 {p}: {e}")


# -----------------------------------------------------------------------------
# rerank 调用封装
# -----------------------------------------------------------------------------


def _compute_scores(
    pairs: List[List[Union[str, Path]]],
    query_type: str,
    doc_type: str,
) -> List[float]:
    """统一调 model.compute_score。pairs = [[query, doc], ...]
    query_type / doc_type ∈ {"text", "image"}。
    """
    try:
        with torch.no_grad():
            scores = model.compute_score(
                pairs,
                max_length=MAX_DOC_LENGTH,
                query_type=query_type,
                doc_type=doc_type,
            )
        # compute_score 返回 numpy / torch tensor，统一转 list[float]
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        return [float(s) for s in scores]
    except Exception as e:
        logger.error(f"compute_score 失败: {e}")
        raise HTTPException(status_code=500, detail=f"rerank 推理失败: {e}")


def _build_response(
    scores: List[float], top_n: Optional[int]
) -> Dict[str, Union[str, List[Dict[str, Union[int, float]]]]]:
    """对齐 Jina Cloud `/v1/rerank` 响应格式：results = [{index, relevance_score}]，按分降序。
    """
    indexed = [
        {"index": i, "relevance_score": float(s)} for i, s in enumerate(scores)
    ]
    indexed.sort(key=lambda x: x["relevance_score"], reverse=True)
    if top_n is not None and top_n >= 0:
        indexed = indexed[:top_n]
    return {"model": model_name, "results": indexed}


# -----------------------------------------------------------------------------
# 请求 schema
# -----------------------------------------------------------------------------


class TextTextRequest(BaseModel):
    query: str
    documents: List[str]
    top_n: Optional[int] = None


class TextImageRequest(BaseModel):
    query: str
    documents: List[str]  # base64 / dataURI / URL
    top_n: Optional[int] = None
    headers: Optional[Dict[str, str]] = None


class ImageTextRequest(BaseModel):
    query: str  # base64 / dataURI / URL
    documents: List[str]  # 文本
    top_n: Optional[int] = None
    headers: Optional[Dict[str, str]] = None


class ImageImageRequest(BaseModel):
    query: str  # base64 / dataURI / URL
    documents: List[str]  # base64 / dataURI / URL
    top_n: Optional[int] = None
    headers: Optional[Dict[str, str]] = None


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@app.post("/rerank/text", summary="文本 query 对文本 documents 重排")
def rerank_text_text(req: TextTextRequest):
    if not req.documents:
        return {"model": model_name, "results": []}
    if len(req.documents) > MAX_BATCH_SIZE * 4:  # 文本 batch 可放宽，model 内部分批
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many documents. Max {MAX_BATCH_SIZE * 4}",
        )
    pairs = [[req.query, doc] for doc in req.documents]
    scores = _compute_scores(pairs, query_type="text", doc_type="text")
    return _build_response(scores, req.top_n)


@app.post("/rerank/text-image", summary="文本 query 对图片 documents 重排（以文搜图二阶段）")
async def rerank_text_image(req: TextImageRequest):
    if not req.documents:
        return {"model": model_name, "results": []}
    if len(req.documents) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many image documents. Max {MAX_BATCH_SIZE}",
        )
    paths: List[str] = []
    try:
        for i, img in enumerate(req.documents):
            paths.append(await materialize_image(img, i, req.headers))
        pairs = [[req.query, p] for p in paths]
        scores = _compute_scores(pairs, query_type="text", doc_type="image")
        return _build_response(scores, req.top_n)
    finally:
        cleanup_paths(paths)


@app.post("/rerank/image-text", summary="图片 query 对文本 documents 重排")
async def rerank_image_text(req: ImageTextRequest):
    if not req.documents:
        return {"model": model_name, "results": []}
    if len(req.documents) > MAX_BATCH_SIZE * 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many documents. Max {MAX_BATCH_SIZE * 4}",
        )
    q_path: Optional[str] = None
    try:
        q_path = await materialize_image(req.query, -1, req.headers)
        pairs = [[q_path, doc] for doc in req.documents]
        scores = _compute_scores(pairs, query_type="image", doc_type="text")
        return _build_response(scores, req.top_n)
    finally:
        cleanup_paths([q_path] if q_path else [])


@app.post("/rerank/image", summary="图片 query 对图片 documents 重排（以图搜图二阶段）")
async def rerank_image_image(req: ImageImageRequest):
    if not req.documents:
        return {"model": model_name, "results": []}
    if len(req.documents) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many image documents. Max {MAX_BATCH_SIZE}",
        )
    q_path: Optional[str] = None
    doc_paths: List[str] = []
    try:
        q_path = await materialize_image(req.query, -1, req.headers)
        for i, img in enumerate(req.documents):
            doc_paths.append(await materialize_image(img, i, req.headers))
        pairs = [[q_path, p] for p in doc_paths]
        scores = _compute_scores(pairs, query_type="image", doc_type="image")
        return _build_response(scores, req.top_n)
    finally:
        cleanup_paths(([q_path] if q_path else []) + doc_paths)


@app.get("/health", summary="服务健康状态检查")
def health_check():
    return {
        "status": "active",
        "model": model_name,
        "device": str(device),
        "max_query_length": MAX_QUERY_LENGTH,
        "max_doc_length": MAX_DOC_LENGTH,
        "max_batch_size": MAX_BATCH_SIZE,
        "api_version": "1.0",
    }
