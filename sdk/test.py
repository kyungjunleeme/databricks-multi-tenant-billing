# rag_core.py — databricks-sdk-py 기반 호출 (권장)
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
import requests

from config import settings

# --- 공통 ---
def _H():
    return {"Authorization": f"Bearer {settings.databricks_token}",
            "Content-Type": "application/json"}

def _vs_query_url():
    # Self-managed 인덱스는 query_vector 필수
    return f"{settings.databricks_host}/api/2.0/vector-search/indexes/{settings.vsearch_index}/query"

# Databricks SDK 클라이언트(호스트/토큰 명시해 자동 감지 차단)
_w = WorkspaceClient(host=settings.databricks_host, token=settings.databricks_token)

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    databricks-sdk-py 시그니처에 맞춰 'input' 사용.
    (임베딩 엔드포인트는 input/extra_params만 쓰는 게 정석)
    """
    resp = _w.serving_endpoints.query(
        name=settings.emb_endpoint,
        input=texts,                  # ✅ 핵심
        # extra_params={"truncate":"NONE"}  # 필요시 모델별 파라미터
    )
    # 응답 파싱: SDK가 dict로 변환해 주므로 케이스 흡수
    # 흔한 형태:
    #  - {"embeddings":[[...],[...]]}
    #  - {"data":[{"embedding":[...]}, ...]}
    #  - {"vectors":[[...],[...]]}
    body = resp.as_dict() if hasattr(resp, "as_dict") else resp  # 안전
    if isinstance(body, dict):
        if "embeddings" in body:
            return body["embeddings"]
        if "data" in body and isinstance(body["data"], list) and body["data"] and "embedding" in body["data"][0]:
            return [row["embedding"] for row in body["data"]]
        if "vectors" in body:
            return body["vectors"]
    if isinstance(body, list):
        return body
    raise ValueError(f"Unexpected embedding response: {body}")