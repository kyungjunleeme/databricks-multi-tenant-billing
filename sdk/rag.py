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

# --- 1) 임베딩: serving_endpoints.query(name, input=...) ---
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

# --- 2) 생성(챗): serving_endpoints.query(name, messages=...) ---
SYS = "당신은 해커톤 규정 전문가입니다. 한국어로 간결히 답하고, 근거 출처를 함께 표기하세요."

def generate(prompt: str, temperature: float = 0.2) -> str:
    msgs = [
        ChatMessage(role=ChatMessageRole.SYSTEM, content=SYS),
        ChatMessage(role=ChatMessageRole.USER,   content=prompt),
    ]
    resp = _w.serving_endpoints.query(
        name=settings.gen_endpoint,
        messages=msgs,                # ✅ 핵심
        temperature=temperature,
        # max_tokens=512,             # 필요 시 조정
    )
    body = resp.as_dict() if hasattr(resp, "as_dict") else resp
    # 문서형 응답 예시: {"choices":[{"message":{"content":"..."}, ...}]}
    try:
        return body["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(body, ensure_ascii=False)

# --- 3) Vector Search: Self-managed → query_vector ---
def vs_query_with_vector(q_vec: list[float], k: int):
    payload = {
        "query_vector": q_vec,
        "k": int(k),
        # ✅ 네 워크스페이스는 columns 필수
        "columns": settings.vs_columns_list,
        # return_metadata는 환경에 따라 무시될 수 있으니 columns로 명시적으로 받자
        # "return_metadata": True,
    }
    url = _vs_query_url()
    r = requests.post(url, headers=_H(), data=json.dumps(payload))
    if r.status_code >= 400:
        raise RuntimeError(f"VS query failed {r.status_code}: {r.text}")
    body = r.json()

    # 표준 응답 1) results 배열
    if "results" in body:
        return body["results"]

    # 표준 응답 2) manifest + result.data_array (워크스페이스마다 다름)
    if "result" in body and "data_array" in body["result"]:
        manifest = body.get("manifest", {})
        cols = [c.get("name") for c in manifest.get("columns", [])]
        rows = body["result"]["data_array"]
        out = []
        for row in rows:
            meta = dict(zip(cols, row))
            out.append({"metadata": meta, "score": None})
        return out

    return []

from collections import defaultdict

CTX_MAX_CHUNKS = 6  # 프롬프트에 넣을 상위 청크 개수(표시는 전부 출력)

def answer(question: str):
    # 1) 질문 → 임베딩
    q_vec = embed_texts([question])[0]
    # 2) VS 질의 (vector)
    hits = vs_query_with_vector(q_vec, settings.top_k)

    def md(h, k):
        m = h.get("metadata", {}) if isinstance(h, dict) else {}
        return m.get(k)

    def safe_score(h):
        s = h.get("score")
        return float(s) if isinstance(s, (int, float)) else None

    # 3) 문서별로 전부 보존 (요약/자르기 없음)
    grouped = defaultdict(list)
    for h in hits:
        grouped[md(h, "source_path")].append({
            "chunk_text": md(h, "chunk_text") or "",
            "chunk_idx": md(h, "chunk_idx"),
            "score": safe_score(h),
        })

    citations = []
    ctx_chunks = []  # 프롬프트용 상위 몇 개만 담음
    for src, items in grouped.items():
        # 점수 desc → chunk_idx asc 정렬
        items_sorted = sorted(
            items,
            key=lambda x: (
                (x["score"] is not None, x["score"]),   # 점수 우선
                (x["chunk_idx"] is not None, -(x["chunk_idx"] or 0))  # 있으면 오름차순
            ),
            reverse=True
        )
        best_score = next((it["score"] for it in items_sorted if it["score"] is not None), None)

        # 컨텍스트(프롬프트)는 상위 일부만
        for it in items_sorted[:max(1, CTX_MAX_CHUNKS // max(1, len(grouped)))]:
            # 말 줄임 없이 전체 텍스트를 그대로 컨텍스트에 사용하면 토큰이 커지므로
            # 프롬프트에는 안전하게 앞부분만 살짝 제한(원문 출력은 UI에서 전부 함)
            t = (it["chunk_text"] or "").strip()
            if len(t) > 800:
                t = t[:800] + "…"
            ctx_chunks.append(t)

        citations.append({
            "source_path": src or "(unknown source)",
            "best_score": best_score,
            # 전체 청크 그대로(요약/자르기 없음)
            "chunks": items_sorted,  # [{chunk_idx, score, chunk_text}, ...]
        })

    # 4) 프롬프트 컨텍스트(상위 일부만)
    ctx = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(ctx_chunks, 1))

    # 5) 생성 호출
    prompt = f"{SYS}\n\n# 컨텍스트\n{ctx}\n\n# 질문\n{question}\n\n# 답변"
    out = generate(prompt)

    return out, citations
