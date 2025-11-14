# rag_core.py
import json
from collections import defaultdict

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

from databricks.vector_search.client import VectorSearchClient  # ★ 새로 사용
from config import settings

# --- 공통 Databricks SDK 클라이언트 ---
_w = WorkspaceClient(
    host=settings.databricks_host,
    token=settings.databricks_token,
)

# --- 1) 임베딩: serving_endpoints.query(name, input=...) ---
def embed_texts(texts: list[str]) -> list[list[float]]:
    if not isinstance(texts, list):
        raise TypeError("texts must be a list[str]")
    if not texts:
        return []
    resp = _w.serving_endpoints.query(
        name=settings.emb_endpoint,
        input=texts,
    )
    body = resp.as_dict() if hasattr(resp, "as_dict") else resp

    if isinstance(body, dict):
        if "embeddings" in body:
            return body["embeddings"]
        if "data" in body and isinstance(body["data"], list) and body["data"] and "embedding" in body["data"][0]:
            return [row["embedding"] for row in body["data"]]
        if "vectors" in body:
            return body["vectors"]
    if isinstance(body, list):
        return body
    print("❌ Unexpected embedding response:")
    print(json.dumps(body, indent=2, ensure_ascii=False))
    raise ValueError("Unexpected embedding response format")

# --- 2) 생성(챗): serving_endpoints.query(name, messages=...) ---
SYS = (
    "You are a Databricks tenant-billing analyst.\n"
    "Explain monthly usage and cost per tenant in clear English, "
    "and focus on business insights (spikes, anomalies, top SKUs, regions).\n"
)

def generate(prompt: str, temperature: float = 0.2) -> str:
    msgs = [
        ChatMessage(role=ChatMessageRole.SYSTEM, content=SYS),
        ChatMessage(role=ChatMessageRole.USER,   content=prompt),
    ]
    resp = _w.serving_endpoints.query(
        name=settings.gen_endpoint,
        messages=msgs,
        temperature=temperature,
    )
    body = resp.as_dict() if hasattr(resp, "as_dict") else resp
    try:
        return body["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(body, ensure_ascii=False)

# --- 3) Vector Search: databricks.vector_search.client 사용 ---
# def vs_query_with_vector(q_vec: list[float], k: int):
#     """
#     VectorSearchClient를 사용해서, endpoint/index 조합에 맞는 올바른 URL로 쿼리.
#     settings.vs_endpoint + settings.vs_index_name 를 사용.
#     """
#     # 1) 클라이언트 생성 (노트북/로컬 동일하게 PAT + host)
#     client = VectorSearchClient(
#         workspace_url=settings.databricks_host,
#         personal_access_token=settings.databricks_token,
#     )
#
#     # 2) 인덱스 핸들 얻기
#     #   - endpoint_name + index_name 조합 사용
#     index = client.get_index(
#         endpoint_name=settings.vs_endpoint,
#         index_name=settings.vs_index_name,
#     )
#     import pdb
#     pdb.set_trace()
#     # 3) 실제 쿼리 (API 버전에 따라 메서드 이름이 다를 수 있는데,
#     #    보통 vector / columns / num_results 형태로 받는다)
#     resp = client.query(
#         query_vector=q_vec,
#         num_results=int(k),
#         columns=settings.vs_columns_list,
#     )
#
#     # resp 구조는 보통 {"results":[{ "id":..., "score":..., "metadata":{...}}, ...]}
#     results = resp.get("results", [])
#     return results

def vs_query_with_vector(q_vec: list[float], k: int):
    """
    VectorSearchClient를 사용해서, endpoint/index 조합에 맞는 올바른 URL로 쿼리.
    settings.vs_endpoint + settings.vs_index_name 를 사용.
    """
    # 1) 클라이언트 생성 (노트북/로컬 동일하게 PAT + host)
    client = VectorSearchClient(
        workspace_url=settings.databricks_host,
        personal_access_token=settings.databricks_token,
    )

    # 2) 인덱스 핸들 얻기
    index = client.get_index(
        endpoint_name=settings.vs_endpoint,
        index_name=settings.vs_index_name,
    )
    # 3) similarity_search 로 쿼리
    resp = index.similarity_search(
        query_vector=q_vec,                  # 벡터로 쿼리
        columns=settings.vs_columns_list,    # 반환할 컬럼 리스트
        num_results=int(k),                  # top-k
    )

    # resp 는 보통 {"results": [...]} 형태
    results = resp.get("result", [])
    return results

# --- 4) RAG 전체 파이프라인 ---
CTX_MAX_CHUNKS = 6  # 프롬프트에 넣을 상위 청크 수

# def answer(question: str):
#     # 1) 질문 → 벡터
#     q_vec = embed_texts([question])[0]
#
#     # 2) VS 검색
#     try:
#         hits = vs_query_with_vector(q_vec, settings.top_k)
#     except Exception as e:
#         # 앱에서 에러 메시지 띄울 수 있도록
#         raise RuntimeError(f"VS query failed: {e}") from e
#
#     def md(h, k):
#         m = h.get("metadata", {}) if isinstance(h, dict) else {}
#         return m.get(k)
#
#     def safe_score(score):
#         try:
#             return float(score)
#         except (TypeError, ValueError):
#             return None
#
#     grouped = defaultdict(list)
#     for h in hits:
#         grouped[md(h, "tenant_name") or "(unknown-tenant)"].append(
#             {
#                 "chunk_text": md(h, "text") or "",
#                 "tenant_name": md(h, "tenant_name"),
#                 "region": md(h, "region"),
#                 "score": safe_score(h),
#             }
#         )

# from collections import defaultdict
#
# def answer(question: str):
#     # 1) 질문 → 벡터
#     q_vec = embed_texts([question])[0]
#
#     # 2) VS 검색
#     try:
#         hits = vs_query_with_vector(q_vec, settings.top_k)
#     except Exception as e:
#         # 앱에서 에러 메시지 띄울 수 있도록
#         raise RuntimeError(f"VS query failed: {e}") from e
#
#     def safe_score(score):
#         try:
#             return float(score)
#         except (TypeError, ValueError):
#             return None
#
#     grouped = defaultdict(list)
#
#     # --- Case A: 지금처럼 dict + data_array 형식일 때 ---
#     if isinstance(hits, dict) and "data_array" in hits:
#         for row in hits.get("data_array", []):
#             # 기대 형식: [id, tenant_name, region, text, score]
#             if len(row) < 5:
#                 # 형식이 예상과 다르면 스킵
#                 continue
#
#             row_id, tenant_name, region, text, score = row[:5]
#
#             tenant_key = tenant_name or "(unknown-tenant)"
#
#             grouped[tenant_key].append(
#                 {
#                     "row_id": row_id,
#                     "chunk_text": text or "",
#                     "tenant_name": tenant_name,
#                     "region": region,
#                     "score": safe_score(score),
#                 }
#             )
#
#     # --- Case B: 예전처럼 VS JSON(list of dict) 형식을 그대로 쓰는 경우도 대비 ---
#     else:
#         def md(h, k):
#             m = h.get("metadata", {}) if isinstance(h, dict) else {}
#             return m.get(k)
#
#         for h in hits or []:
#             tenant_key = md(h, "tenant_name") or "(unknown-tenant)"
#
#             grouped[tenant_key].append(
#                 {
#                     "chunk_text": md(h, "text") or "",
#                     "tenant_name": md(h, "tenant_name"),
#                     "region": md(h, "region"),
#                     "score": safe_score(
#                         h.get("score") if isinstance(h, dict) else None
#                     ),
#                 }
#             )
#
#     # 여기서부터는 기존 answer 로직(LLM 호출, context 생성 등) 그대로 이어가면 됨
#     # 예시:
#     #
#     # all_chunks = []
#     # for tenant, rows in grouped.items():
#     #     all_chunks.extend(rows)
#     # ...
#     # return final_answer, all_chunks
#     import pdb
#     pdb.set_trace()
#     # return grouped  # 일단 디버그용으로 이렇게 돌려보고, 나중에 LLM 호출 부분 연결
#
#     citations = []
#     ctx_chunks = []
#     import pdb
#     pdb.set_trace()
#     for tenant, items in grouped.items():
#         items_sorted = sorted(
#             items,
#             key=lambda x: (x["score"] is not None, x["score"]),
#             reverse=True,
#         )
#         best_score = next((it["score"] for it in items_sorted if it["score"] is not None), None)
#
#         # 상위 일부 텍스트만 프롬프트 컨텍스트로 사용
#         for it in items_sorted[:max(1, CTX_MAX_CHUNKS // max(1, len(grouped)))]:
#             t = (it["chunk_text"] or "").strip()
#             if len(t) > 800:
#                 t = t[:800] + "…"
#             ctx_chunks.append(f"[{tenant}] {t}")
#
#         citations.append(
#             {
#                 "tenant_name": tenant,
#                 "best_score": best_score,
#                 "chunks": items_sorted,
#             }
#         )
#
#     ctx = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(ctx_chunks, 1))
#
#     prompt = (
#         f"{SYS}\n\n"
#         f"# Context (tenant monthly usage rows)\n{ctx}\n\n"
#         f"# User question\n{question}\n\n"
#         "# Answer in business terms, focusing on tenant-level usage & cost insights.\n"
#     )
#
#     out = generate(prompt)
#     return out, citations

from collections import defaultdict

CTX_MAX_CHUNKS = 6  # 필요에 따라 조정

def answer(question: str):
    # 1) 질문 → 벡터
    q_vec = embed_texts([question])[0]

    # 2) VS 검색
    try:
        hits = vs_query_with_vector(q_vec, settings.top_k)
    except Exception as e:
        raise RuntimeError(f"VS query failed: {e}") from e

    def safe_score(score):
        try:
            return float(score)
        except (TypeError, ValueError):
            return None

    grouped = defaultdict(list)

    # --- 지금 구조: {"row_count": 6, "data_array": [[id, tenant, region, text, score], ...]} ---
    if isinstance(hits, dict) and "data_array" in hits:
        for row in hits.get("data_array", []):
            # 기대 구조: [id, tenant_name, region, text, score]
            if len(row) < 5:
                continue

            row_id, tenant_name, region, text, score = row[:5]
            tenant_key = tenant_name or "(unknown-tenant)"

            grouped[tenant_key].append(
                {
                    "row_id": row_id,
                    "chunk_text": text or "",
                    "tenant_name": tenant_name,
                    "region": region,
                    "score": safe_score(score),
                }
            )

    # --- fallback: 예전 VS JSON(list of dict) 응답일 때도 돌아가게 해둠 ---
    else:
        def md(h, k):
            m = h.get("metadata", {}) if isinstance(h, dict) else {}
            return m.get(k)

        for h in hits or []:
            tenant_key = md(h, "tenant_name") or "(unknown-tenant)"
            grouped[tenant_key].append(
                {
                    "chunk_text": md(h, "text") or "",
                    "tenant_name": md(h, "tenant_name"),
                    "region": md(h, "region"),
                    "score": safe_score(
                        h.get("score") if isinstance(h, dict) else None
                    ),
                }
            )

    # 3) 컨텍스트 & citation 구성
    citations = []
    ctx_chunks = []

    for tenant, items in grouped.items():
        # score 기준 내림차순 정렬 (None은 뒤로)
        items_sorted = sorted(
            items,
            key=lambda x: (x.get("score") is not None, x.get("score") or 0.0),
            reverse=True,
        )


        best_score = next(
            (it["score"] for it in items_sorted if it["score"] is not None),
            None,
        )

        # 테넌트 수에 따라 적당히 분배해서 상위 N개만 컨텍스트에 사용
        per_tenant = max(1, CTX_MAX_CHUNKS // max(1, len(grouped)))

        for it in items_sorted[:per_tenant]:
            # 여기서 반드시 str로 캐스팅해서 [object Object] 방지
            t = str(it.get("chunk_text", "") or "").strip()
            if len(t) > 800:
                t = t[:800] + "…"
            ctx_chunks.append(f"[{tenant}] {t}")

        citations.append(
            {
                "tenant_name": tenant,
                "best_score": best_score,
                "chunks": items_sorted,  # 여기엔 dict 리스트 그대로 둬도 됨 (JSON 직렬화 가능)
            }
        )

    # 번호 붙여서 최종 컨텍스트 문자열로
    ctx = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(ctx_chunks, 1))

    prompt = (
        f"{SYS}\n\n"
        f"# Context (tenant monthly usage rows)\n{ctx}\n\n"
        f"# User question\n{question}\n\n"
        "# Answer in business terms, focusing on tenant-level usage & cost insights.\n"
    )

    out = generate(prompt)
    return out, citations
