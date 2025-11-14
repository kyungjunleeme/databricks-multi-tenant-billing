import json

import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

from config import settings


_w = WorkspaceClient(
    host=settings.databricks_host,
    token=settings.databricks_token,
)

def _H():
    return {"Authorization": f"Bearer {settings.databricks_token}",
            "Content-Type": "application/json"}


def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = _w.serving_endpoints.query(
        name=settings.emb_endpoint,
        input=texts,
    )
    body = resp.as_dict() if hasattr(resp, "as_dict") else resp

    if isinstance(body, dict) and "embeddings" in body:
        return body["embeddings"]
    if isinstance(body, dict) and "data" in body and body["data"] and "embedding" in body["data"][0]:
        return [row["embedding"] for row in body["data"]]
    if isinstance(body, dict) and "vectors" in body:
        return body["vectors"]
    if isinstance(body, list):
        return body
    raise ValueError(f"Unexpected embedding response: {body}")


SYS = (
    "You are a FinOps assistant for Databricks SaaS billing.\n"
    "You receive tenant-level monthly usage data and must answer in English.\n"
    "Explain usage patterns, high-cost months, and anomalies in a concise way."
)


def generate(prompt: str) -> str:
    msgs = [
        ChatMessage(role=ChatMessageRole.SYSTEM, content=SYS),
        ChatMessage(role=ChatMessageRole.USER, content=prompt),
    ]
    resp = _w.serving_endpoints.query(
        name=settings.gen_endpoint,
        messages=msgs,
        temperature=0.2,
    )
    body = resp.as_dict() if hasattr(resp, "as_dict") else resp
    try:
        return body["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(body, ensure_ascii=False)


def vs_query_vector(q_vec: list[float], k: int, tenant_filter: str | None = None):
    url = f"{settings.databricks_host}/api/2.0/vector-search/indexes/{settings.vsearch_index}/query"
    constraints = []
    if tenant_filter:
        constraints.append({
            "column": "tenant_name",
            "value": tenant_filter,
            "operator": "EQ",
        })

    payload = {
        "query_vector": q_vec,
        "k": int(k),
        "columns": [
            "chunk_id", "text",
            "tenant_name", "year", "month",
            "sku_name", "usage_type", "usage_unit",
            "region", "node_type",
            "monthly_dbu", "est_cost_usd", "monthly_records",
        ],
    }
    if constraints:
        payload["filters"] = {"constraints": constraints}
    r = requests.post(url, headers=_H(), data=json.dumps(payload))
    r.raise_for_status()
    body = r.json()

    if "results" in body:
        return body["results"]
    if "result" in body and "data_array" in body["result"]:
        cols = [c["name"] for c in body["manifest"]["columns"]]
        rows = body["result"]["data_array"]
        out = []
        for row in rows:
            meta = dict(zip(cols, row))
            out.append({"metadata": meta, "score": None})
        return out
    return []


def answer(question: str, tenant_filter: str | None = None, k: int = 8):
    # 1) 질문 임베딩
    q_vec = embed_texts([question])[0]

    # 2) VS 조회
    hits = vs_query_vector(q_vec, k=k, tenant_filter=tenant_filter)

    ctx_lines = []
    citations = []
    for i, h in enumerate(hits, 1):
        meta = h.get("metadata", h)
        line = (
            f"[{i}] Tenant={meta.get('tenant_name')} "
            f"Year={meta.get('year')} Month={meta.get('month')} "
            f"SKU={meta.get('sku_name')} UsageType={meta.get('usage_type')} "
            f"DBU={meta.get('monthly_dbu')} CostUSD={meta.get('est_cost_usd')}\n"
            f"Text: {meta.get('text')}"
        )
        ctx_lines.append(line)
        citations.append(meta)

    ctx = "\n\n".join(ctx_lines)

    prompt = (
        f"{SYS}\n\n"
        f"# Context (top {len(hits)} monthly billing records)\n"
        f"{ctx}\n\n"
        f"# Question\n{question}\n\n"
        f"# Answer:\n"
    )
    out = generate(prompt)
    return out, citations
