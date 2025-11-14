# config.py
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # 기본 Databricks 접속 정보
    databricks_host: str = os.getenv("DATABRICKS_HOST", "")
    databricks_token: str = os.getenv("DATABRICKS_TOKEN", "")

    # Unity Catalog 위치
    catalog: str = os.getenv("UC_CATALOG", "saas_billing_analytics")
    uc_schema: str = os.getenv("UC_SCHEMA", "prod")
    sql_server_hostname: str = os.getenv(
        "DATABRICKS_SERVER_HOSTNAME",
        "",
    )
    sql_http_path: str = os.getenv(
        "DATABRICKS_HTTP_PATH",
        "",
    )

    @property
    def schema(self) -> str:
        # 나머지 코드에서 settings.schema 써도 동작하게
        return self.uc_schema

    # Gold 테이블 (월별 tenant billing 집계)
    gold_table: str = os.getenv("GOLD_TABLE", "billing_gold_tenant_monthly")

    # Vector Search 설정
    # 1) endpoint 기반이라면:
    vs_endpoint: str = os.getenv("VS_ENDPOINT", "billing-vs-endpoint")   # 네가 만든 엔드포인트 이름
    vsearch_index: str = os.getenv(
        "VSEARCH_INDEX",
        f"{catalog}.{schema}.billing_billing_vs_index",
    )

    # 2) UC 인덱스 풀네임으로도 필요할 수 있으니 property로 하나 만들어 둠
    @property
    def vsearch_index_fqn(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.vs_index_name}"

    # LLM / 임베딩 엔드포인트
    gen_endpoint: str = os.getenv("GEN_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
    emb_endpoint: str = os.getenv("EMB_ENDPOINT", "hack-embedder")

    # Vector Search에서 가져올 컬럼
    vs_columns: str = os.getenv("VS_COLUMNS", "chunk_id,tenant_name,region,text")

    @property
    def vs_columns_list(self):
        return [c.strip() for c in self.vs_columns.split(",") if c.strip()]

    # RAG 검색 TopK
    top_k: int = int(os.getenv("TOP_K", "5"))


settings = Settings()
