import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from databricks import sdk
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient  # ★ 새로 사용

load_dotenv()


class Settings(BaseSettings):
    """
    Databricks Service Principal 기반 설정 파일.
    - PAT 대신 OAuth Client-Credential 인증 사용
    - 모든 Databricks SDK & HTTPS API 호출에 동일한 토큰 사용
    """

    # --- Databricks 기본 접속 설정 ---
    databricks_host: str = os.getenv(
        "DATABRICKS_HOST",
        "https://dbc-ac824534-b453.cloud.databricks.com"
    )
    databricks_client_id: str = os.getenv("DATABRICKS_CLIENT_ID", "")
    databricks_client_secret: str = os.getenv("DATABRICKS_CLIENT_SECRET", "")

    # --- Unity Catalog 위치 ---
    catalog: str = os.getenv("UC_CATALOG", "saas_billing_analytics")
    uc_schema: str = os.getenv("UC_SCHEMA", "prod")

    @property
    def schema(self) -> str:
        return self.uc_schema

    # --- SQL Warehouse 연결 정보 ---
    sql_server_hostname: str = os.getenv("DATABRICKS_SERVER_HOSTNAME", "")
    sql_http_path: str = os.getenv("DATABRICKS_HTTP_PATH", "")
    databricks_token: str = os.getenv("DATABRICKS_TOKEN", "")
    # --- Gold 테이블 ---
    gold_table: str = os.getenv(
        "GOLD_TABLE",
        "billing_gold_tenant_monthly"
    )

    # --- Vector Search 설정 ----
    vs_endpoint: str = os.getenv("VS_ENDPOINT", "billing-vs-endpoint")
    vs_index_name: str = os.getenv(
        "VSEARCH_INDEX",
        "saas_billing_analytics.prod.billing_billing_vs_index"
    )

    # --- Vector Search / LLM 등 나머지 설정은 그대로... ---
    # vs_endpoint, vs_index_name, gen_endpoint, emb_endpoint 등 생략

    # --- LLM / 임베딩 엔드포인트 --
    gen_endpoint: str = os.getenv("GEN_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
    emb_endpoint: str = os.getenv("EMB_ENDPOINT", "hack-embedder")

    # --- Vector Search 컬럼 ---
    vs_columns: str = os.getenv("VS_COLUMNS", "chunk_id,tenant_name,region,text")

    @property
    def vs_columns_list(self):
        return [c.strip() for c in self.vs_columns.split(",") if c.strip()]

    top_k: int = int(os.getenv("TOP_K", "5"))

    # -------------------------------------------
    # 🔥 Service Principal 기반 Workspace Client
    # -------------------------------------------
    def get_workspace_client(self) -> WorkspaceClient:
        """
        서비스 프린시플 기반 OAuth Client-Credential 인증 적용한 WorkspaceClient 생성
        """
        return WorkspaceClient(
            host=self.databricks_host,
            client_id=self.databricks_client_id,
            client_secret=self.databricks_client_secret,
        )

    # -------------------------------------------
    # 🔥 REST API 호출용 Authorization 헤더 생성
    # -------------------------------------------
    def get_auth_headers(self) -> dict:
        """
        WorkspaceClient의 OAuth 인증 토큰을 가져와 Authorization 헤더 생성
        """
        try:
            w = self.get_workspace_client()
            auth_header = w.config.authenticate()

            # case 1: {"Authorization": "Bearer xxx"} 형태
            if isinstance(auth_header, dict) and "Authorization" in auth_header:
                return {**auth_header, "Content-Type": "application/json"}

            # case 2: 반환값이 문자열(token)일 수도 있음
            token = str(auth_header) if auth_header else None
            if not token:
                raise RuntimeError("OAuth 토큰을 가져올 수 없습니다.")

            return {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }

        except Exception as e:
            raise RuntimeError(f"OAuth 인증 실패: {e}")


    # -------------------------------------------
    # 🔥 Service Principal 기반 Workspace Client
    # -------------------------------------------
    def get_vectors_search_client(self) -> VectorSearchClient:
        """
        서비스 프린시플 기반 OAuth Client-Credential 인증 적용한 WorkspaceClient 생성
        """
        # if not self.databricks_token:
        return VectorSearchClient(
            workspace_url=self.databricks_host,
            service_principal_client_id=self.databricks_client_id,
            service_principal_client_secret=self.databricks_client_secret,
        )
        # # else:
        # return VectorSearchClient(
        #     workspace_url=self.databricks_host,
        #     personal_access_token=self.databricks_token,
        # )

# Settings 싱글톤
settings = Settings()
