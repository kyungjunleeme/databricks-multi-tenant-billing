import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
load_dotenv()

class Settings(BaseSettings):
    databricks_host: str = os.getenv('DATABRICKS_HOST', '')
    databricks_token: str = os.getenv('DATABRICKS_TOKEN', '')
    catalog: str = os.getenv('UC_CATALOG', 'saas_billing_analytics')
    uc_schema: str = os.getenv('UC_SCHEMA', 'prod')  # 이름 변경
    vsearch_index: str = f"{catalog}.{uc_schema}.billing_billing_vs_index"
    gen_endpoint: str = os.getenv('GEN_ENDPOINT', 'hack-llm-generate')
    emb_endpoint: str = os.getenv('EMB_ENDPOINT', 'hack-embedder')
    top_k: int = int(os.getenv('TOP_K', '5'))
    vs_columns: str = os.getenv('VS_COLUMNS', 'chunk_id,tenant_name,source_path')

    @property
    def vs_columns_list(self):
        return [c.strip() for c in self.vs_columns.split(',') if c.strip()]