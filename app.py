import os
import logging

import pandas as pd
import streamlit as st
from databricks.sdk.core import Config, oauth_service_principal
from databricks import sql

from config import settings
from rag_core import answer

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------
# Databricks SQL Connector helper
# ---------------------------

def query_df(sql_text: str) -> pd.DataFrame:
    """
    네가 직접 테스트해서 성공한 방식과 동일한 패턴으로,
    Databricks SQL Connector를 사용해 DataFrame을 조회한다.
    """
    if not settings.databricks_token:
        def credential_provider():
            config = Config(
                host=f"{settings.sql_server_hostname}",
                client_id=os.getenv("DATABRICKS_CLIENT_ID"),
                client_secret=os.getenv("DATABRICKS_CLIENT_SECRET"))
            return oauth_service_principal(config)

        connection = sql.connect(
            server_hostname=settings.sql_server_hostname,
            http_path=settings.sql_http_path,
            credentials_provider=credential_provider,
        )
    else:
        connection = sql.connect(
            server_hostname=settings.sql_server_hostname,
            http_path=settings.sql_http_path,
            access_token=settings.databricks_token,
        )
    try:
        cursor = connection.cursor()
        cursor.execute(sql_text)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return pd.DataFrame.from_records(rows, columns=cols)
    finally:
        cursor.close()
        connection.close()

@st.cache_data
def load_billing_data() -> pd.DataFrame:
    gold_fqn = f"{settings.catalog}.{settings.schema}.{settings.gold_table}"
    q = f"""
        SELECT
          year,
          month,
          tenant_name,
          sku_name,
          usage_type,
          usage_unit,
          region,
          node_type,
          monthly_dbu,
          est_cost_usd,
          monthly_records
        FROM {gold_fqn}
    """
    return query_df(q)


# ---------------------------
# Streamlit Layout
# ---------------------------

st.set_page_config(
    page_title="Tenant Billing RAG Center",
    page_icon="🧱",
    layout="wide",
)

st.title("🧱 Tenant Billing RAG Center")

df_billing = load_billing_data()
if df_billing.empty:
    st.error("No billing data loaded from Databricks.")
    st.stop()

df_billing["year"] = df_billing["year"].astype(int)
df_billing["month"] = df_billing["month"].astype(int)

with st.sidebar:
    st.header("🔎 Filters")
    tenants = sorted(df_billing["tenant_name"].dropna().unique().tolist())
    sel_tenants = st.multiselect("Tenant", ["(All)"] + tenants, default="(All)")
    if "(All)" in sel_tenants:
        tenant_filter = tenants
    else:
        tenant_filter = sel_tenants

    years = sorted(df_billing["year"].unique().tolist())
    sel_years = st.multiselect("Year", years, default=years)

    months = sorted(df_billing["month"].unique().tolist())
    sel_months = st.multiselect("Month", months, default=months)

    regions = sorted(df_billing["region"].dropna().unique().tolist())
    sel_regions = st.multiselect("Region", ["(All)"] + regions, default="(All)")
    if "(All)" in sel_regions:
        region_filter = regions
    else:
        region_filter = sel_regions

    usage_types = sorted(df_billing["usage_type"].dropna().unique().tolist())
    sel_usage_types = st.multiselect("Usage Type", ["(All)"] + usage_types, default="(All)")
    if "(All)" in sel_usage_types:
        usage_type_filter = usage_types
    else:
        usage_type_filter = sel_usage_types

df_filtered = df_billing[
    df_billing["tenant_name"].isin(tenant_filter)
    & df_billing["year"].isin(sel_years)
    & df_billing["month"].isin(sel_months)
    & df_billing["region"].isin(region_filter)
    & df_billing["usage_type"].isin(usage_type_filter)
]

col1, col2, col3 = st.columns(3)
total_cost = df_filtered["est_cost_usd"].sum()
recent = df_filtered.sort_values(["year", "month"]).tail(1)
recent_cost = float(recent["est_cost_usd"].sum()) if not recent.empty else 0.0
tenant_count = df_filtered["tenant_name"].nunique()

col1.metric("Total Estimated Cost (USD)", f"{total_cost:,.2f}")
col2.metric("Last Period Cost (USD)", f"{recent_cost:,.2f}")
col3.metric("Active Tenants", tenant_count)

st.subheader("💰 Monthly Cost by Tenant")
cost_by_tenant = (
    df_filtered.groupby("tenant_name", as_index=False)["est_cost_usd"]
    .sum()
    .sort_values("est_cost_usd", ascending=False)
)
st.bar_chart(cost_by_tenant, x="tenant_name", y="est_cost_usd")

st.markdown("#### Detailed Rows")
st.dataframe(
    df_filtered.sort_values(["year", "month", "tenant_name"]),
    width="stretch",
)

st.subheader("🤖 Ask about Tenant Billing (RAG)")
default_q = "Which tenants have the highest monthly cost trend and why?"
question = st.text_input("Ask a question (English, for the judges)", value=default_q)

if st.button("Run RAG Analysis", type="primary"):
    with st.spinner("Running semantic search + LLM..."):
        try:
            ans, hits = answer(question)
        except Exception as e:
            st.error(f"RAG pipeline failed: {e}")
        else:
            st.markdown("##### 📌 LLM Answer")
            st.markdown(ans)

            import pandas as pd

            st.markdown("##### 🔎 Retrieved Context (Vector Search)")
            if not hits:
                st.info("No relevant rows found in the vector index.")
            else:
                rows = []
                for h in hits:
                    tenant = h["tenant_name"]
                    best_score = h["best_score"]
                    for ch in h["chunks"]:
                        rows.append({
                            "tenant_name": tenant,
                            "best_score": best_score,
                            **ch,  # row_id, chunk_text, region, score ...
                        })

                df = pd.DataFrame(rows)
                st.dataframe(df, width="stretch")
