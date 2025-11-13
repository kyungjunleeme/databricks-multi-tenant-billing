from pyspark import pipelines as dp
from pyspark.sql import functions as F

# 파이프라인 config에서 읽기 (기본값: 2년치 샘플 테이블)
SOURCE_TABLE = spark.conf.get("p_source_table", "mock_billing_usage_clean")
TENANT_LIST  = spark.conf.get(
    "p_tenant_list",
    "천재교육,천재교과서,해법에듀,밀크티",
)

VALID_TENANTS = [t.strip() for t in TENANT_LIST.split(",") if t.strip()]
TENANT_IN_CLAUSE = ",".join(f"'{t}'" for t in VALID_TENANTS) if VALID_TENANTS else "''"


@dp.materialized_view(
    name="billing_bronze_usage",
    comment="Bronze: cleaned mock billing usage with tenant & basic filtering.",
    table_properties={"pipelines.reset.allowed": "false"},
)
@dp.expect_or_drop(
    "record_id_not_null",
    "record_id IS NOT NULL",
)
@dp.expect_or_drop(
    "usage_quantity_non_negative",
    "usage_quantity >= 0",
)
@dp.expect_or_drop(
    "valid_tenant_name",
    f"tenant_name IN ({TENANT_IN_CLAUSE})",
)
def billing_bronze_usage():
    """
    - mock_billing_usage_clean (또는 p_source_table)에서 읽음
    - 기본 품질 필터 적용
    """
    df = spark.read.table(SOURCE_TABLE)

    cols = [
        "record_id",
        "account_id",
        "workspace_id",
        "tenant_name",
        "sku_name",
        "cloud",
        "usage_start_time",
        "usage_end_time",
        "usage_date",
        "usage_unit",
        "usage_quantity",
        "billing_origin_product",
        "usage_type",
        "node_type",
        "region",
    ]
    existing_cols = [c for c in cols if c in df.columns]
    return df.select(*existing_cols)
