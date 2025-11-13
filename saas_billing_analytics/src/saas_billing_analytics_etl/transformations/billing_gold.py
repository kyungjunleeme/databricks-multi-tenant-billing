from pyspark import pipelines as dp
from pyspark.sql import functions as F

SILVER_TABLE    = "billing_silver_tenant_daily"
DBU_PRICE_TABLE = "dim_dbu_price"  # saas_billing_analytics.prod.dim_dbu_price


@dp.materialized_view(
    name="billing_gold_tenant_monthly",
    comment="Gold: tenant-level monthly billing view (USD) for account managers.",
    table_properties={"pipelines.reset.allowed": "false"},
)
@dp.expect(
    "estimated_cost_non_negative",
    "est_cost_usd >= 0",
)
def billing_gold_tenant_monthly():
    """
    - Silver(일별 DBU) + dim_dbu_price 조인
    - 연/월/테넌트/SKU/usage_type/region/node_type 기준 월별 DBU & USD 비용 집계
    """
    silver = spark.read.table(SILVER_TABLE)
    price  = spark.read.table(DBU_PRICE_TABLE)

    df = (
        silver
        .withColumn("year",  F.year("usage_date"))
        .withColumn("month", F.month("usage_date"))
    )

    monthly = (
        df.groupBy(
            "year",
            "month",
            "tenant_name",
            "sku_name",
            "usage_type",
            "usage_unit",
            "region",
            "node_type",
        )
        .agg(
            F.sum("total_dbu").alias("monthly_dbu"),
            F.sum("record_cnt").alias("monthly_records"),
        )
    )

    joined = (
        monthly
        .join(
            price,
            on=["sku_name", "region", "node_type"],
            how="left",
        )
        .withColumn("usd_per_dbu", F.coalesce(F.col("usd_per_dbu"), F.lit(0.0)))
        .withColumn("est_cost_usd", F.col("monthly_dbu") * F.col("usd_per_dbu"))
    )

    result = (
        joined
        .select(
            "year",
            "month",
            "tenant_name",
            "sku_name",
            "usage_type",
            "usage_unit",
            "region",
            "node_type",
            "monthly_dbu",
            "est_cost_usd",
            "monthly_records",
        )
        .orderBy("year", "month", "tenant_name", "sku_name", "node_type")
    )

    return result
