from pyspark import pipelines as dp
from pyspark.sql import functions as F

BRONZE_TABLE = "billing_bronze_usage"


@dp.materialized_view(
    name="billing_silver_tenant_daily",
    comment="Silver: tenant-level daily DBU usage (by SKU, region, node_type).",
    table_properties={"pipelines.reset.allowed": "false"},
)
@dp.expect_or_drop("usage_unit_is_dbu",      "usage_unit = 'DBU'")
@dp.expect_or_drop("usage_quantity_positive","total_dbu > 0")
@dp.expect_or_drop("tenant_not_null",       "tenant_name IS NOT NULL")
@dp.expect_or_drop("usage_date_not_null",   "usage_date IS NOT NULL")
def billing_silver_tenant_daily():
    df = spark.read.table(BRONZE_TABLE)
    df = df.withColumn("usage_date", F.to_date("usage_date"))

    silver = (
        df.groupBy(
            "usage_date",
            "tenant_name",
            "sku_name",
            "usage_type",
            "usage_unit",
            "region",
            "node_type",
        )
        .agg(
            F.sum("usage_quantity").alias("total_dbu"),
            F.countDistinct("record_id").alias("record_cnt"),
        )
    )
    return silver

