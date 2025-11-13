from pyspark import pipelines as dp
from pyspark.sql import functions as F


@dp.table(
    name="billing_gold_tenant_monthly_fx_streaming",
    comment="Streaming: monthly tenant billing with real-time FX (USD→KRW).",
)
def billing_gold_tenant_monthly_fx_streaming():
    """
    - v_fx_usd_krw (Streaming FX)  +  v_billing_monthly_usd (Batch billing)
    - FX가 새로 들어올 때마다 KRW 비용이 실시간으로 다시 계산됨
    """
    fx_stream = spark.readStream.table("v_fx_usd_krw")
    billing_batch = spark.read.table("v_billing_monthly_usd")

    joined = (
        billing_batch.crossJoin(fx_stream)
            .withColumn("est_cost_krw", F.col("est_cost_usd") * F.col("fx_rate"))
    )

    return joined
