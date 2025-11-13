from pyspark import pipelines as dp


@dp.temporary_view(name="v_billing_monthly_usd")
def v_billing_monthly_usd():
    """
    - billing_gold_tenant_monthly 배치 골드 테이블을 읽는 뷰
    - 문서 예제의 v_customers 에 해당 (batch side)
    """
    return spark.read.table("billing_gold_tenant_monthly")
