from pyspark import pipelines as dp


@dp.temporary_view(name="v_fx_usd_krw")
def v_fx_usd_krw():
    """
    - fx_rates_bronze 스트리밍 테이블에서 USD → KRW 만 필터링한 뷰
    - 문서 예제의 v_transactions 에 해당 (streaming side)
    """
    return (
        spark.readStream.table("fx_rates_bronze")
            .filter("base_ccy = 'USD' AND quote_ccy = 'KRW'")
    )
