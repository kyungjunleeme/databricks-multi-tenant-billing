from pyspark import pipelines as dp

FX_FILES_PATH = "/Volumes/saas_billing_analytics/prod/fx_raw_json"


@dp.table(
    name="fx_rates_bronze",
    comment="Streaming FX rates ingested from JSON files via Auto Loader.",
)
def fx_rates_bronze():
    """
    - FX JSON 파일을 Auto Loader(cloudFiles)로 읽어오는 Streaming Table
    - 예: {"as_of_ts": "...", "base_ccy": "USD", "quote_ccy": "KRW", "fx_rate": 1450.23}
    """
    return (
        spark.readStream
            .format("cloudFiles")
            .option("cloudFiles.format", "json")
            .option("cloudFiles.inferColumnTypes", "true")
            .load(FX_FILES_PATH)
    )
