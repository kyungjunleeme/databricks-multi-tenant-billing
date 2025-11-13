USE CATALOG saas_billing_analytics;
USE SCHEMA prod;

CREATE OR REPLACE TABLE billing_tenant_monthly_text AS
SELECT
  /* PK용 ID */
  monotonically_increasing_id() AS chunk_id,
  tenant_name,
  year,
  month,
  sku_name,
  usage_type,
  usage_unit,
  region,
  node_type,
  monthly_dbu,
  est_cost_usd,
  monthly_records,
  /* 사람이 읽기 좋은 설명 텍스트 (한국어로!) */
  concat(
    '테넌트 ', tenant_name,
    '의 ', year, '년 ', lpad(month, 2, '0'), '월 사용 요약입니다. ',
    'SKU는 ', coalesce(sku_name, 'N/A'),
    ', 사용 유형은 ', coalesce(usage_type, 'N/A'),
    ', 지역(region)은 ', coalesce(region, 'N/A'),
    ', 노드 타입은 ', coalesce(node_type, 'N/A'),
    ' 입니다. ',
    '해당 기간 동안 DBU 사용량은 ', cast(monthly_dbu as string),
    ', 예상 비용은 약 ', cast(round(est_cost_usd, 2) as string), ' USD 이고, ',
    '레코드 수는 ', cast(monthly_records as string), ' 개입니다.'
  ) AS text
FROM billing_gold_tenant_monthly;


ALTER TABLE billing_tenant_monthly_text
ADD COLUMN embedding ARRAY<FLOAT>;