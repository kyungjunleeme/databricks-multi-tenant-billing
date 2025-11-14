# helpers.py

from databricks import sql
from databricks.sql.client import Connection, List, Row, Cursor

def get_connection_personal_access_token(
  server_hostname: str,
  http_path: str,
  access_token: str
) -> Connection:
  return sql.connect(
    server_hostname = server_hostname,
    http_path = http_path,
    access_token = access_token
  )

def select_nyctaxi_trips(
  connection: Connection,
  num_rows: int
) -> List[Row]:
  cursor: Cursor = connection.cursor()
  cursor.execute("SELECT * FROM samples.nyctaxi.trips LIMIT ?", [num_rows])
  result: List[Row] = cursor.fetchall()
  return result