"""
db_utils.py
-----------
Single source of truth for database connectivity across both pipelines
(underperforming-product/city detection and stockout/reorder prediction).

Credentials are read ONLY from config/db_config.json, per the task spec.
An optional DB_PASSWORD environment variable can override an empty
password in the config file (useful for local dev / CI) without ever
hardcoding a secret in source.
"""

import json
import os
import threading

import pandas as pd
from sqlalchemy import create_engine, text

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "db_config.json")

_lock = threading.Lock()
_engine = None
_config = None


def load_config(path: str = None) -> dict:
    """Load db_config.json (cached after first read)."""
    global _config
    path = path or _CONFIG_PATH
    if _config is None:
        with open(path, "r") as f:
            _config = json.load(f)
    return _config


def get_schema() -> str:
    return load_config().get("schema", "public")


def _connection_string() -> str:
    cfg = load_config()
    db = cfg["database"]
    password = db.get("password") or os.environ.get("DB_PASSWORD", "")
    user = db["user"]
    host = db["host"]
    port = db["port"]
    dbname = db["dbname"]
    auth = f"{user}:{password}" if password else user
    return f"postgresql+psycopg2://{auth}@{host}:{port}/{dbname}"


def get_engine():
    """Lazily create (and cache) a SQLAlchemy engine from db_config.json."""
    global _engine
    with _lock:
        if _engine is None:
            _engine = create_engine(_connection_string(), pool_pre_ping=True)
    return _engine


def read_table(table_name: str, columns=None, schema: str = None, where: str = None) -> pd.DataFrame:
    """Read a full table (or a column subset / filtered subset) from the warehouse."""
    schema = schema or get_schema()
    cols = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    query = f'SELECT {cols} FROM "{schema}"."{table_name}"'
    if where:
        query += f" WHERE {where}"
    return pd.read_sql(query, get_engine())


def read_query(sql: str, params: dict = None) -> pd.DataFrame:
    """Run an arbitrary read-only SQL query against the warehouse."""
    return pd.read_sql(sql, get_engine(), params=params)


def test_connection() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"DB connection test failed: {e}")
        return False
