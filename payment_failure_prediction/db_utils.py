"""
db_utils.py
-----------
Shared infrastructure for the Stage 1 + Stage 2 orchestrator (main_pipeline.py).

Scope for THIS pass only (per design discussion — "Option A: shared
infrastructure lives in its own file, not inlined into the orchestrator"):

    - Load the central db_config.json (separate from validation_config.json,
      which file_validation.py and business_rules.py already load
      themselves at import time).
    - Open a Postgres connection for business_rules.run_all_rules(cur=...)
      to use for its warehouse-augmentation lookups.

Nothing related to CDC, SCD2, or fact/dimension table loading belongs here
— that is future scope and is intentionally NOT pre-built.
"""

import os
import json
import logging
import psycopg2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load central config at import time (mirrors how file_validation.py /
# business_rules.py already load validation_config.json at import time).
# ---------------------------------------------------------------------------
_DB_CONFIG_PATH = os.environ.get(
    "DB_CONFIG_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "db_config.json"),
)

with open(_DB_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _DB_CONFIG = json.load(_f)

_DATABASE = _DB_CONFIG["database"]

# Resolved schema string, exposed for main_pipeline.py to use when setting
# search_path on its own connection/cursor.
SCHEMA = _DB_CONFIG.get("schema", "insightlens")

logger.info(
    "db_utils: loaded central config from %s (schema=%s)",
    _DB_CONFIG_PATH, SCHEMA,
)


def get_connection():
    """
    Opens a new psycopg2 connection to the InsightLens warehouse using the
    credentials in db_config.json's "database" block.

    Deliberately does NOT set search_path itself and does NOT open a
    cursor — main_pipeline.py owns the cursor lifecycle (it needs the same
    cursor for both the search_path SET and business_rules.run_all_rules
    (cur=...)), and owns commit/rollback/close around Stage 2.
    """
    conn = psycopg2.connect(
        host=_DATABASE["host"],
        port=_DATABASE["port"],
        dbname=_DATABASE["dbname"],
        user=_DATABASE["user"],
        password=_DATABASE.get("password", ""),
    )
    logger.info(
        "db_utils: opened connection to %s:%s/%s as %s",
        _DATABASE["host"], _DATABASE["port"], _DATABASE["dbname"], _DATABASE["user"],
    )
    return conn
