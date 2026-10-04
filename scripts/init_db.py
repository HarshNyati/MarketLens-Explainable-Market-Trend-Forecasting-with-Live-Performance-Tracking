#!/usr/bin/env python3
"""
scripts/init_db.py
──────────────────
One-shot script to initialise the PostgreSQL database.
Reads connection settings from .env (or environment variables)
and applies schema.sql.

Usage:
    python scripts/init_db.py
"""
import sys
import os

# Make project root importable when running as a standalone script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import get_db_connection  # noqa: E402 – path set above

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "schema.sql")


def init_db() -> None:
    schema_file = os.path.abspath(SCHEMA_PATH)

    if not os.path.exists(schema_file):
        print(f"❌  Schema file not found: {schema_file}")
        sys.exit(1)

    with open(schema_file, "r") as fh:
        sql = fh.read()

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        cur.close()
        conn.close()
        print("✅  Database initialised successfully (schema.sql applied).")
    except Exception as exc:
        print(f"❌  Database initialisation failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    init_db()
