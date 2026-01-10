"""
Slice a large SQL dump by limiting inserted data size to ~10 GB (or custom target).

This script reads a .sql dump, applies schema statements, and stops ingesting INSERTs
once the cumulative INSERT text size reaches the target budget. Useful to create a
smaller SQLite database for quick experiments.

Example:
  python scripts/slice_sql_dump.py --dump "path\\to\\cepdb_2013-06-21.sql" --target-gb 10 --out "path\\to\\cepdb_10gb.db"
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def slice_dump(dump_path: Path, out_db: Path, target_gb: float) -> None:
    target_bytes = int(target_gb * 1024**3)
    out_db.unlink(missing_ok=True)
    con = sqlite3.connect(out_db)
    buf = []
    insert_bytes = 0
    with dump_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            buf.append(line)
            if ";" not in line:
                continue
            stmt = "".join(buf)
            buf.clear()
            stripped = stmt.strip()
            upper = stripped.upper()
            # Skip non-SQLite statements often present in MySQL/Postgres dumps
            skip_prefixes = (
                "SET ",
                "LOCK TABLES",
                "UNLOCK TABLES",
                "DELIMITER",
                "CREATE DATABASE",
                "USE ",
                "DROP DATABASE",
                "DROP TABLE",
                "DROP VIEW",
                "/*!",
                "--",
            )
            if upper.startswith(skip_prefixes):
                continue
            if upper.startswith("INSERT"):
                if insert_bytes >= target_bytes:
                    break
                insert_bytes += len(stmt.encode("utf-8"))
            # quick MySQL-ish cleanups
            stmt_sqlite = (
                stmt.replace("AUTO_INCREMENT", "")
                .replace("UNSIGNED", "")
                .replace("DEFAULT CHARSET=", "")
            )
            if "ENGINE=" in stmt_sqlite:
                # drop engine/options after closing paren
                if ") ENGINE=" in stmt_sqlite:
                    stmt_sqlite = stmt_sqlite.split(") ENGINE=")[0] + ");"
                else:
                    stmt_sqlite = stmt_sqlite.replace("ENGINE=", "")
            try:
                con.execute("BEGIN")
                con.executescript(stmt_sqlite)
                con.execute("COMMIT")
            except Exception as exc:
                # Skip statements SQLite cannot parse from the dump
                try:
                    con.execute("ROLLBACK")
                except Exception:
                    pass
                print(f"Skipping statement due to error ({exc}); first 80 chars: {stripped[:80]}")
    con.close()
    print(f"Done. Wrote ~{insert_bytes / 1024**3:.2f} GB of INSERT text to {out_db}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice a SQL dump to a target GB of INSERT text into a SQLite DB.")
    parser.add_argument("--dump", required=True, help="Path to .sql dump.")
    parser.add_argument("--out", required=True, help="Output SQLite DB path.")
    parser.add_argument("--target-gb", type=float, default=10.0, help="Target GB of INSERT text to import (default: 10).")
    args = parser.parse_args()

    slice_dump(Path(args.dump), Path(args.out), args.target_gb)


if __name__ == "__main__":
    main()
