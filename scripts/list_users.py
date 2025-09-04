#!/usr/bin/env python3
# scripts/list_users.py
from __future__ import annotations

import os, sys, sqlite3
from textwrap import shorten

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import DB_PATH

def main():
    if not os.path.exists(DB_PATH):
        print(f"No DB found at {DB_PATH}. Open the app once so it initializes the DB.")
        return
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "SELECT id, email, display_name, google_sub, created_at FROM users ORDER BY created_at DESC"
    )
    rows = cur.fetchall()
    if not rows:
        print("No users in DB yet. Sign in through the app first.")
        return
    print(f"{'id':>4} | {'email':<28} | {'name':<20} | {'sub':<22} | created_at")
    print("-" * 96)
    for r in rows:
        uid, email, name, sub, created = r
        print(f"{uid:>4} | {shorten(email or '', 28):<28} | {shorten(name or '',20):<20} | "
              f"{shorten(sub or '',22):<22} | {created}")
    conn.close()

if __name__ == "__main__":
    main()
