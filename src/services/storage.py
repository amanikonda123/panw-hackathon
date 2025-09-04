# src/services/storage.py
import sqlite3, json, os
import pandas as pd
from typing import List, Dict, Any
from src.config import DB_PATH

def _connect():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _connect() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            google_sub TEXT UNIQUE,
            email TEXT,
            display_name TEXT,
            picture_url TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            ts TEXT NOT NULL,
            text TEXT NOT NULL,
            sentiment_label TEXT,
            sentiment_score REAL,
            top_keywords TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """)
        conn.commit()

# ---------- Google users ----------
def get_user_by_google_sub(sub: str) -> Dict[str, Any] | None:
    with _connect() as conn:
        cur = conn.execute(
            "SELECT id, google_sub, email, display_name, picture_url FROM users WHERE google_sub = ?",
            (sub,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {"id": row[0], "google_sub": row[1], "email": row[2], "display_name": row[3], "picture_url": row[4]}

def create_user_google(sub: str, email: str, display_name: str, picture_url: str) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO users (google_sub, email, display_name, picture_url) VALUES (?, ?, ?, ?)",
            (sub, email, display_name, picture_url),
        )
        conn.commit()
        return cur.lastrowid

def get_or_create_user_google(sub: str, email: str, display_name: str, picture_url: str) -> Dict[str, Any]:
    user = get_user_by_google_sub(sub)
    if user:
        return user
    uid = create_user_google(sub, email, display_name, picture_url)
    return {"id": uid, "google_sub": sub, "email": email, "display_name": display_name, "picture_url": picture_url}

# ---------- Entries ----------
def insert_entry(user_id: int, ts: str, text: str, sent_label: str, sent_score: float, top_keywords: List[str]):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO entries (user_id, ts, text, sentiment_label, sentiment_score, top_keywords) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, ts, text, sent_label, float(sent_score), json.dumps(top_keywords)),
        )
        conn.commit()

def load_entries_df(user_id: int) -> pd.DataFrame:
    with _connect() as conn:
        df = pd.read_sql_query(
            "SELECT id, ts, text, sentiment_label, sentiment_score, top_keywords FROM entries WHERE user_id = ? ORDER BY ts DESC",
            conn, params=(user_id,),
        )
    if not df.empty:
        df["top_keywords"] = df["top_keywords"].apply(lambda s: json.loads(s) if isinstance(s, str) and s else [])
        df["ts"] = pd.to_datetime(df["ts"])
    return df

# ---------- Danger zone ----------
def reset_user_data(user_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM entries WHERE user_id = ?", (user_id,))
        conn.commit()

def full_reset_db() -> None:
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    init_db()
