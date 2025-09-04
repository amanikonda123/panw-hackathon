#!/usr/bin/env python3
# scripts/seed_week.py
from __future__ import annotations

import argparse
import os, sys, sqlite3
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import DB_PATH
from src.services.storage import init_db, insert_entry, reset_user_data
from src.services.nlp import analyze_entry

# Student-flavored highs/lows for a week
SAMPLES_STUDENT: List[Tuple[str, int]] = [
    ("Mondays are brutal. Missed the early bus and spilled coffee before my 8am lecture. "
     "I still made it to class and took decent notes, but I felt scattered and tired all morning.", 8),
    ("Study group totally clicked today. We cracked the hardest part of the problem set after TA hours, "
     "and I finally understand dynamic programming. Walked out feeling proud and energized.", 14),
    ("Group presentation day… nerves were high and one teammate bailed last minute. "
     "I stumbled on one slide and my voice shook. I’m anxious about the grade and wish I’d practiced more.", 19),
    ("Squeezed in a quick run before lab and it cleared my head. Our CS project milestone passed all the tests! "
     "We grabbed burritos afterward and I felt genuinely proud of the progress.", 11),
    ("Part-time shift ran long and I missed the robotics club meeting. I’m exhausted, my head hurts, "
     "and I’m behind on readings. Feeling overwhelmed and a little frustrated.", 16),
    ("Movie night with friends was perfect—we laughed a ton and it felt good to unplug. "
     "Called family afterward and felt a little homesick, but mostly grateful and lighter.", 21),
    ("Sunday reset: laundry, cleaned my desk, and mapped next week’s deadlines. "
     "Prepped two easy meals and blocked focused study time. I feel calmer and ready for Monday.", 17),
]

def _get_user_id_by_email(email: str) -> Optional[int]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT id FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()
    return int(row[0]) if row else None

def _get_user_id_by_sub(sub: str) -> Optional[int]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT id FROM users WHERE google_sub = ?", (sub,))
    row = cur.fetchone()
    conn.close()
    return int(row[0]) if row else None

def _get_most_recent_user_id() -> Optional[int]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT id FROM users ORDER BY created_at DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    return int(row[0]) if row else None

def seed_week(user_id: int, days: int = 7, wipe: bool = False, end_offset_days: int = 1) -> None:
    """
    Seed `days` entries ending `end_offset_days` before now.
    By default end_offset_days=1 → last entry is *yesterday*.
    """
    if wipe:
        reset_user_data(user_id)

    init_db()  # ensure tables exist

    # the last (most recent) seeded entry will be at this base day
    base = (datetime.now() - timedelta(days=end_offset_days)).replace(minute=0, second=0, microsecond=0)

    # repeat pattern if days > samples
    data = (SAMPLES_STUDENT * ((days + len(SAMPLES_STUDENT) - 1) // len(SAMPLES_STUDENT)))[:days]

    hist_texts: List[str] = []
    for idx, (text, hour) in enumerate(data):
        # spread backward so first entry is oldest, last is exactly at `base`
        ts = (base - timedelta(days=(days - 1 - idx))).replace(hour=hour)
        analysis = analyze_entry(text, hist_texts)
        insert_entry(
            user_id=user_id,
            ts=ts.isoformat(timespec="seconds"),
            text=text,
            sent_label=analysis["sentiment_label"],
            sent_score=float(analysis["sentiment_score"]),
            top_keywords=analysis["keywords"],
        )
        hist_texts.append(text)
        print(f"[OK] {ts.isoformat(timespec='seconds')}  {analysis['sentiment_label']} "
              f"{analysis['sentiment_score']:+.2f}  kw={', '.join(analysis['keywords'][:3])}")

def main():
    ap = argparse.ArgumentParser(description="Seed mock student entries into an EXISTING user account.")
    ap.add_argument("--user-id", type=int, help="Seed entries for this user id.")
    ap.add_argument("--email", type=str, help="Or: seed for user with this email.")
    ap.add_argument("--sub", type=str, help="Or: seed for user with this Google sub.")
    ap.add_argument("--days", type=int, default=7, help="How many days to seed (default 7).")
    ap.add_argument("--wipe", action="store_true", help="Delete this user's existing entries first.")
    ap.add_argument("--end-offset-days", type=int, default=1,
                    help="How many days before today the last entry should be (default 1=yesterday; use 0 for today).")
    args = ap.parse_args()

    init_db()

    uid: Optional[int] = None
    if args.user_id:
        uid = args.user_id
    elif args.sub:
        uid = _get_user_id_by_sub(args.sub.strip())
    elif args.email:
        uid = _get_user_id_by_email(args.email.strip())
    else:
        uid = _get_most_recent_user_id()

    if not uid:
        print("No matching user found. Try one of:\n"
              "  python scripts/list_users.py\n"
              "  python scripts/seed_week.py --user-id 1 --wipe\n"
              "  python scripts/seed_week.py --email you@school.edu --wipe\n"
              "  python scripts/seed_week.py --sub <google_sub> --wipe")
        sys.exit(1)

    print(f"Seeding {args.days} day(s) into user_id={uid}  (wipe={args.wipe})  end_offset_days={args.end_offset_days}")
    seed_week(uid, days=max(1, args.days), wipe=bool(args.wipe), end_offset_days=max(0, args.end_offset_days))
    print("Done.")

if __name__ == "__main__":
    main()
