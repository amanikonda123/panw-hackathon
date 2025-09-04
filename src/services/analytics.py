# src/services/analytics.py
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Dict, Tuple

import pandas as pd


@dataclass
class Streaks:
    current: int
    best: int
    days_active_30: int


def _unique_dates(df: pd.DataFrame) -> List[date]:
    if df.empty:
        return []
    return pd.to_datetime(df["ts"]).dt.date.dropna().unique().tolist()


def compute_streaks(df: pd.DataFrame) -> Streaks:
    """
    Computes current streak (ending today), best streak overall,
    and count of active days in last 30 days.
    """
    udates = sorted(set(_unique_dates(df)))
    if not udates:
        return Streaks(current=0, best=0, days_active_30=0)

    # best streak
    best = 1
    cur = 1
    for i in range(1, len(udates)):
        if udates[i] == udates[i - 1] + timedelta(days=1):
            cur += 1
            best = max(best, cur)
        else:
            cur = 1

    # current streak (count back from today)
    today = date.today()
    dset = set(udates)
    curstreak = 0
    d = today
    while d in dset:
        curstreak += 1
        d -= timedelta(days=1)

    # active days in last 30
    window_start = today - timedelta(days=29)
    days_active_30 = sum(1 for d in udates if d >= window_start)

    return Streaks(current=curstreak, best=best, days_active_30=days_active_30)


def entries_by_day(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=int)
    d = df.copy()
    d["date"] = pd.to_datetime(d["ts"]).dt.date
    return d.groupby("date").size().rename("entries")


def avg_sentiment_by_day(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    d = df.copy()
    d["date"] = pd.to_datetime(d["ts"]).dt.date
    # map labels to -1/0/1 for a simple, readable line
    label_map = {"negative": -1, "neutral": 0, "positive": 1}
    d["sent_val"] = d["sentiment_label"].map(label_map).fillna(0)
    return d.groupby("date")["sent_val"].mean().rename("avg_sentiment")


def best_journaling_hour(df: pd.DataFrame) -> Tuple[int | None, pd.Series]:
    """
    Returns (best_hour, counts_by_hour Series). best_hour is 0-23 or None.
    """
    if df.empty:
        return None, pd.Series(dtype=int)
    d = df.copy()
    d["hour"] = pd.to_datetime(d["ts"]).dt.hour
    counts = d.groupby("hour").size().reindex(range(24), fill_value=0)
    best = int(counts.idxmax()) if counts.sum() > 0 else None
    return best, counts.rename("entries")


def top_keywords_overall(df: pd.DataFrame, k: int = 12) -> pd.DataFrame:
    """
    Flattens top_keywords lists and returns a DataFrame with columns [term, count],
    sorted by count desc.
    """
    if df.empty or "top_keywords" not in df.columns:
        return pd.DataFrame(columns=["term", "count"])
    all_terms: List[str] = []
    for row in df["top_keywords"].tolist():
        if isinstance(row, list):
            all_terms.extend([str(t).lower() for t in row if t])
        elif isinstance(row, str):
            # if stored as JSON string by mistake, be robust
            try:
                import json
                lst = json.loads(row)
                if isinstance(lst, list):
                    all_terms.extend([str(t).lower() for t in lst if t])
            except Exception:
                pass
    counter = Counter(all_terms)
    common = counter.most_common(k)
    return pd.DataFrame(common, columns=["term", "count"])


def emotion_distribution(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=int)
    d = df.copy()
    d["sentiment_label"] = d["sentiment_label"].fillna("neutral")
    counts = d["sentiment_label"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
    return counts.rename("entries")
