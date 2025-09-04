# src/services/analytics.py
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List
import json
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
    """Count of entries per calendar day."""
    if df.empty:
        return pd.Series(dtype=int)
    s = df.copy()
    s["date"] = s["ts"].dt.normalize()
    return s.groupby("date").size()

def avg_sentiment_by_day(df: pd.DataFrame) -> pd.Series:
    """
    TRUE daily average of VADER compound scores (continuous in [-1, 1]).
    No rounding, no sign mapping.
    """
    if df.empty:
        return pd.Series(dtype=float)
    s = df.copy()
    s["date"] = s["ts"].dt.normalize()
    # ensure numeric and drop NaNs
    s["sentiment_score"] = pd.to_numeric(s["sentiment_score"], errors="coerce")
    s = s.dropna(subset=["sentiment_score"])
    return s.groupby("date")["sentiment_score"].mean()

def best_journaling_hour(df: pd.DataFrame):
    """Unchanged: returns (best_hour_int, counts_series) scoped to df."""
    if df.empty:
        return None, pd.Series(dtype=int)
    s = df.copy()
    s["_hour"] = s["ts"].dt.hour
    counts = s.groupby("_hour").size().sort_values(ascending=False)
    best = int(counts.index[0]) if not counts.empty else None
    return best, counts

def top_keywords_overall(df: pd.DataFrame, k: int = 12) -> pd.DataFrame:
    """Unchanged: returns keywords & counts from df['top_keywords'] lists."""
    if df.empty or "top_keywords" not in df.columns:
        return pd.DataFrame(columns=["keyword", "count"])
    bag = []
    for row in df["top_keywords"].tolist():
        if isinstance(row, list):
            bag.extend([str(x).strip() for x in row if x])
        elif isinstance(row, str) and row:
            try:
                lst = json.loads(row)
                if isinstance(lst, list):
                    bag.extend([str(x).strip() for x in lst if x])
            except Exception:
                pass
    c = Counter([w for w in bag if w])
    items = c.most_common(k)
    return pd.DataFrame(items, columns=["keyword", "count"])

def emotion_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Unchanged: counts by sentiment_label in df (pos/neu/neg)."""
    if df.empty or "sentiment_label" not in df.columns:
        return pd.DataFrame({"label": [], "count": []})
    counts = df["sentiment_label"].value_counts()
    return pd.DataFrame({"label": counts.index, "count": counts.values})
