# src/services/calendar.py
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from googleapiclient.discovery import build
from src.services.google_auth import load_user_token

LOCAL_TZ = ZoneInfo("America/Los_Angeles")  # adjust if you want


def _iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LOCAL_TZ)
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso_to_local_dt(iso_str: Optional[str]) -> Optional[datetime]:
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.astimezone(LOCAL_TZ)
    except Exception:
        return None


def get_events_today(google_sub: str, max_results: int = 10) -> List[Dict]:
    """
    Return today's events as a list of dicts with both display strings and real datetimes:
    {
      'summary', 'start_local', 'end_local', 'location',
      'start_dt', 'end_dt'   # tz-aware datetimes in LOCAL_TZ (or None for all-day)
    }
    """
    creds = load_user_token(google_sub)
    if not creds:
        return []

    service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    now_local = datetime.now(LOCAL_TZ)
    start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    resp = service.events().list(
        calendarId="primary",
        timeMin=_iso_z(start),
        timeMax=_iso_z(end),
        singleEvents=True,
        orderBy="startTime",
        maxResults=max_results,
    ).execute()

    items: List[Dict] = []
    for ev in resp.get("items", []):
        start_iso = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date")
        end_iso = ev.get("end", {}).get("dateTime") or ev.get("end", {}).get("date")

        s_dt = _parse_iso_to_local_dt(start_iso)
        e_dt = _parse_iso_to_local_dt(end_iso)

        def _fmt(dt: Optional[datetime]) -> str:
            if dt is None:
                return "All day"
            return dt.strftime("%-I:%M %p")

        items.append({
            "summary": ev.get("summary", "(no title)"),
            "location": ev.get("location", ""),
            "start_local": _fmt(s_dt),
            "end_local": _fmt(e_dt),
            "start_dt": s_dt,
            "end_dt": e_dt,
        })
    return items


# ---------- NEW: turn today's events into useful insights ----------
def analyze_day_schedule(events: List[Dict], min_gap_min: int = 15) -> Dict:
    """
    Given today's events (from get_events_today), compute:
      - total_events
      - first_start / last_end (display strings)
      - back_to_back_blocks (count), longest_back_to_back_len, longest_window (start,end)
      - best_free_block (start,end,minutes) — biggest gap between meetings
    """
    evs = [e for e in events if e.get("start_dt") and e.get("end_dt")]
    evs.sort(key=lambda x: x["start_dt"])

    out = {
        "total_events": len(events),
        "first_start": evs[0]["start_dt"].strftime("%-I:%M %p") if evs else None,
        "last_end": evs[-1]["end_dt"].strftime("%-I:%M %p") if evs else None,
        "back_to_back_blocks": 0,
        "longest_back_to_back_len": 0,
        "longest_window": None,  # (start_str, end_str)
        "best_free_block": None, # (start_str, end_str, minutes)
    }

    if not evs:
        return out

    # back-to-back detection
    gap = timedelta(minutes=min_gap_min)
    b2b_blocks = []
    cur_block = [evs[0]]
    for prev, nxt in zip(evs, evs[1:]):
        if nxt["start_dt"] <= prev["end_dt"] + gap:
            cur_block.append(nxt)
        else:
            if len(cur_block) > 1:
                b2b_blocks.append(cur_block)
            cur_block = [nxt]
    if len(cur_block) > 1:
        b2b_blocks.append(cur_block)

    out["back_to_back_blocks"] = len(b2b_blocks)
    if b2b_blocks:
        longest = max(b2b_blocks, key=len)
        out["longest_back_to_back_len"] = len(longest)
        start = longest[0]["start_dt"].strftime("%-I:%M %p")
        end = longest[-1]["end_dt"].strftime("%-I:%M %p")
        out["longest_window"] = (start, end)

    # biggest free block
    day_start = evs[0]["start_dt"].replace(hour=8, minute=0, second=0, microsecond=0)
    day_end = evs[-1]["end_dt"].replace(hour=18, minute=0, second=0, microsecond=0)
    # build gaps: before first, between, after last
    candidates: List[Tuple[datetime, datetime]] = []
    if evs[0]["start_dt"] - day_start > timedelta(minutes=0):
        candidates.append((day_start, evs[0]["start_dt"]))
    for a, b in zip(evs, evs[1:]):
        if b["start_dt"] - a["end_dt"] > timedelta(minutes=0):
            candidates.append((a["end_dt"], b["start_dt"]))
    if day_end - evs[-1]["end_dt"] > timedelta(minutes=0):
        candidates.append((evs[-1]["end_dt"], day_end))

    if candidates:
        best = max(candidates, key=lambda x: (x[1] - x[0]))
        minutes = int((best[1] - best[0]).total_seconds() // 60)
        out["best_free_block"] = (
            best[0].strftime("%-I:%M %p"),
            best[1].strftime("%-I:%M %p"),
            minutes,
        )

    return out
