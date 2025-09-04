# src/services/prompts.py
from __future__ import annotations
from typing import Optional, Dict, List

def build_freewrite_prompt(
    *,
    goal: Optional[str],
    themes: List[str],
    mood_trend: str,                # "up" | "flat" | "down"
    days_since_last: Optional[int],
    today_event: Optional[Dict] = None,    # {"summary": str, "start_local": str, "end_local": str}
    tomorrow_event: Optional[Dict] = None, # same shape
    **kwargs,  # ignore any extra keys safely
) -> str:
    """
    Deterministic, calendar-aware pre-write prompt (no quotes, no therapy language).
    Produces: opener line + optional habit line + up to 2 bullet questions.
    """

    def _title(ev: Optional[Dict]) -> str:
        if not isinstance(ev, dict):
            return ""
        t = (ev.get("summary") or "").strip()
        # keep it readable
        return t if len(t) <= 80 else (t[:77] + "…")

    lines: List[str] = []
    lines.append("Take one slow breath. Set a 3–5 minute timer and free-write.")

    # Habit / recency line
    if isinstance(days_since_last, int) and days_since_last >= 2:
        plural = "s" if days_since_last != 1 else ""
        lines.append(f"It’s been **{days_since_last}** day{plural} since your last entry.")

    # Build questions (max 2)
    questions: List[str] = []

    # 1) Today’s finished event
    t_title = _title(today_event)
    if t_title:
        questions.append(f"How did **{t_title}** go?")

    # 2) Earliest event tomorrow
    tm_title = _title(tomorrow_event)
    if tm_title and len(questions) < 2:
        questions.append(f"Are you ready for **{tm_title}** tomorrow?")

    # 3) Goal-guided (if room)
    if goal and len(questions) < 2:
        g = goal.strip()
        if g:
            questions.append(f"What’s one small step toward **{g}** you could take today or tomorrow?")

    # 4) Theme-guided (if room)
    if themes and len(questions) < 2:
        questions.append(f"You’ve been writing about **{themes[0]}**—what’s one detail from the last day you want to capture?")

    # 5) Mood-trend fallback
    if len(questions) < 2:
        if mood_trend == "up":
            questions.append("What worked recently that you’d like to repeat this week?")
        elif mood_trend == "down":
            questions.append("What felt heaviest lately, and what would make it 10% lighter?")
        else:
            questions.append("What feels most important right now? Pick one thread and follow it for a few sentences.")

    lines.append("")
    for q in questions[:2]:
        lines.append(f"• {q}")

    return "\n".join(lines).strip()
