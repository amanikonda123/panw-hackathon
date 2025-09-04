# src/services/music.py
from __future__ import annotations
from typing import List, Dict
from urllib.parse import quote

# --- helpers ---
def _links_for(query: str) -> Dict[str, str]:
    q = quote(query)
    return {
        "Spotify": f"https://open.spotify.com/search/{q}",
        "Apple Music": f"https://music.apple.com/us/search?term={q}",
        "YouTube Music": f"https://music.youtube.com/search?q={q}",
    }

# --- rule engine: map emotion → bucket ---
EMO_KEYWORDS = {
    "anger": {"angry", "mad", "furious", "rage", "irritated", "annoyed"},
    "stress": {"deadline", "overwhelmed", "anxious", "stress", "pressure"},
    "sad": {"sad", "down", "lonely", "upset", "depressed", "blue"},
    "sleep": {"sleep", "tired", "exhausted", "insomnia", "rest"},
    "focus": {"study", "focus", "deep work", "concentrate", "code"},
    "hype": {"gym", "run", "workout", "pump", "energized"},
}

def pick_bucket(
    sent_label: str,
    sent_score: float,
    keywords: List[str],
    quick_mood_emoji: str | None = None,
) -> str:
    # emoji first (explicit user intent)
    emoji_map = {
        "🙂": "uplift",
        "😐": "focus",
        "🙁": "calm",
        "😴": "sleep",
        "😤": "catharsis",
        "💪": "confidence",
    }
    if quick_mood_emoji in emoji_map:
        return emoji_map[quick_mood_emoji]

    kws = {k.lower() for k in (keywords or [])}
    if kws & EMO_KEYWORDS["sleep"]:
        return "sleep"
    if kws & EMO_KEYWORDS["focus"]:
        return "focus"
    if kws & EMO_KEYWORDS["hype"]:
        return "confidence"
    if kws & EMO_KEYWORDS["anger"]:
        return "catharsis"
    if kws & EMO_KEYWORDS["stress"] or sent_score <= -0.25:
        return "calm"
    if sent_score >= 0.25 or sent_label == "positive":
        return "uplift"
    return "focus"

def bucket_suggestions(bucket: str, keywords: List[str], goal: str | None) -> Dict:
    # base queries by bucket
    BASE = {
        "calm": ["lofi beats", "ambient calm", "piano chill", "nature sounds", "breathing music 60 bpm"],
        "focus": ["deep focus", "instrumental hip hop", "coding music", "alpha waves focus"],
        "uplift": ["feel good pop", "happy indie", "sunny vibes", "good mood mix"],
        "catharsis": ["hard rock catharsis", "post-hardcore release", "alt rock anthems", "aggressive rap workout"],
        "sleep": ["sleep ambient", "rain sounds sleep", "piano for sleep", "delta waves"],
        "confidence": ["power workout", "motivational anthems", "confidence boost", "epic instrumental"],
    }
    # include a keyword-flavored query
    kwq = None
    if keywords:
        kwq = f"{keywords[0]} {' '.join(keywords[1:3])} mix".strip()
    queries = (BASE.get(bucket, [])[:3]) + ([kwq] if kwq else [])

    # rationale
    r_map = {
        "calm": "Soothing, low-stimulus tracks to lower arousal and ease stress.",
        "focus": "Steady, low-lyric textures that support sustained attention.",
        "uplift": "Major-key, higher-energy songs to nudge mood upward.",
        "catharsis": "High-energy release to process anger, then reset.",
        "sleep": "Slow, predictable textures to encourage winding down.",
        "confidence": "Driving rhythms and anthems to boost agency.",
    }
    if goal:
        r_map[bucket] += f" Tied to your goal: *{goal}*."

    items = [{"title": q.title(), "links": _links_for(q)} for q in queries if q]
    return {"bucket": bucket, "rationale": r_map[bucket], "items": items}

def suggest_music(
    sent_label: str,
    sent_score: float,
    keywords: List[str],
    quick_mood_emoji: str | None,
    user_goal: str | None,
) -> Dict:
    bucket = pick_bucket(sent_label, sent_score, keywords, quick_mood_emoji)
    return bucket_suggestions(bucket, keywords, user_goal)
