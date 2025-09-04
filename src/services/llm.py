# src/services/llm.py
from __future__ import annotations

import os
import json
import re
from typing import Dict, Any, Optional, List

import requests

# ---- Config (Ollama only) ----
def _ollama_url() -> str:
    return os.getenv("OLLAMA_URL") or "http://127.0.0.1:11434"

def _model() -> str:
    # e.g., "qwen2.5:3b-instruct" or your preferred local tag
    return os.getenv("LLM_MODEL") or "qwen2.5:3b-instruct"

# ---- Status for UI debug (no fallback; just ok/error) ----
_LAST_STATUS: dict = {}

def get_last_llm_status() -> dict:
    return _LAST_STATUS or {"path": "unknown"}

# ---- System prompt (event-first; generic keyword rules) ----
_PREWRITE_SYSTEM = (
    "You are a warm, concise journaling companion.\n"
    "Task: generate a short pre-write prompt BEFORE the user types anything today.\n"
    "Context fields:\n"
    "  - goal (optional)\n"
    "  - themes (0–3 strings)\n"
    "  - mood_trend (up/flat/down)\n"
    "  - days_since_last (int or null)\n"
    "  - today_event / tomorrow_event (optional)\n"
    "  - keywords (ordered list; may be empty)\n"
    "\n"
    "EVENT PRIORITY:\n"
    "  If ANY event is provided, include EXACTLY ONE event question as the FIRST bullet.\n"
    "  Prefer 'past_today'; else 'later_today'; else 'tomorrow'.\n"
    "  Forms:\n"
    "    past_today  →  How did <summary> go?\n"
    "    later_today →  What would help you feel ready for <summary> later today?\n"
    "    tomorrow    →  Are you ready for <summary> tomorrow?\n"
    "  Use <summary> exactly; do not add times; no quotes.\n"
    "\n"
    "KEYWORDS (single pass):\n"
    "  Consider keywords IN ORDER; pick the FIRST that supports a concrete, sensible question.\n"
    "  If a keyword is vague, SKIP it. If it's a single bare word, rephrase into a natural topic\n"
    "  (e.g., progress with <Word>). Use at most ONE keyword-derived question.\n"
    "\n"
    "Do not invent details. Do not reference any current draft text.\n"
    "Output (STRICT JSON): {\"opener\": str, \"habit\": str|null, \"questions\": [str, ...]}\n"
    "Rules:\n"
    "1) Keep to 3–4 lines total when rendered.\n"
    "2) Opener: short and neutral, e.g., Set a 3–5 min timer and free-write.\n"
    "3) Habit line only if days_since_last ≥ 2, e.g., It’s been 3 days since your last entry. What have you been up to?\n"
    "4) Questions (1–2 bullets, ~18 words max each):\n"
    "   - If event exists, the FIRST bullet is the event question (per EVENT PRIORITY).\n"
    "   - Then ONE personal question (keyword/goal/theme/mood as needed).\n"
    "5) Style guardrails: no quotes, no exclamation marks, no therapy/medical advice, no generic filler.\n"
    "Return only the JSON object—nothing else."
)

_WEEKLY_SYSTEM = (
    "You are an insightful, concise journaling analyst.\n"
    "Task: Write a short weekly summary for the user **only** from the provided metrics.\n"
    "Goals: 3–5 bullet points max + 1 small suggestion. Neutral, supportive. No exclamation marks, no therapy advice.\n"
    "If entries < 3, acknowledge the light data and keep it extra brief.\n"
    "Return Markdown (bullets ok). Do not ask questions; summarize and suggest.\n"
)

# ---- Helpers ----
def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Turn model output into JSON dict (unwrap fences, fix trailing commas/smart quotes)."""
    def load_try(s: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    if not isinstance(text, str) or not text.strip():
        return None
    s = text.strip()

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.S | re.I)
    if fence:
        s = fence.group(1).strip()

    obj = load_try(s)
    if obj:
        return obj

    block = re.search(r"\{[\s\S]*\}", s)
    if block:
        s = block.group(0)

    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    if "'" in s and '"' not in s[: min(80, len(s))]:
        s = re.sub(r"'", '"', s)

    return load_try(s)

def _naturalize_topic(word: str) -> str:
    if not word:
        return word
    if " " in word.strip():
        return word.strip()
    return f"progress with {word.strip().capitalize()}"

def _sanitize_questions(qs: List[str]) -> List[str]:
    out = []
    for q in qs:
        if not isinstance(q, str):
            continue
        s = q.strip()
        s = re.sub(r"[\"“”‘’']([^\"“”‘’']+)[\"“”‘’']", r"\1", s)  # remove quotes
        # toward/for/about X  -> naturalize single-word X
        def repl(m):
            prep, token = m.group(1), m.group(2)
            if " " not in token and re.match(r"^[A-Za-z0-9][\w\-]*$", token):
                topic = _naturalize_topic(token)
                return f"{prep} {topic}"
            return m.group(0)
        s = re.sub(r"\b(toward|towards|for|about)\s+([A-Za-z0-9][\w\-]+)\b", repl, s, flags=re.I)
        s = re.sub(
            r"(toward|towards|for|about)\s+([A-Za-z0-9][\w\-]+)(\s*\?)$",
            lambda m: f"{m.group(1)} {_naturalize_topic(m.group(2))}{m.group(3)}",
            s, flags=re.I,
        )
        out.append(s)
    return out

def _slim_event(ev: Optional[Dict[str, Any]], default_when: str = "") -> Optional[Dict[str, str]]:
    if not isinstance(ev, dict):
        return None
    return {
        "summary": (ev.get("summary") or "").strip(),
        "when": (ev.get("when") or default_when).strip(),
    }

def _ollama_generate(prompt: str) -> str:
    r = requests.post(
        f"{_ollama_url()}/api/generate",
        json={
            "model": _model(),
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.4,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_ctx": 4096,
            },
        },
        timeout=90,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("response", "") if isinstance(data, dict) else ""

def _ollama_generate_text(prompt: str, timeout_s: int = 90, num_predict: int = 200) -> str:
    """
    Freeform text generation with a small token budget and a retry that
    both extends the timeout and shrinks the target length if needed.
    """
    url = f"{_ollama_url()}/api/generate"

    def _call(tp: int, npred: int):
        r = requests.post(
            url,
            json={
                "model": _model(),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "num_ctx": 4096,
                    "num_predict": npred,   # cap output length
                },
            },
            timeout=tp,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", "") if isinstance(data, dict) else ""

    try:
        return _call(timeout_s, num_predict)
    except requests.exceptions.ReadTimeout:
        # Retry once: longer timeout, shorter output
        return _call(timeout_s * 2, max(120, int(num_predict * 0.7)))



# ---- Public: LLM-only prewrite generator ----
def generate_prewrite_prompt_llm(ctx: Dict[str, Any]) -> str:
    """
    Build an LLM-phrased pre-write prompt.
    Expects ctx fields: goal, themes, mood_trend, days_since_last, today_event, tomorrow_event, keywords.
    No fallback; raises on error so the UI can surface it.
    """
    status = {"provider": "ollama", "model": _model(), "path": "", "reason": ""}
    # choose a single event preference: past_today > later_today > tomorrow
    ev = None
    te = ctx.get("today_event")
    to = ctx.get("tomorrow_event")
    if isinstance(te, dict) and te.get("when") == "past_today":
        ev = te
    elif isinstance(te, dict) and te.get("when") == "later_today":
        ev = te
    elif isinstance(to, dict):
        ev = to

    payload = {
        "goal": ctx.get("goal") or "",
        "themes": ctx.get("themes") or [],
        "mood_trend": ctx.get("mood_trend") or "flat",
        "days_since_last": ctx.get("days_since_last"),
        "event": _slim_event(ev) or None,
        "keywords": ctx.get("keywords") or [],
    }
    user_json = json.dumps(payload, ensure_ascii=False, indent=2)

    raw = _ollama_generate(f"{_PREWRITE_SYSTEM}\n\nCONTEXT:\n{user_json}\n")
    data = _extract_json(raw)
    if not isinstance(data, dict):
        _LAST_STATUS.update({"path": "error", "reason": "invalid JSON from LLM", "raw_sample": (raw or "")[:180]})
        raise RuntimeError("LLM returned invalid JSON")

    opener = (data.get("opener") or "").strip()
    habit  = (data.get("habit") or "").strip()
    qs = data.get("questions")
    if not opener:
        _LAST_STATUS.update({"path": "error", "reason": "missing opener"})
        raise RuntimeError("LLM response missing 'opener'")
    questions: List[str] = []
    if isinstance(qs, list):
        questions = [str(q).strip() for q in qs if str(q).strip()]
    elif isinstance(qs, str) and qs.strip():
        questions = [qs.strip()]
    questions = _sanitize_questions(questions)[:2]

    lines = [opener]
    if habit:
        lines.extend(["", habit])
    if questions:
        lines.append("")
        lines.extend([f"- {q}" for q in questions])

    _LAST_STATUS.update({"path": "llm_ok", "model": _model(), "raw_sample": (raw or "")[:180]})
    return "\n".join(lines).strip()

def generate_weekly_summary_llm(ctx: Dict[str, Any]) -> str:
    """
    ctx expects:
      {
        "window": {"start":"YYYY-MM-DD","end":"YYYY-MM-DD"},
        "totals": {"entries": int, "days_active": int},
        "streak": {"current": int, "best": int},
        "avg_sent": {"week": float, "prev_week": float | None},
        "best_hour_label": str | None,   # e.g., "9 PM"
        "top_themes": [str, ...],        # up to ~6
        "daily": [{"date":"YYYY-MM-DD","count":int,"avg":float}, ...],  # last 7 days
        "peaks": {"best_day": {"date":..., "avg":...} | None,
                  "tough_day":{"date":..., "avg":...} | None}
      }
    """
    payload = json.dumps(ctx, ensure_ascii=False, indent=2)
    prompt = f"{_WEEKLY_SYSTEM}\n\nWEEK METRICS (JSON):\n{payload}\n"
    text = _ollama_generate_text(prompt).strip()
    return text or "No summary available."