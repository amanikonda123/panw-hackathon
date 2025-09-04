# src/ui.py
from __future__ import annotations

from datetime import datetime, timedelta
import base64
import json
import requests
from typing import List
from collections import Counter
import hashlib

import altair as alt
import pandas as pd
import streamlit as st

from .services.storage import (
    load_entries_df,
    insert_entry,
    reset_user_data,
    full_reset_db,
    get_or_create_user_google,
)
from .services.google_auth import sign_in_with_google, save_user_token, load_user_token
from .services.session_store import create_session, get_session, delete_session
from .services.calendar import get_events_today
from .services.nlp import analyze_entry, personalized_sentiment
from .services.analytics import (
    compute_streaks,
    entries_by_day,
    avg_sentiment_by_day,
    best_journaling_hour,
    top_keywords_overall,
    emotion_distribution,
)
from .services.llm import (
    generate_prewrite_prompt_llm, 
    get_last_llm_status, 
    _model, 
    generate_weekly_summary_llm
)


# ---------- URL query param session helpers ----------
def _qp_get(key: str):
    val = st.query_params.get(key)
    if isinstance(val, (list, tuple)):
        return val[0] if val else None
    return val

def _qp_set(key: str, value: str):
    st.query_params[key] = value

def _qp_del(key: str):
    try:
        del st.query_params[key]
    except Exception:
        pass

def _get_sid() -> str | None:
    return _qp_get("sid")

def _set_sid(sid: str) -> None:
    _qp_set("sid", sid)

def _clear_sid() -> None:
    _qp_del("sid")

def _restore_user_from_sid():
    if st.session_state.get("user"):
        return
    sid = _get_sid()
    if not sid:
        return
    user = get_session(sid)
    if user:
        st.session_state.user = user
    else:
        _clear_sid()

# ---------- Prompt caching helpers (session-based, not recomputed on keystrokes) ----------
def _prompt_key(ctx: dict, provider: str, model: str) -> str:
    safe_ctx = json.dumps(ctx, sort_keys=True, ensure_ascii=False, default=str)
    return f"{provider}|{model}|{safe_ctx}"

def _render_prompt_box(prompt_text: str):
    t = (prompt_text or "").replace("\n• ", "\n- ").replace("• ", "- ")
    if "\n- " in t and "\n\n- " not in t:
        t = t.replace("\n- ", "\n\n- ", 1)
    st.markdown(
        f"""
        <div style="
          background: rgba(56, 97, 140, 0.20);
          border: 1px solid rgba(255,255,255,.08);
          padding: 12px 14px; border-radius: 10px;
          white-space: pre-wrap;
        ">
        {t}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Avatar helpers ----------
def _fetch_profile_photo_b64(google_sub: str, picture_url: str, size_px: int = 96) -> str | None:
    if not picture_url:
        return None
    url = picture_url
    if "googleusercontent.com" in url and "=s" not in url and "?sz=" not in url:
        url += ("&" if "?" in url else "?") + f"sz={size_px}"
    headers = {}
    creds = load_user_token(google_sub)
    if creds and getattr(creds, "token", None):
        headers["Authorization"] = f"Bearer {creds.token}"
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
            return base64.b64encode(r.content).decode("ascii")
    except Exception:
        pass
    return None

def _avatar_html_for_user(user: dict, diameter: int = 34) -> str:
    uname = user.get("username", "You")
    initial = (uname[:1] or "U").upper()
    sub = user.get("google_sub", "")
    pic = user.get("picture_url", "")
    b64 = _fetch_profile_photo_b64(sub, pic, size_px=128) if sub and pic else None
    if b64:
        return f'<img src="data:image/png;base64,{b64}" width="{diameter}" height="{diameter}" style="border-radius:50%; display:block;" />'
    return f'''
        <div style="width:{diameter}px;height:{diameter}px;border-radius:50%;
        display:flex;align-items:center;justify-content:center;font-weight:700;color:white;
        background:linear-gradient(135deg,#8A7AFF,#4C3FFF);border:1px solid rgba(255,255,255,.15);">
        {initial}</div>'''

# ---------- Journal save helper ----------
def _save_entry(
    user_id: int,
    text: str,
    add_time: bool,
    sent_label: str,
    sent_score: float,
    keywords: List[str],
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    final_text = f"[{now}] {text}" if add_time else text
    insert_entry(
        user_id=user_id,
        ts=now,
        text=final_text,
        sent_label=sent_label,
        sent_score=float(sent_score),
        top_keywords=keywords,
    )

# ---------- Main render ----------
def render_app():
    _restore_user_from_sid()
    st.title("📝 Ink and Insights")

    # ---------- Sidebar ----------
    with st.sidebar:
        if "user" not in st.session_state:
            st.session_state.user = None

        if st.session_state.user:
            # Account card
            user = st.session_state.user
            uname = user.get("username", "You")
            st.markdown(
                """
                <style>
                .acct-name{font-weight:600;margin:0;padding:0}
                .acct-sub{opacity:.7;font-size:12px;margin-top:2px}
                </style>
                """,
                unsafe_allow_html=True,
            )
            with st.container(border=True):
                c1, c2 = st.columns([0.2, 0.8])
                with c1:
                    st.markdown(_avatar_html_for_user(user), unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="acct-name">{uname}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="acct-sub">Signed in</div>', unsafe_allow_html=True)
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                if st.button("Log out", type="secondary", use_container_width=True):
                    sid = _get_sid()
                    if sid:
                        delete_session(sid)
                        _clear_sid()
                    st.session_state.user = None
                    # also clear the prompt cache in session
                    st.session_state.pop("prewrite_key", None)
                    st.session_state.pop("prewrite_text", None)
                    st.rerun()
        else:
            # Sign-in card
            with st.expander("🔐 Sign in", expanded=True):
                st.caption("Use your Google account to sign in. (A local browser window will open.)")
                if st.button("Sign in with Google", type="primary", use_container_width=True):
                    result, err = sign_in_with_google()
                    if err:
                        st.error(f"Google sign-in failed: {err}")
                    else:
                        idinfo = result["idinfo"]
                        sub = idinfo["sub"]
                        email = idinfo.get("email", "")
                        name = idinfo.get("name") or (email.split("@")[0] if email else "You")
                        picture = idinfo.get("picture", "")
                        save_user_token(sub, result["creds"])
                        db_user = get_or_create_user_google(sub=sub, email=email, display_name=name, picture_url=picture)
                        st.session_state.user = {
                            "id": db_user["id"],
                            "google_sub": sub,
                            "email": email,
                            "username": name,
                            "picture_url": picture,
                        }
                        sid = create_session(st.session_state.user)
                        _set_sid(sid)
                        # clear prompt cache when user changes
                        st.session_state.pop("prewrite_key", None)
                        st.session_state.pop("prewrite_text", None)
                        st.success(f"Signed in as {name}")
                        st.rerun()

        st.divider()
        enable_calendar = st.toggle("Use Google Calendar context", value=True, disabled=(st.session_state.user is None))
        st.caption("If on, today and tomorrow’s events may shape the pre-write prompt.")

        st.divider()
        user_goals = st.text_input(
            "Your current focus (optional)",
            placeholder="e.g., reduce work stress",
            disabled=(st.session_state.user is None),
        )

        # Danger zone
        st.divider()
        with st.expander("⚠️ Danger zone", expanded=False):
            if st.session_state.user:
                if st.button("Delete **my** entries", type="secondary"):
                    reset_user_data(st.session_state.user["id"])
                    # prompt will need recompute after entries cleared
                    st.session_state.pop("prewrite_key", None)
                    st.session_state.pop("prewrite_text", None)
                    st.success("All your entries were deleted.")
                    st.rerun()
            st.caption("Full reset wipes *all* accounts and entries.")
            _ = st.text_input("Type `RESET ALL` to confirm full wipe", value="", key="full_wipe_confirm")
            if st.button("Full reset (wipe everyone)", type="secondary"):
                full_reset_db()
                st.success("Database fully reset. Please sign in again.")
                sid = _get_sid()
                if sid:
                    delete_session(sid)
                    _clear_sid()
                st.session_state.user = None
                st.session_state.pop("prewrite_key", None)
                st.session_state.pop("prewrite_text", None)
                st.rerun()

    # Gate main UI
    if st.session_state.user is None:
        st.info("Sign in with Google to start journaling.")
        return

    user_id = st.session_state.user["id"]
    user_sub = st.session_state.user["google_sub"]

    # ---------- Tabs ----------
    tab_write, tab_wrapped, tab_history = st.tabs(["✍️ Write", "📈 Trends", "🗂️ History"])

    # ============ Write ============
    with tab_write:
        st.subheader("Today’s cue:")

        # (A) History data
        df_hist = load_entries_df(user_id)

        # Time since last entry
        days_since_last = None
        if not df_hist.empty:
            last_ts = pd.to_datetime(df_hist.iloc[0]["ts"])
            days_since_last = (pd.Timestamp.now() - last_ts).days

        # Mood trend: last 7d vs last 30d
        def _avg(df: pd.DataFrame) -> float | None:
            return df["sentiment_score"].astype(float).mean() if not df.empty else None

        avg7 = _avg(df_hist[df_hist["ts"] >= (pd.Timestamp.now() - pd.Timedelta(days=7))]) if not df_hist.empty else None
        avg30 = _avg(df_hist[df_hist["ts"] >= (pd.Timestamp.now() - pd.Timedelta(days=30))]) if not df_hist.empty else None
        mood_trend = "flat"
        if avg7 is not None and avg30 is not None:
            if avg7 - avg30 >= 0.05:
                mood_trend = "up"
            elif avg30 - avg7 >= 0.05:
                mood_trend = "down"

        # Recent themes (aggregate from last ~20 entries)
        themes: list[str] = []
        if not df_hist.empty:
            bag = []
            for row in df_hist["top_keywords"].head(20).tolist():  # df is DESC; head() = most recent
                if isinstance(row, list):
                    bag.extend([str(x).strip() for x in row if x])
                elif isinstance(row, str) and row:
                    try:
                        lst = json.loads(row)
                        if isinstance(lst, list):
                            bag.extend([str(x).strip() for x in lst if x])
                    except Exception:
                        pass
            if bag:
                c = Counter([w.lower() for w in bag if w])
                themes = [w for w, _ in c.most_common(3)]

        # Build ordered keyword candidates from the *last* entry, then themes
        keyword_candidates: list[str] = []
        if not df_hist.empty:
            last_kw = df_hist.iloc[0].get("top_keywords", [])
            if isinstance(last_kw, list):
                keyword_candidates.extend([str(k).strip() for k in last_kw if k])
        for t in themes:
            if t and t not in keyword_candidates:
                keyword_candidates.append(t)
        keyword_candidates = keyword_candidates[:8]

        # (B) Calendar today/tomorrow: prefer past/ongoing today, otherwise upcoming today, otherwise earliest tomorrow
        today_event = None
        tomorrow_event = None

        def _parse_hhmm(s: str):
            s = (s or "").strip()
            for fmt in ("%I:%M %p", "%I %p", "%H:%M", "%H"):
                try:
                    return datetime.strptime(s, fmt).time()
                except Exception:
                    pass
            return None

        def _with_when(ev: dict, when: str) -> dict:
            e = dict(ev) if isinstance(ev, dict) else {}
            e["when"] = when
            return e

        if enable_calendar:
            # --- Today
            events_today = get_events_today(user_sub) or []
            now_t = datetime.now().time()
            past_or_ongoing, upcoming = [], []

            for ev in events_today:
                t = _parse_hhmm(ev.get("start_local", "")) or _parse_hhmm(ev.get("end_local", ""))
                if not t:
                    continue
                if t <= now_t:
                    past_or_ongoing.append((t, ev))
                else:
                    upcoming.append((t, ev))

            if past_or_ongoing:
                past_or_ongoing.sort(key=lambda x: x[0])
                today_event = _with_when(past_or_ongoing[-1][1], "past_today")
            elif upcoming:
                upcoming.sort(key=lambda x: x[0])
                today_event = _with_when(upcoming[0][1], "later_today")

            # --- Tomorrow
            try:
                from .services.calendar import get_events_tomorrow
                events_tomorrow = get_events_tomorrow(user_sub) or []
            except Exception:
                events_tomorrow = []
            tom = []
            for ev in events_tomorrow:
                t = _parse_hhmm(ev.get("start_local", ""))
                if t:
                    tom.append((t, ev))
            if tom:
                tom.sort(key=lambda x: x[0])
                tomorrow_event = _with_when(tom[0][1], "tomorrow")


        # (C) Build context for LLM, but DO NOT regenerate on keystrokes
        # (C) Build the LLM-powered pre-write prompt (cached by context)
        ctx = {
            "goal": (user_goals.strip() if user_goals else None),
            "themes": themes,
            "mood_trend": mood_trend,
            "days_since_last": days_since_last,
            "today_event": today_event,
            "tomorrow_event": tomorrow_event,
        }

        # cache key based on context
        ctx_for_key = {
            "goal": ctx["goal"],
            "themes": ctx["themes"],
            "mood_trend": ctx["mood_trend"],
            "days_since_last": ctx["days_since_last"],
            "event": (
                {"summary": today_event.get("summary",""), "when": today_event.get("when","past_today")}
                if today_event else
                ({"summary": tomorrow_event.get("summary",""), "when": "tomorrow"} if tomorrow_event else None)
            ),
        }
        key = "prewrite|" + hashlib.sha1(
            json.dumps(ctx_for_key, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

        prompt_text = st.session_state.get(key)
        if not prompt_text:
            with st.spinner("Thinking up a gentle prompt…"):
                prompt_text = generate_prewrite_prompt_llm(ctx)
            st.session_state[key] = prompt_text

        st.info(prompt_text)


        # (D) Editor + controls
        entry = st.text_area(
            "Write freely.",
            height=200,
            placeholder="How are you feeling? What happened today? What do you want to remember?",
            key="entry_text",
        )
        colA, colB = st.columns([1, 1])
        with colA:
            quick_mood = st.segmented_control(
                "Quick mood",
                options=["🙂", "😐", "🙁", "😴", "😤", "💪"],
                selection_mode="single",
            )
        with colB:
            add_time = st.checkbox("Add timestamp to entry", value=True)

        # (E) Save entry (on-device NLP analysis + personalized baseline)
        if st.button("Save entry", type="primary", disabled=not entry and not quick_mood):
            text_to_save = entry if entry.strip() else f"Quick mood: {quick_mood}"

            hist_texts, hist_scores = [], []
            if not df_hist.empty:
                df_sorted = df_hist.sort_values("ts")  # older→newer for analysis
                hist_texts = [
                    (t.split("]", 1)[-1].strip() if isinstance(t, str) and t.startswith("[") else t)
                    for t in df_sorted["text"].tail(20).tolist()
                ]
                hist_scores = df_sorted["sentiment_score"].astype(float).tolist()

            analysis = analyze_entry(text_to_save, hist_texts)
            p = personalized_sentiment(
                current_compound=analysis["sentiment_score"],
                history_compounds=hist_scores,
                window=30,
                z_pos=0.5,
                z_neg=-0.5,
            )

            _save_entry(
                user_id=user_id,
                text=text_to_save,
                add_time=add_time,
                sent_label=analysis["sentiment_label"],
                sent_score=analysis["sentiment_score"],
                keywords=analysis["keywords"],
            )

            st.success(
                f"Saved! ✨  (global: {analysis['sentiment_label']} · {analysis['sentiment_score']:+.2f} · "
                f"personalized z: {p['z']:+.2f} → {p['label']})"
            )
            if p["n_history"] < 5:
                st.caption("Tip: write a few more entries to calibrate your personal baseline.")
            if analysis["keywords"]:
                st.caption("Themes: " + ", ".join(analysis["keywords"]))

            # Invalidate the prompt so next render reflects new history/themes
            st.session_state.pop("prewrite_key", None)
            st.session_state.pop("prewrite_text", None)
            st.rerun()

    # ============ Wrapped (Weekly only) ============
    with tab_wrapped:
        st.subheader("Your Weekly Insights")

        # ---- Load data ----
        df_all = load_entries_df(user_id)
        if df_all.empty:
            st.info("No entries yet. Write a few thoughts and come back!")
            st.stop()

        # ---- Weekly window (last 7 days, inclusive of today) ----
        today = pd.Timestamp.now().normalize()               # e.g., 2025-09-04 00:00
        start = (today - pd.Timedelta(days=6)).normalize()   # 7-day window start
        end_excl = (today + pd.Timedelta(days=1)).normalize()# exclusive upper bound (tomorrow 00:00)

        df_window = df_all[(df_all["ts"] >= start) & (df_all["ts"] < end_excl)].copy()
        df_prev   = df_all[(df_all["ts"] >= (start - pd.Timedelta(days=7))) &
                        (df_all["ts"] < start)].copy()


        if df_window.empty:
            st.info(f"No entries in the last 7 days ({start.date()} → {end_excl.date()}).")
            st.stop()

        # All-time streak stats (use the full dataset)
        s_all = compute_streaks(df_all)   # -> has .current, .best, .days_active_30
        best_streak = int(getattr(s_all, "best", 0))
        current_streak = int(getattr(s_all, "current", 0))
        active_30 = int(getattr(s_all, "days_active_30", 0))

        # --- helpers ---
        def _hour_label(h: int) -> str:
            if h == 0: return "12 AM"
            if h < 12: return f"{h} AM"
            if h == 12: return "12 PM"
            return f"{h-12} PM"

        def compound_to_pct(x: float) -> int:
            try:
                return int(round((float(x) + 1.0) * 50))  # [-1,1] -> [0,100]
            except Exception:
                return 50

        def compound_to_band(x: float):
            x = float(x)
            if x <= -0.60:     return ("Very negative", "🙁")
            if x <= -0.05:     return ("Negative", "😕")
            if x <  0.05:      return ("Neutral", "😐")
            if x <  0.60:      return ("Positive", "🙂")
            return ("Very positive", "😊")

        # ---- Weekly metrics (scoped to df_window) ----
        days_active = int(df_window["ts"].dt.date.nunique())
        total_entries_week = int(len(df_window))
        avg_week = float(df_window["sentiment_score"].astype(float).mean())
        avg_prev = float(df_prev["sentiment_score"].astype(float).mean()) if not df_prev.empty else None

        # daily aggregates
        daily_counts = entries_by_day(df_window)       # Series[date] -> count
        daily_avgs   = avg_sentiment_by_day(df_window) # Series[date] -> avg compound
        daily = []
        for d in pd.date_range(start=start, end=end_excl, freq="D"):
            dk = d.normalize()
            c = int(daily_counts.get(dk, 0)) if hasattr(daily_counts, "get") else (int(daily_counts[dk]) if dk in daily_counts.index else 0)
            a = float(daily_avgs.get(dk, 0.0)) if hasattr(daily_avgs, "get") else (float(daily_avgs[dk]) if dk in daily_avgs.index else 0.0)
            daily.append({"date": dk.strftime("%Y-%m-%d"), "count": c, "avg": a})

        # peaks inside week
        best_day = tough_day = None
        if not daily_avgs.empty:
            try:
                bd = daily_avgs.idxmax()
                td = daily_avgs.idxmin()
                best_day  = {"date": pd.to_datetime(bd).strftime("%Y-%m-%d"), "avg": float(daily_avgs.max())}
                tough_day = {"date": pd.to_datetime(td).strftime("%Y-%m-%d"), "avg": float(daily_avgs.min())}
            except Exception:
                pass

        # best journaling hour (within week)
        best_hour_label = None
        df_window["_hour"] = df_window["ts"].dt.hour
        cnt = Counter(df_window["_hour"].tolist())
        if cnt:
            best_hour = max(cnt.items(), key=lambda x: x[1])[0]
            best_hour_label = _hour_label(int(best_hour))

        # top themes (within week)
        top_themes: list[str] = []
        try:
            kwdf_window = top_keywords_overall(df_window, k=10)
            if not kwdf_window.empty:
                if "keyword" in kwdf_window.columns:
                    top_themes = kwdf_window["keyword"].astype(str).tolist()
                else:
                    top_themes = kwdf_window.iloc[:, 0].astype(str).tolist()
            top_themes = top_themes[:8]
        except Exception:
            pass

        # convert to 0–100 once
        avg_pct  = compound_to_pct(avg_week)
        prev_pct = compound_to_pct(avg_prev) if avg_prev is not None else None

        # daily: keep date & count, convert avg→mood_pct
        daily_pct = [
            {"date": d["date"], "count": int(d["count"]), "mood_pct": compound_to_pct(d["avg"])}
            for d in daily
        ]

        # peaks: convert avg→mood_pct
        peaks_pct = {
            "best_day": ({"date": best_day["date"], "mood_pct": compound_to_pct(best_day["avg"])} if best_day else None),
            "tough_day": ({"date": tough_day["date"], "mood_pct": compound_to_pct(tough_day["avg"])} if tough_day else None),
        }

        # build the context WITHOUT raw -1..1 values
        week_ctx = {
            "period": "weekly",
            "window": {"start": start.strftime("%Y-%m-%d"), "end": end_excl.strftime("%Y-%m-%d")},
            "totals": {"entries": total_entries_week, "days_active": days_active},
            "avg_mood": {"period_pct": avg_pct, "prev_pct": prev_pct},
            "best_hour_label": best_hour_label,
            "top_themes": top_themes,
            "daily_mood": daily_pct,    # <-- use this
            "peaks_mood": peaks_pct,    # <-- and this
        }


        cache_key = f"summary|weekly|{start.date()}|{end_excl.date()}"
        weekly_md = st.session_state.get(cache_key)

        if weekly_md is None:
            gen = st.button("Generate weekly summary", use_container_width=True)
            if gen:
                with st.spinner("Summarizing your week…"):
                    try:
                        weekly_md = generate_weekly_summary_llm(week_ctx)
                        st.session_state[cache_key] = weekly_md
                    except Exception as e:
                        weekly_md = f"_Summary unavailable (LLM error: {e})._"

        if weekly_md is not None:
            with st.container(border=True):
                st.markdown("### Weekly summary")
                st.markdown(weekly_md)

            if st.button("Refresh weekly summary", use_container_width=True, key=f"refresh_{cache_key}"):
                with st.spinner("Refreshing summary…"):
                    st.session_state.pop(cache_key, None)
                    st.rerun()
        else:
            st.caption("Tip: Generate a weekly summary when you’re ready. It’s cached for this week.")


        st.divider()

        # ---------- Weekly analytics (ALL scoped to df_window) ----------
        avg_pct = compound_to_pct(avg_week)
        label, emoji = compound_to_band(avg_week)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total entries (weekly)", f"{total_entries_week}")
        with m2:
            st.metric("Days active (weekly)", f"{days_active}")
        with m3:
            st.metric(
                "Mood (weekly)",
                f"{avg_pct} {emoji}",
                help="VADER avg mapped to 0–100 (50 = neutral)"
            )
        with m4:
            st.metric("Best streak (all time)", f"{best_streak} days", help="Longest daily writing streak")

        st.caption(f"Current streak: **{current_streak}** days · Active last 30d: **{active_30}** days")

        st.divider()

        st.markdown("#### Entries per day (weekly)")
        daily_overall = entries_by_day(df_window)
        st.line_chart(daily_overall, height=220)

        st.markdown("#### Average sentiment by day (weekly)")
        daily_avg_overall = avg_sentiment_by_day(df_window).reset_index()
        daily_avg_overall.columns = ["date", "compound"]
        daily_avg_overall["mood_pct"] = daily_avg_overall["compound"].apply(compound_to_pct)

        line = (
            alt.Chart(daily_avg_overall)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("mood_pct:Q", title="Mood (0–100)", scale=alt.Scale(domain=[0, 100])),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("compound:Q", title="VADER", format="+.2f"),
                    alt.Tooltip("mood_pct:Q", title="Mood (0–100)"),
                ],
            )
            .properties(height=220)
        )
        neutral_rule = alt.Chart(pd.DataFrame({"y": [50]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
        st.altair_chart(line + neutral_rule, use_container_width=True)

        st.divider()

        st.markdown("#### Best journaling hour (weekly)")
        bh, by_hour = best_journaling_hour(df_window)
        if bh is not None:
            st.caption(f"You tend to write most around **{_hour_label(bh)}**.")
        hour_df = pd.DataFrame({"hour": [int(h) for h in by_hour.index], "entries": by_hour.values})
        hour_df["label"] = hour_df["hour"].apply(_hour_label)
        hour_chart = (
            alt.Chart(hour_df)
            .mark_bar()
            .encode(
                x=alt.X("label:N", sort=None, axis=alt.Axis(title="Hour", labelAngle=-90, labelAlign="right")),
                y=alt.Y("entries:Q", title="Entries"),
                tooltip=["label:N", "entries:Q"],
            )
            .properties(height=280)
        )
        st.altair_chart(hour_chart, use_container_width=True)

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Top themes (weekly)")
            kwdf = top_keywords_overall(df_window, k=12)
            if kwdf.empty:
                st.info("Not enough keyword data yet.")
            else:
                st.dataframe(kwdf, use_container_width=True, height=260)

        with c2:
            st.markdown("#### Emotion mix (weekly)")

            emos = emotion_distribution(df_window)
            if emos.empty:
                st.info("No sentiment yet.")
            else:
                # Normalize to a tidy df: label, count
                if isinstance(emos, pd.Series):
                    emos_df = emos.reset_index()
                    emos_df.columns = ["label", "count"]
                else:
                    # expect columns like ["label","count"] or ["sentiment","count"]
                    if "label" not in emos.columns:
                        emos_df = emos.rename(columns={"sentiment": "label"})[["label", "count"]].copy()
                    else:
                        emos_df = emos[["label", "count"]].copy()

                # Order + percent
                order = ["positive", "neutral", "negative"]  # or ["negative","neutral","positive"]
                total = int(emos_df["count"].sum())
                if total == 0:
                    st.info("No sentiment yet.")
                else:
                    emos_df["pct"] = (emos_df["count"] / total * 100).round(0)
                    emos_df["label_txt"] = emos_df.apply(lambda r: f"{int(r['count'])}  ({int(r['pct'])}%)", axis=1)

                    # Consistent, readable colors
                    palette = {"negative": "#ef4444", "neutral": "#9ca3af", "positive": "#22c55e"}

                    bars = (
                        alt.Chart(emos_df)
                        .mark_bar()
                        .encode(
                            y=alt.Y("label:N", sort=order, title=None),
                            x=alt.X("count:Q", title="Entries", scale=alt.Scale(domainMin=0, nice=True)),
                            color=alt.Color("label:N",
                                            scale=alt.Scale(domain=list(palette.keys()), range=list(palette.values())),
                                            legend=None),
                            tooltip=[
                                alt.Tooltip("label:N", title="Sentiment"),
                                alt.Tooltip("count:Q", title="Entries"),
                                alt.Tooltip("pct:Q", title="Share (%)"),
                            ],
                        )
                        .properties(height=220)
                    )

                    labels = (
                        bars.mark_text(
                            align="left",
                            baseline="middle",
                            dx=6,
                        )
                        .encode(text="label_txt:N")
                    )

                    st.altair_chart(bars + labels, use_container_width=True)


        st.divider()

        st.markdown("#### Export your data (all time)")
        out = df_all.copy()
        out["top_keywords"] = out["top_keywords"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else (x or "")
        )
        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="journal_export.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ============ History (Calendar view) ============
    with tab_history:
        st.subheader("Browse your history by calendar")

        df = load_entries_df(user_id)
        if df.empty:
            st.info("No entries yet.")
            st.stop()

        # Helpers
        def _compound_to_pct(x: float) -> int:
            try:
                return int(round((float(x) + 1.0) * 50))  # [-1,1] -> [0,100]
            except Exception:
                return 50

        def _sentiment_badge(score: float) -> str:
            pct = _compound_to_pct(score)
            if score <= -0.60:  label, color = "very negative", "#ef4444"
            elif score <= -0.05: label, color = "negative", "#f97316"
            elif score < 0.05:   label, color = "neutral", "#9ca3af"
            elif score < 0.60:   label, color = "positive", "#22c55e"
            else:                label, color = "very positive", "#16a34a"
            return f"<span style='background:{color}22;border:1px solid {color};color:{color};padding:2px 8px;border-radius:12px;font-size:12px;'>{label} · {pct}</span>"

        def _chip(word: str) -> str:
            return f"<span style='display:inline-block;margin:2px 6px 2px 0;padding:2px 8px;border-radius:12px;border:1px solid #3b3b3b;background:#262626;font-size:12px;'>{word}</span>"

        def _set_hist_date(d):
            st.session_state.history_date = d

        # Prepare dates
        df = df.copy()
        df["date"] = df["ts"].dt.date
        df["time"] = df["ts"].dt.strftime("%I:%M %p")
        min_d = df["date"].min()
        max_d = df["date"].max()

        # Calendar picker (defaults to most recent date you wrote)
        default_date = st.session_state.get("history_date", max_d)
        sel_date = st.date_input(
            "Pick a day",
            value=default_date,
            min_value=min_d,
            max_value=max_d,
            key="history_date",
        )

        # Prev / Next day shortcuts (safe against edges)
        unique_days = sorted(df["date"].unique())

        # If the selected date isn't in the list (e.g., changed via calendar),
        # snap to the nearest valid writing day (latest).
        if sel_date not in unique_days:
            if unique_days:
                sel_date = unique_days[-1]
                st.session_state.history_date = sel_date
            else:
                unique_days = []  # no writing days at all (shouldn't happen since df not empty)

        idx = unique_days.index(sel_date) if unique_days else 0

        prev_target = unique_days[idx - 1] if idx > 0 else None
        next_target = unique_days[idx + 1] if unique_days and idx < len(unique_days) - 1 else None

        cprev, cspace, cnext = st.columns([1, 6, 1])
        with cprev:
            if prev_target is not None:
                st.button("◀ Prev", on_click=_set_hist_date, args=(prev_target,), use_container_width=True)
            else:
                st.button("◀ Prev", disabled=True, use_container_width=True)

        with cnext:
            if next_target is not None:
                st.button("Next ▶", on_click=_set_hist_date, args=(next_target,), use_container_width=True)
            else:
                st.button("Next ▶", disabled=True, use_container_width=True)


        st.divider()

        # Entries for the selected day
        day_df = df[df["date"] == sel_date].sort_values("ts")
        if day_df.empty:
            st.info("No entries for this day.")
        else:
            st.markdown(f"### {sel_date.strftime('%A, %b %d, %Y')}")
            for _, row in day_df.iterrows():
                text = row["text"]
                # If you stored "[timestamp] actual text", strip the bracketed prefix for display
                if isinstance(text, str) and text.startswith("[") and "]" in text[:30]:
                    text = text.split("]", 1)[-1].strip()

                with st.container(border=True):
                    top = st.columns([1, 3, 2])
                    with top[0]:
                        st.caption(row["time"])
                    with top[1]:
                        st.markdown(_sentiment_badge(float(row["sentiment_score"])), unsafe_allow_html=True)
                    with top[2]:
                        # keywords
                        kws = row["top_keywords"] if isinstance(row["top_keywords"], list) else []
                        if not kws and isinstance(row["top_keywords"], str):
                            try:
                                import json as _json
                                tmp = _json.loads(row["top_keywords"])
                                if isinstance(tmp, list):
                                    kws = tmp
                            except Exception:
                                pass
                        if kws:
                            st.markdown("".join(_chip(w) for w in kws[:10]), unsafe_allow_html=True)

                    st.markdown(text)

        st.caption("Tip: use the calendar to jump to any day you wrote. Prev/Next quickly hops between writing days.")
