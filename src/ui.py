# src/ui.py
from __future__ import annotations

from datetime import datetime, timedelta
import base64
import json
import requests
from typing import List

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
from .services.calendar import get_events_today  # we'll import get_events_tomorrow lazily
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
    st.title("📝 Empathetic Journaling Companion")

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
    tab_write, tab_wrapped, tab_history = st.tabs(["✍️ Write", "🎧 Wrapped", "🗂️ History"])

    # ============ Write ============
    with tab_write:
        st.subheader("Dynamic, empathetic prompt")

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
        from collections import Counter
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
        ctx = {
            "goal": (user_goals.strip() if user_goals else None),
            "themes": themes,
            "mood_trend": mood_trend,
            "days_since_last": days_since_last,
            "today_event": today_event,
            "tomorrow_event": tomorrow_event,
            "keywords": keyword_candidates,  # lets the model self-validate a useful keyword
        }

        prov = "ollama"
        modl = _model()
        pkey = _prompt_key(ctx, prov, modl)

        need_new = st.session_state.get("prewrite_key") != pkey or not st.session_state.get("prewrite_text")
        if need_new:
            try:
                prompt_text = generate_prewrite_prompt_llm(ctx)
                st.session_state["prewrite_key"] = pkey
                st.session_state["prewrite_text"] = prompt_text
            except Exception as e:
                st.error(f"LLM prompt generation failed: {e}")
                # keep any previous good prompt in cache; if none, show a minimal notice
                if not st.session_state.get("prewrite_text"):
                    st.info("Prompt unavailable. Try 'Refresh prompt' after your LLM is running.")


        prompt_text = st.session_state.get("prewrite_text", "")
        if prompt_text:
            _render_prompt_box(prompt_text)

        c_dbg1, c_dbg2 = st.columns([1, 5])
        with c_dbg1:
            if st.button("Refresh prompt", use_container_width=True):
                st.session_state.pop("prewrite_key", None)
                st.session_state.pop("prewrite_text", None)
                st.rerun()
        with c_dbg2:
            _st = get_last_llm_status()
            if _st.get("path") == "llm_ok":
                st.caption(f"LLM: Ollama · {_st.get('model')}")
            elif _st.get("path") == "error":
                st.caption("LLM error; see message above.")


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

    # ============ Wrapped ============
    with tab_wrapped:
        st.subheader("Your ‘Wrapped’ for Emotions")

        # ---- Period filter ----
        period = st.segmented_control(
            "Time range",
            options=["Daily", "Weekly", "Monthly", "Yearly"],
            selection_mode="single",
            key="analytics_period",
        )
        days_map = {"Daily": 1, "Weekly": 7, "Monthly": 30, "Yearly": 365}
        window_days = days_map.get(period, 7)

        df_all = load_entries_df(user_id)
        if df_all.empty:
            st.info("No entries yet. Write a few thoughts and come back!")
            st.stop()

        # ---- Build time window & previous window ----
        now = pd.Timestamp.now()
        end = now.normalize()
        start = (end - pd.Timedelta(days=window_days - 1)).normalize()

        df_window = df_all[(df_all["ts"] >= start) & (df_all["ts"] <= end)].copy()
        df_prev = df_all[
            (df_all["ts"] >= (start - pd.Timedelta(days=window_days))) &
            (df_all["ts"] < start)
        ].copy()

        # ---- Metrics for the chosen window ----
        days_active = int(df_window["ts"].dt.date.nunique()) if not df_window.empty else 0
        total_entries_window = int(len(df_window))

        # overall streaks (meaningful to users)
        s_overall = compute_streaks(df_all)

        avg_window = float(df_window["sentiment_score"].astype(float).mean()) if not df_window.empty else 0.0
        avg_prev   = float(df_prev["sentiment_score"].astype(float).mean()) if not df_prev.empty else None

        # daily aggregates inside the window
        daily_counts = entries_by_day(df_window)       # Series[date] -> count
        daily_avgs   = avg_sentiment_by_day(df_window) # Series[date] -> avg sentiment
        daily = []
        for d in pd.date_range(start=start, end=end, freq="D"):
            dk = d.normalize()
            c = int(daily_counts.get(dk, 0)) if hasattr(daily_counts, "get") else (int(daily_counts[dk]) if dk in daily_counts.index else 0)
            a = float(daily_avgs.get(dk, 0.0)) if hasattr(daily_avgs, "get") else (float(daily_avgs[dk]) if dk in daily_avgs.index else 0.0)
            daily.append({"date": dk.strftime("%Y-%m-%d"), "count": c, "avg": a})

        # peaks within the window (best/tough day)
        best_day = tough_day = None
        if not daily_avgs.empty:
            try:
                bd = daily_avgs.idxmax()
                td = daily_avgs.idxmin()
                best_day  = {"date": pd.to_datetime(bd).strftime("%Y-%m-%d"), "avg": float(daily_avgs.max())}
                tough_day = {"date": pd.to_datetime(td).strftime("%Y-%m-%d"), "avg": float(daily_avgs.min())}
            except Exception:
                pass

        # journaling hour label (within the window)
        def _hour_label(h: int) -> str:
            if h == 0: return "12 AM"
            if h < 12: return f"{h} AM"
            if h == 12: return "12 PM"
            return f"{h-12} PM"

        best_hour_label = None
        if not df_window.empty:
            from collections import Counter
            df_window["_hour"] = df_window["ts"].dt.hour
            cnt = Counter(df_window["_hour"].tolist())
            if cnt:
                best_hour = max(cnt.items(), key=lambda x: x[1])[0]
                best_hour_label = _hour_label(int(best_hour))

        # Top themes for the window (up to 6–8)
        top_themes: list[str] = []
        try:
            kwdf_window = top_keywords_overall(df_window, k=10)  # pull a few extra, cap below
            if not kwdf_window.empty:
                if "keyword" in kwdf_window.columns:
                    top_themes = kwdf_window["keyword"].astype(str).tolist()
                else:
                    top_themes = kwdf_window.iloc[:, 0].astype(str).tolist()
        except Exception:
            pass  # keep []

        # ---- Compact payload for longer ranges (speed + reliability) ----
        def _sample_daily(daily_list: list[dict], max_points: int) -> list[dict]:
            n = len(daily_list)
            if n <= max_points:
                return daily_list
            # even sampling including first/last
            idxs = sorted({int(round(i * (n - 1) / (max_points - 1))) for i in range(max_points)})
            return [daily_list[i] for i in idxs]

        def _cap(lst: list[str], k: int) -> list[str]:
            return lst[:k] if lst else []

        max_points = 10 if period in ("Monthly", "Yearly") else 7
        daily_compact = _sample_daily(daily, max_points)
        top_themes = _cap(top_themes, 6 if period in ("Monthly", "Yearly") else 8)

        # ---- LLM summary for selected period (cached per window) ----
        period_label = period
        week_ctx = {
            "window": {"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")},
            "totals": {"entries": total_entries_window, "days_active": days_active},
            "streak": {"current": int(s_overall.current), "best": int(s_overall.best)},
            "avg_sent": {"week": avg_window, "prev_week": avg_prev},
            "best_hour_label": best_hour_label,
            "top_themes": top_themes,
            "daily": daily_compact,
            "peaks": {"best_day": best_day, "tough_day": tough_day},
        }

        cache_key = f"summary|{period}|{start.date()}|{end.date()}"
        weekly_md = st.session_state.get(cache_key)
        if not weekly_md:
            try:
                weekly_md = generate_weekly_summary_llm(week_ctx)
                st.session_state[cache_key] = weekly_md
            except Exception as e:
                weekly_md = f"_Summary unavailable (LLM error: {e})._"

        with st.container(border=True):
            st.markdown(f"### {period_label} summary")
            st.markdown(weekly_md)

        # refresh button BELOW the summary
        if st.button("Refresh summary", use_container_width=True):
            st.session_state.pop(cache_key, None)
            st.rerun()

        st.divider()

        # ---------- Analytics (scoped to selected period; keep some overall context) ----------
        df = df_window

        s = compute_streaks(df_all)  # overall streak context
        total_entries_all = int(len(df_all))
        last_7 = df_all[df_all["ts"] >= (pd.Timestamp.now() - pd.Timedelta(days=7))]
        avg7 = last_7["sentiment_score"].astype(float).mean() if not last_7.empty else 0.0

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total entries (all time)", f"{total_entries_all}")
        with m2:
            st.metric("Current streak", f"{s.current} days", help="Consecutive days ending today")
            st.caption(f"Best: **{s.best}** · Active last 30d: **{s.days_active_30}**")
        with m3:
            mood_emoji = "🙂" if avg7 >= 0.1 else ("🙁" if avg7 <= -0.1 else "😐")
            st.metric("Past 7d mood", f"{avg7:+.2f} {mood_emoji}", help="Average VADER compound score over last 7 days")

        st.divider()

        st.markdown(f"#### Entries per day ({period_label.lower()})")
        daily_overall = entries_by_day(df)
        st.line_chart(daily_overall, height=220)

        st.markdown(f"#### Average sentiment by day ({period_label.lower()})")
        daily_avg_overall = avg_sentiment_by_day(df)
        st.line_chart(daily_avg_overall, height=220)

        st.divider()

        st.markdown(f"#### Best journaling hour ({period_label.lower()})")
        # yearly might want more data; otherwise, use window
        best_hour, by_hour = best_journaling_hour(df_all if period == "Yearly" else df)
        if best_hour is not None:
            st.caption(f"You tend to write most around **{_hour_label(best_hour)}**.")

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
            st.markdown(f"#### Top themes ({period_label.lower()})")
            kwdf = top_keywords_overall(df, k=12)
            if kwdf.empty:
                st.info("Not enough keyword data yet.")
            else:
                st.dataframe(kwdf, use_container_width=True, height=260)

        with c2:
            st.markdown(f"#### Emotion mix ({period_label.lower()})")
            emos = emotion_distribution(df)
            if emos.empty:
                st.info("No sentiment yet.")
            else:
                st.bar_chart(emos, height=260)

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



    # ============ History ============
    with tab_history:
        st.subheader("Your entries")
        df = load_entries_df(user_id)
        if df.empty:
            st.info("No entries yet.")
        else:
            st.dataframe(
                df[["ts", "sentiment_label", "sentiment_score", "top_keywords", "text"]],
                use_container_width=True,
                height=420,
            )
