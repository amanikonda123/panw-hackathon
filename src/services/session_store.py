# src/services/session_store.py
import secrets
from typing import Dict, Any
import streamlit as st

@st.cache_resource
def _store() -> Dict[str, Any]:
    # survives reruns & page refreshes, but NOT server restarts
    return {}

def create_session(user: Dict[str, Any]) -> str:
    sid = secrets.token_urlsafe(16)
    _store()[sid] = user
    return sid

def get_session(sid: str) -> Dict[str, Any] | None:
    return _store().get(sid)

def delete_session(sid: str) -> None:
    _store().pop(sid, None)
