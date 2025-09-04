# src/services/google_auth.py
import os
from typing import Optional, Dict, Any, Tuple

from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

from src.config import SCOPES, CLIENT_SECRET_PATH, TOKENS_DIR

os.makedirs(TOKENS_DIR, exist_ok=True)

def _token_path_for_sub(sub: str) -> str:
    return os.path.join(TOKENS_DIR, f"{sub}.json")

def save_user_token(sub: str, creds: Credentials) -> None:
    with open(_token_path_for_sub(sub), "w") as f:
        f.write(creds.to_json())

def load_user_token(sub: str) -> Optional[Credentials]:
    path = _token_path_for_sub(sub)
    if not os.path.exists(path):
        return None
    creds = Credentials.from_authorized_user_file(path, SCOPES)
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            save_user_token(sub, creds)
        except Exception:
            return None
    return creds

def sign_in_with_google() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Local OAuth flow → returns ({"creds": Credentials, "idinfo": {...}}, None) on success,
    or (None, "error") on failure.
    """
    if not os.path.exists(CLIENT_SECRET_PATH):
        return None, f"Missing client secret at {CLIENT_SECRET_PATH}"

    try:
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_PATH, SCOPES)
        creds = flow.run_local_server(port=0)
    except Exception as e:
        return None, f"OAuth flow failed: {e}"

    try:
        # Robust identity via Userinfo API
        oauth2 = build("oauth2", "v2", credentials=creds, cache_discovery=False)
        u = oauth2.userinfo().get().execute()  # id, email, name, picture
        sub = u.get("id")
        if not sub:
            return None, "Could not determine Google user ID (sub)."
        idinfo = {
            "sub": sub,
            "email": u.get("email", ""),
            "name": u.get("name") or (u.get("email", "").split("@")[0] if u.get("email") else "You"),
            "picture": u.get("picture", ""),
        }
        return {"creds": creds, "idinfo": idinfo}, None
    except Exception as e:
        return None, f"Failed to fetch Google userinfo: {e}"
