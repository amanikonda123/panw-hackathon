APP_TITLE = "Empathetic Journaling Companion"
DB_PATH = "data/journal.db"

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/calendar.readonly",  # keep for later calendar features
]

TOKENS_DIR = "tokens"
CLIENT_SECRET_PATH = "credentials/client_secret.json"
