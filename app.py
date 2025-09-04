import streamlit as st
from src.ui import render_app
from src.services.storage import init_db

st.set_page_config(page_title="Empathetic Journaling Companion", page_icon="📝", layout="wide")

def main():
    init_db()
    render_app()

if __name__ == "__main__":
    main()
