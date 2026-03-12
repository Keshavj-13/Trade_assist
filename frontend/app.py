"""Streamlit UI for Market Assistant manual research with wallet-aware controls."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from infra.user_store import (
    authenticate,
    create_user,
    get_wallet,
    update_wallet,
    ensure_user,
    user_exists,
)
from service.database import get_open_positions
from service.research import (
    format_summary_text,
    persist_scan_results,
    perform_scan,
)

ensure_user("admin")
ensure_user("telegram")

st.set_page_config(
    page_title="Market Assistant",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        background: linear-gradient(180deg, #080c16 0%, #0f1a37 40%, #111b2d 100%);
    }
    .stApp {
        background-color: #0b0f1e;
        color: #f8fafc;
    }
    .stButton>button {
        background-color: #2d9cdb;
        color: #fff;
        border-radius: 10px;
        padding: 0.65rem 1.1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #1f6fa3;
    }
    .stTextInput>div>div>input {
        background-color: rgba(255,255,255,0.05);
        border-radius: 8px;
        color: #f8fafc;
        padding: 0.55rem;
    }
    .stRadio>div>label>div, .stRadio>div>label>span {
        color: #f8fafc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='display:flex;align-items:center;gap:12px;margin-bottom:6px;'>"
    "<div style='font-size:34px;font-weight:700;'>Market Assistant</div>"
    "<div style='color:#a6b1e1'>Intraday intuition, AI guardrails, human oversight</div>"
    "</div>",
    unsafe_allow_html=True,
)

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = ""

if not st.session_state["authenticated"]:
    mode = st.radio("Access mode", ["Log in", "Sign up"], index=0)
    if mode == "Log in":
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
            if submitted:
                if authenticate(username.strip(), password or ""):
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username.strip()
                    st.success(f"Welcome back, {username.strip()}!")
                else:
                    st.error("Invalid credentials.")
    else:
        with st.form("signup_form"):
            username = st.text_input("Choose a username")
            password = st.text_input("Choose a password", type="password")
            password2 = st.text_input("Confirm password", type="password")
            wallet_amount = st.number_input("Initial wallet amount", min_value=0.0, value=1000.0, step=100.0)
            submitted = st.form_submit_button("Sign up")
            if submitted:
                username = username.strip()
                if not username:
                    st.error("Username cannot be blank.")
                elif password != password2:
                    st.error("Passwords do not match.")
                elif user_exists(username):
                    st.error("Username already exists.")
                else:
                    create_user(username, password, wallet_amount)
                    st.success(f"Account created. You can now log in as {username}.")
    if not st.session_state["authenticated"]:
        st.stop()

with st.sidebar:
    st.success(f"Logged in as {st.session_state['username']}")
    if st.button("Log out"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = ""

wallet_amount = get_wallet(st.session_state["username"])
st.subheader("Wallet & profile")
metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric(label="Available Cash", value=f"₹{wallet_amount:,.2f}")
metric_col2.metric(label="Current Scan", value="Real-time intraday")
metric_col3.metric(label="Risk Guard", value="Stop-loss & RL +0.25")
new_wallet = st.number_input("Update wallet amount", value=float(wallet_amount), step=100.0)
if st.button("Set wallet"):
    update_wallet(st.session_state["username"], new_wallet)
    st.success("Wallet updated.")

st.subheader("Portfolio overview")
positions = get_open_positions()
if positions:
    st.table(positions)
else:
    st.info("No open positions recorded yet.")

st.subheader("Manual research desk")
with st.container():
    st.markdown(
        "<div style='background:#131c36; border-radius:12px; padding:16px;'>"
        "<h3 style='margin:0; color:#f8fafc;'>Run a guided research scan</h3>"
        "<p style='color:#b2c1ff;'>Choose whole universe to guard entry, portfolio scan to focus existing positions.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Full universe research"):
            with st.spinner("Scanning NSE symbols..."):
                scan_result = perform_scan(scope="whole", wallet=wallet_amount)
                persist_scan_results(scan_result, username=st.session_state["username"])
                st.text(format_summary_text(scan_result))
    with col2:
        if st.button("Portfolio-only research"):
            with st.spinner("Scanning logged positions..."):
                scan_result = perform_scan(scope="portfolio", wallet=wallet_amount)
                persist_scan_results(scan_result, username=st.session_state["username"])
                st.text(format_summary_text(scan_result))

st.markdown("---")
st.caption(
    "User data now lives in `data/market.db`. Wallets and RL-backed preferences persist across sessions via SQLite."
)
