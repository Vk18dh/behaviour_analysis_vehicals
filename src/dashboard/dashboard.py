"""
src/dashboard/dashboard.py
Streamlit Human-in-the-Loop Dashboard

Features:
  - Pending violation table with evidence thumbnails
  - Approve / Reject buttons per violation (calls /review API)
  - OCR low-confidence rows highlighted in orange
  - Credit score time-series chart per vehicle (Plotly)
  - System status panel (cameras, FPS, today's violation count)
  - Video clip playback for approved/pending clips
  - Sidebar filters: date, violation type, status, plate search
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.helpers import load_config

# ══════════════════════════════════════════════════════════════════════
# Config & API Client
# ══════════════════════════════════════════════════════════════════════

_CFG      = load_config()
_API_BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
_TIMEOUT  = 10.0


def _api(method: str, path: str, **kwargs):
    """Make an authenticated API call. Returns parsed JSON or None."""
    token = st.session_state.get("token")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = httpx.request(
            method, f"{_API_BASE}{path}",
            headers=headers, timeout=_TIMEOUT, **kwargs
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# Login
# ══════════════════════════════════════════════════════════════════════

def _login_page():
    st.title("🚦 Traffic Enforcement System — Login")
    username = st.text_input("Username", value="admin")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        try:
            resp = httpx.post(
                f"{_API_BASE}/auth/token",
                data={"username": username, "password": password},
                timeout=_TIMEOUT,
            )
            if resp.status_code == 200:
                st.session_state["token"]    = resp.json()["access_token"]
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Invalid credentials.")
        except Exception as e:
            st.error(f"Could not reach API: {e}")


# ══════════════════════════════════════════════════════════════════════
# Sidebar Filters
# ══════════════════════════════════════════════════════════════════════

def _sidebar_filters():
    st.sidebar.header("🔍 Filters")

    status_filter = st.sidebar.selectbox(
        "Status", ["all", "pending", "low_confidence", "approved", "rejected"],
        index=0,
    )
    vtype_filter = st.sidebar.selectbox(
        "Violation Type",
        ["all", "ZIGZAG", "TAILGATING", "RED_LIGHT", "OVERSPEED",
         "WRONG_DIRECTION", "HIGHWAY_RESTRICTION", "LANE_VIOLATION",
         "RASH_DRIVING", "NO_HELMET", "NO_SEATBELT", "TRIPLE_RIDING",
         "ILLEGAL_TURN", "PHONE_USE"],
    )
    plate_search = st.sidebar.text_input("Plate Search")
    date_from    = st.sidebar.date_input("From Date", value=date.today() - timedelta(days=7))
    date_to      = st.sidebar.date_input("To Date",   value=date.today())

    return {
        "status":    None if status_filter == "all" else status_filter,
        "type":      None if vtype_filter  == "all" else vtype_filter,
        "plate":     plate_search or None,
        "date_from": date_from.isoformat(),
        "date_to":   date_to.isoformat(),
    }


def _sidebar_settings():
    st.sidebar.header("⚙️ Settings")
    
    if "hw_restrict_enabled" not in st.session_state:
        st.session_state["hw_restrict_enabled"] = True
        
    enabled = st.sidebar.toggle(
        "Highway Restriction Detection", 
        value=st.session_state["hw_restrict_enabled"]
    )
    
    if enabled != st.session_state["hw_restrict_enabled"]:
        st.session_state["hw_restrict_enabled"] = enabled
        res = _api("PATCH", "/config/highway_restriction", json={"enabled": enabled})
        if res:
            st.sidebar.success(f"Detection {'enabled' if enabled else 'disabled'}!")
        else:
            st.sidebar.error("Failed to update setting.")


# ══════════════════════════════════════════════════════════════════════
# Violations Table
# ══════════════════════════════════════════════════════════════════════

def _violations_page(filters: dict):
    st.header("📋 Violations — Human Review Queue")

    params = {k: v for k, v in filters.items() if v is not None}
    data   = _api("GET", "/violations", params=params)

    if not data:
        st.info("No violations found for selected filters.")
        return

    for v in data:
        conf        = v.get("ocr_confidence", 0) or 0
        is_low_conf = conf < 0.90
        v_status    = v.get("status", "pending")

        # Card background colour
        card_color  = "#fff3cd" if is_low_conf else "#f8f9fa"
        border_color= "#ffc107" if is_low_conf else "#dee2e6"

        with st.container():
            st.markdown(
                f'<div style="background:{card_color};border:2px solid {border_color};'
                f'border-radius:8px;padding:12px;margin-bottom:12px;">',
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.markdown(f"**#{v['id']} — {v['type']}**")
                st.write(f"🚗 Plate: `{v.get('plate_text','—')}`")
                st.write(f"⚡ Speed: {v.get('speed_kmh','—')} km/h")
                st.write(f"💰 Fine: INR {v.get('fine_inr','—')}")
                st.write(f"📷 Camera: {v.get('camera_id','—')}")
                st.write(f"🕐 {v.get('timestamp','—')}")
                if is_low_conf:
                    st.warning(f"⚠️ Low OCR confidence: {conf*100:.0f}%")
                st.write(f"Status: `{v_status}`")
                mv = v.get("mv_act_section")
                if mv:
                    st.caption(f"MV Act: {mv}")

            with col2:
                img_path = v.get("evidence_image")
                if img_path and Path(img_path).exists():
                    st.image(img_path, caption="Evidence", use_container_width=True)
                else:
                    st.caption("No image available")

                clip_path = v.get("evidence_clip")
                if clip_path and Path(clip_path).exists():
                    with open(clip_path, "rb") as vf:
                        st.video(vf.read())

            with col3:
                if v_status not in ("approved", "rejected"):
                    vid = v["id"]
                    if st.button(f"✅ Approve", key=f"approve_{vid}"):
                        result = _api(
                            "PATCH", f"/review/{vid}",
                            json={"status": "approved"},
                        )
                        if result:
                            st.success("Approved! Notification sent.")
                            st.rerun()
                    if st.button(f"❌ Reject", key=f"reject_{vid}"):
                        result = _api(
                            "PATCH", f"/review/{vid}",
                            json={"status": "rejected"},
                        )
                        if result:
                            st.warning("Rejected.")
                            st.rerun()
                else:
                    st.write(f"✔ {v_status.capitalize()}")

            st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# Credit Score Chart
# ══════════════════════════════════════════════════════════════════════

def _score_page():
    st.header("📊 Vehicle Credit Score")
    vehicle_id = st.number_input("Enter Vehicle DB ID", min_value=1, step=1, value=1)

    if st.button("Load Score"):
        score_data = _api("GET", f"/score/{vehicle_id}")
        if score_data:
            score    = score_data["score"]
            category = score_data["category"]
            fines    = score_data["total_fines_inr"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Credit Score", f"{score} / 100")
            col2.metric("Risk Category", category)
            col3.metric("Total Fines", f"INR {fines:.0f}")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = score,
                domain= {"x": [0, 1], "y": [0, 1]},
                title = {"text": "Credit Score"},
                gauge = {
                    "axis":  {"range": [0, 100]},
                    "bar":   {"color": "#27ae60" if score >= 80
                               else "#f39c12" if score >= 50
                               else "#e74c3c"},
                    "steps": [
                        {"range": [0,   50], "color": "#fadbd8"},
                        {"range": [50,  80], "color": "#fdebd0"},
                        {"range": [80, 100], "color": "#d5f5e3"},
                    ],
                    "threshold": {
                        "line":      {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value":     50,
                    },
                },
            ))
            st.plotly_chart(fig, use_container_width=True)

            # Violation history table
            veh_data = _api("GET", f"/vehicle/{vehicle_id}")
            if veh_data and veh_data.get("violations"):
                df = pd.DataFrame(veh_data["violations"])
                if not df.empty and "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    fig2 = px.bar(
                        df, x="timestamp", y="fine_inr",
                        color="type", title="Violation Fine History",
                        labels={"fine_inr": "Fine (INR)", "timestamp": "Date"},
                    )
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error("Vehicle not found or no score recorded.")


# ══════════════════════════════════════════════════════════════════════
# System Status Panel
# ══════════════════════════════════════════════════════════════════════

def _status_page():
    st.header("🖥️ System Status")

    col1, col2, col3 = st.columns(3)

    # Fetch today's violations
    today = date.today().isoformat()
    data  = _api("GET", "/violations", params={"date_from": today, "date_to": today})
    total_today = len(data) if data else 0
    col1.metric("Violations Today",  total_today)
    col2.metric("API Status",        "🟢 Online")
    col3.metric("Dashboard Version", "1.0.0")

    if data:
        df = pd.DataFrame(data)
        if not df.empty and "type" in df.columns:
            breakdown = df["type"].value_counts().reset_index()
            breakdown.columns = ["Violation Type", "Count"]
            fig = px.bar(
                breakdown, x="Violation Type", y="Count",
                title="Today's Violations Breakdown",
                color="Violation Type",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Status distribution
        if "status" in df.columns:
            status_counts = df["status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig2 = px.pie(status_counts, names="Status", values="Count",
                          title="Review Status Distribution")
            st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# Video Upload Page
# ══════════════════════════════════════════════════════════════════════

def _upload_page():
    st.header("📤 Upload Traffic Video")
    st.write("Upload a pre-recorded `.mp4` or `.avi` traffic video to process it through the AI pipeline.")
    
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        if st.button("Start Processing"):
            with st.spinner("Uploading and starting batch pipeline..."):
                token = st.session_state.get("token")
                headers = {"Authorization": f"Bearer {token}"} if token else {}
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "video/mp4")}
                    resp = httpx.post(
                        f"{_API_BASE}/upload_video",
                        headers=headers,
                        files=files,
                        timeout=30.0
                    )
                    if resp.status_code in (200, 202):
                        st.success("✅ Video successfully uploaded! The AI pipeline is now processing it in the background.")
                        st.info("Check the 'Violations Review' tab in a few moments to see the detected violations appear.")
                    else:
                        st.error(f"Failed to start processing: {resp.text}")
                except Exception as e:
                    st.error(f"Upload failed: {e}")


# ══════════════════════════════════════════════════════════════════════
# Main App
# ══════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Traffic Enforcement System",
        page_icon="🚦",
        layout="wide",
    )

    if "token" not in st.session_state:
        _login_page()
        return

    # Top bar
    st.sidebar.title("🚦 Traffic Enforcement")
    st.sidebar.write(f"👤 {st.session_state.get('username', 'user')}")
    if st.sidebar.button("Logout"):
        del st.session_state["token"]
        st.rerun()

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🔍 Violations Review", "📊 Credit Scores", "🖥️ System Status", "📤 Upload Video"],
    )

    filters = _sidebar_filters()
    _sidebar_settings()

    if page == "🔍 Violations Review":
        _violations_page(filters)
    elif page == "📊 Credit Scores":
        _score_page()
    elif page == "🖥️ System Status":
        _status_page()
    elif page == "📤 Upload Video":
        _upload_page()


if __name__ == "__main__":
    main()
