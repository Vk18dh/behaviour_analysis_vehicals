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
_TIMEOUT  = 30.0


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
        # Fetch actual state from API instead of defaulting to True
        res = _api("GET", "/config/highway_restriction")
        st.session_state["hw_restrict_enabled"] = res.get("highway_restriction_enabled", False) if res else False
        
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
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.header("📋 Violations — Human Review Queue")
    
    with col_h2:
        if st.session_state.get("role", "admin") == "admin":
            with st.expander("🗑️ Danger Zone"):
                st.warning("This will permanently delete all violations.")
                if st.button("Clear All Violations", type="primary", use_container_width=True):
                    res = _api("DELETE", "/violations")
                    if res and res.get("status") == "ok":
                        st.success(f"Cleared {res.get('deleted_count')} violations!")
                        st.rerun()
                    else:
                        st.error("Failed to clear database.")

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
                    try:
                        with open(img_path, "rb") as im_f:
                            st.image(im_f.read(), caption="Evidence")
                    except Exception as e:
                        st.error(f"Could not load image: {e}")
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
        token   = st.session_state.get("token")
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        if st.button("🚀 Start Processing", type="primary"):
            video_bytes = uploaded_file.getvalue()

            # ── 1. Submit to full detection pipeline (background) ────
            with st.spinner("Submitting to detection pipeline…"):
                try:
                    files = {"file": (uploaded_file.name, video_bytes, "video/mp4")}
                    resp  = httpx.post(
                        f"{_API_BASE}/upload_video",
                        headers=headers, files=files, timeout=30.0,
                    )
                    if resp.status_code in (200, 202):
                        st.success(
                            "✅ Video submitted! The AI detection pipeline is running in the background. "
                            "Check **Violations Review** in a few moments."
                        )
                    else:
                        st.error(f"Pipeline submission failed: {resp.text}")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

            # ── 2. Run DIP analysis immediately and show inline ──────
            st.divider()
            st.subheader("🔬 DIP Processing Report for this Video")
            with st.spinner("Analysing video quality frame-by-frame…"):
                try:
                    files2 = {"file": (uploaded_file.name, video_bytes, "video/mp4")}
                    resp2  = httpx.post(
                        f"{_API_BASE}/dip/analyze_video",
                        headers=headers, files=files2,
                        params={"max_frames": 30},
                        timeout=120.0,
                    )
                    if resp2.status_code == 200:
                        st.session_state["upload_dip_result"] = resp2.json()
                    else:
                        st.warning(f"DIP analysis failed: {resp2.text}")
                except Exception as e:
                    st.warning(f"DIP analysis error: {e}")

        # ── Show DIP results if available ────────────────────────────
        if "upload_dip_result" in st.session_state:
            result  = st.session_state["upload_dip_result"]
            reports = result.get("reports", [])

            if not reports:
                st.info("No DIP data available for this video.")
            else:
                total_sampled = result.get("total_frames_sampled", len(reports))
                total_video   = result.get("total_frames_in_video", "?")
                pc = result.get("problem_counts", {})
                fc = result.get("fix_counts", {})

                st.caption(
                    f"Sampled **{total_sampled}** frames from {total_video} total frames"
                )

                # ── Problem metrics row ───────────────────────────────
                if pc:
                    cols = st.columns(min(len(pc), 4))
                    for i, (prob, count) in enumerate(pc.items()):
                        pct = round(count / max(total_sampled, 1) * 100, 1)
                        emoji = {
                            "LOW_LIGHT": "🌑", "OVEREXPOSED": "☀️",
                            "BLUR": "🔲", "NOISE": "📡",
                            "HAZE_FOG": "🌫", "COLOR_CAST": "🎨",
                        }.get(prob, "⚠️")
                        cols[i % 4].metric(
                            f"{emoji} {prob.replace('_', ' ')}",
                            f"{count} frames", f"{pct}% of video"
                        )
                else:
                    st.success("✅ No quality problems detected in this video.")

                # ── Charts side by side ───────────────────────────────
                left, right = st.columns(2)

                if pc:
                    with left:
                        prob_df = pd.DataFrame(list(pc.items()), columns=["Problem", "Frames"])
                        fig_p = px.bar(
                            prob_df, x="Problem", y="Frames",
                            title="Problems Detected", color="Problem",
                            color_discrete_map={
                                "LOW_LIGHT": "#3498db", "OVEREXPOSED": "#e67e22",
                                "BLUR": "#9b59b6", "NOISE": "#e74c3c",
                                "HAZE_FOG": "#95a5a6", "COLOR_CAST": "#f1c40f",
                            },
                        )
                        fig_p.update_layout(showlegend=False, height=260, margin=dict(t=35, b=10))
                        st.plotly_chart(fig_p, use_container_width=True)

                if fc:
                    with right:
                        fix_df = pd.DataFrame(list(fc.items()), columns=["Fix", "Frames"])
                        fig_f = px.bar(
                            fix_df, x="Fix", y="Frames",
                            title="Fixes Applied", color="Fix",
                        )
                        fig_f.update_layout(showlegend=False, height=260, margin=dict(t=35, b=10))
                        st.plotly_chart(fig_f, use_container_width=True)

                # ── Quality timeline ──────────────────────────────────
                df = pd.DataFrame(reports)
                fig_tl = go.Figure()
                fig_tl.add_trace(go.Scatter(
                    x=df["frame_idx"], y=df["brightness_before"],
                    name="Brightness (raw)", line=dict(color="#aed6f1", dash="dot"),
                ))
                fig_tl.add_trace(go.Scatter(
                    x=df["frame_idx"], y=df["brightness_after"],
                    name="Brightness (processed)", line=dict(color="#2980b9"),
                ))
                fig_tl.add_trace(go.Scatter(
                    x=df["frame_idx"], y=df["blur_score_before"],
                    name="Blur Score (raw)", line=dict(color="#f9ca8f", dash="dot"),
                    yaxis="y2",
                ))
                fig_tl.add_trace(go.Scatter(
                    x=df["frame_idx"], y=df["blur_score_after"],
                    name="Blur Score (processed)", line=dict(color="#e67e22"),
                    yaxis="y2",
                ))
                fig_tl.update_layout(
                    title="Quality Before → After DIP Processing",
                    xaxis_title="Frame #",
                    yaxis=dict(title="Brightness (0–255)", side="left"),
                    yaxis2=dict(title="Blur Score", side="right", overlaying="y"),
                    legend=dict(orientation="h", y=-0.25),
                    height=320, margin=dict(t=40, b=10),
                )
                st.plotly_chart(fig_tl, use_container_width=True)

                # ── Worst frame ───────────────────────────────────────
                df["n_problems"] = df["problems_detected"].apply(len)
                worst = reports[int(df["n_problems"].idxmax())]
                if worst["problems_detected"]:
                    st.info(
                        f"🔍 **Worst frame #{worst['frame_idx']}** — "
                        f"Problems: `{'  ·  '.join(worst['problems_detected'])}` → "
                        f"Fixes: `{'  ·  '.join(worst['fixes_applied'])}`  |  "
                        f"Brightness {worst['brightness_before']:.0f}→{worst['brightness_after']:.0f}  "
                        f"Blur {worst['blur_score_before']:.0f}→{worst['blur_score_after']:.0f}"
                    )

                with st.expander("📋 Full Frame Report Table"):
                    st.dataframe(
                        df[["frame_idx", "problems_detected", "fixes_applied",
                            "brightness_before", "brightness_after",
                            "blur_score_before", "blur_score_after", "noise_level"]],
                        use_container_width=True,
                    )


# ══════════════════════════════════════════════════════════════════════
# DIP Monitor Page
# ══════════════════════════════════════════════════════════════════════

def _dip_monitor_page():
    st.header("🔬 DIP Monitor — Frame Processing Analysis")
    st.write(
        "See exactly what the Digital Image Processing pipeline detected and fixed "
        "for every frame — both live streams and uploaded videos."
    )

    live_tab, upload_tab = st.tabs(["📡 Live Stream Monitor", "📁 Video Analysis"])

    # ── Tab 1: Live stream DIP stats ─────────────────────────────────
    with live_tab:
        st.subheader("Live Stream DIP Report")
        st.caption("Shows the last N DIP-processed frames from the active pipeline. Click Refresh to update.")

        col_left, col_right = st.columns([3, 1])
        n_frames    = col_left.slider("Frames to display", 20, 500, 100, step=10)
        do_refresh  = col_right.button("🔄 Refresh", use_container_width=True)

        placeholder = st.empty()

        def _render_live_dip(container):
            data = _api("GET", "/dip/stats", params={"n": n_frames})
            if data is None or data.get("status") == "no_active_pipeline":
                container.info("⏸ No active live pipeline. Start a live stream first via 'System Status' or the API.")
                return
            reports = data.get("reports", [])
            if not reports:
                container.info("No DIP reports yet — pipeline may just have started.")
                return

            import pandas as pd
            df = pd.DataFrame(reports)

            # ── Metric cards ──────────────────────────────────────────
            last = reports[-1]
            m1, m2, m3, m4 = container.columns(4)
            m1.metric(
                "💡 Brightness",
                f"{last['brightness_after']:.0f}",
                delta=f"{last['brightness_after'] - last['brightness_before']:.0f}",
                delta_color="normal",
            )
            m2.metric(
                "🔲 Contrast",
                f"{last['contrast_after']:.0f}",
                delta=f"{last['contrast_after'] - last['contrast_before']:.0f}",
            )
            m3.metric(
                "📐 Blur Score",
                f"{last['blur_score_after']:.0f}",
                delta=f"{last['blur_score_after'] - last['blur_score_before']:.0f}",
            )
            m4.metric("🌫 Noise Level", f"{last['noise_level']:.1f}")

            # ── Problems detected (last frame) ────────────────────────
            probs = last.get("problems_detected", [])
            fixes = last.get("fixes_applied", [])
            if probs:
                container.warning(f"**Last frame problems:** {' · '.join(probs)}")
                container.success(f"**Fixes applied:** {' · '.join(fixes)}")
            else:
                container.success("✅ Last frame: No problems detected — no fixes needed.")

            # ── Problem frequency bar chart ───────────────────────────
            all_probs = [p for r in reports for p in r.get("problems_detected", [])]
            if all_probs:
                prob_counts = pd.Series(all_probs).value_counts().reset_index()
                prob_counts.columns = ["Problem", "Count"]
                fig_prob = px.bar(
                    prob_counts, x="Problem", y="Count",
                    title=f"Problem Frequency (last {len(reports)} frames)",
                    color="Problem",
                    color_discrete_map={
                        "LOW_LIGHT":   "#3498db",
                        "OVEREXPOSED": "#e67e22",
                        "BLUR":        "#9b59b6",
                        "NOISE":       "#e74c3c",
                        "HAZE_FOG":    "#95a5a6",
                        "COLOR_CAST":  "#f1c40f",
                    },
                )
                fig_prob.update_layout(showlegend=False, height=280)
                container.plotly_chart(fig_prob, use_container_width=True)

            # ── Fix frequency bar chart ───────────────────────────────
            all_fixes = [f for r in reports for f in r.get("fixes_applied", [])]
            if all_fixes:
                fix_counts = pd.Series(all_fixes).value_counts().reset_index()
                fix_counts.columns = ["Fix", "Count"]
                fig_fix = px.bar(
                    fix_counts, x="Fix", y="Count",
                    title="Fixes Applied Frequency",
                    color="Fix",
                )
                fig_fix.update_layout(showlegend=False, height=260)
                container.plotly_chart(fig_fix, use_container_width=True)

            # ── Quality timeline ──────────────────────────────────────
            if len(df) > 1:
                fig_tl = go.Figure()
                fig_tl.add_trace(go.Scatter(
                    x=df["frame_idx"], y=df["brightness_before"],
                    name="Brightness (raw)", line=dict(color="#aed6f1", dash="dot"),
                ))
                fig_tl.add_trace(go.Scatter(
                    x=df["frame_idx"], y=df["brightness_after"],
                    name="Brightness (processed)", line=dict(color="#2980b9"),
                ))
                fig_tl.add_trace(go.Scatter(
                    x=df["frame_idx"], y=df["blur_score_before"],
                    name="Blur Score (raw)", line=dict(color="#f9ca8f", dash="dot"),
                    yaxis="y2",
                ))
                fig_tl.add_trace(go.Scatter(
                    x=df["frame_idx"], y=df["blur_score_after"],
                    name="Blur Score (processed)", line=dict(color="#e67e22"),
                    yaxis="y2",
                ))
                fig_tl.update_layout(
                    title="Frame Quality Timeline",
                    xaxis_title="Frame #",
                    yaxis=dict(title="Brightness (0–255)", side="left"),
                    yaxis2=dict(title="Blur Score (Laplacian var)", side="right", overlaying="y"),
                    legend=dict(orientation="h", y=-0.2),
                    height=350,
                )
                container.plotly_chart(fig_tl, use_container_width=True)

            # ── Raw reports table ─────────────────────────────────────
            with container.expander("📋 Raw Frame Reports Table"):
                display_df = df[["frame_idx", "problems_detected", "fixes_applied",
                                  "brightness_before", "brightness_after",
                                  "blur_score_before", "blur_score_after",
                                  "noise_level"]].tail(50)
                st.dataframe(display_df, use_container_width=True)

        _render_live_dip(placeholder)

    # ── Tab 2: Uploaded video DIP analysis ───────────────────────────
    with upload_tab:
        st.subheader("Upload Video for DIP Analysis")
        st.write(
            "Upload any traffic video. The system will run **only** the DIP analysis "
            "(no full detection) and show you which frames had problems and what was fixed."
        )

        up_file   = st.file_uploader("Choose video", type=["mp4", "avi", "mov"], key="dip_upload")
        max_frames = st.slider("Frames to sample (more = slower)", 10, 100, 30, step=10)

        if up_file and st.button("🔬 Analyse DIP Processing"):
            with st.spinner("Running frame-by-frame DIP analysis…"):
                token   = st.session_state.get("token")
                headers = {"Authorization": f"Bearer {token}"} if token else {}
                try:
                    resp = httpx.post(
                        f"{_API_BASE}/dip/analyze_video",
                        headers=headers,
                        files={"file": (up_file.name, up_file.getvalue(), "video/mp4")},
                        params={"max_frames": max_frames},
                        timeout=120.0,
                    )
                    if resp.status_code == 200:
                        result = resp.json()
                        st.session_state["dip_analysis"] = result
                    else:
                        st.error(f"Analysis failed: {resp.text}")
                except Exception as e:
                    st.error(f"Request error: {e}")

        if "dip_analysis" in st.session_state:
            result = st.session_state["dip_analysis"]
            reports = result.get("reports", [])
            if not reports:
                st.warning("No frames analysed.")
            else:
                import pandas as pd

                df = pd.DataFrame(reports)
                total_sampled = result.get("total_frames_sampled", len(reports))
                total_video   = result.get("total_frames_in_video", "?")

                st.success(
                    f"✅ Analysed **{total_sampled}** sampled frames "
                    f"from {total_video} total frames."
                )

                # ── Summary metrics ───────────────────────────────────
                pc = result.get("problem_counts", {})
                fc = result.get("fix_counts", {})

                if pc:
                    st.subheader("📊 Problem Summary")
                    mc = st.columns(min(len(pc), 4))
                    for i, (prob, count) in enumerate(pc.items()):
                        pct = round(count / total_sampled * 100, 1)
                        mc[i % 4].metric(prob.replace("_", " "), f"{count} frames", f"{pct}% of video")

                # ── Problem frequency bar ─────────────────────────────
                if pc:
                    prob_df = pd.DataFrame(list(pc.items()), columns=["Problem", "Frames"])
                    fig_p = px.bar(
                        prob_df, x="Problem", y="Frames",
                        title="Frames Affected by Each Problem",
                        color="Problem",
                    )
                    fig_p.update_layout(showlegend=False, height=280)
                    st.plotly_chart(fig_p, use_container_width=True)

                # ── Fix frequency ─────────────────────────────────────
                if fc:
                    fix_df = pd.DataFrame(list(fc.items()), columns=["Fix Applied", "Frames"])
                    fig_f = px.bar(
                        fix_df, x="Fix Applied", y="Frames",
                        title="Fixes Applied (how many frames each fix was used)",
                        color="Fix Applied",
                    )
                    fig_f.update_layout(showlegend=False, height=280)
                    st.plotly_chart(fig_f, use_container_width=True)

                # ── Quality improvement timeline ──────────────────────
                st.subheader("📈 Quality Before vs After — Full Video")
                fig_tl = go.Figure()
                fig_tl.add_trace(go.Scatter(
                    x=df["frame_idx"], y=df["brightness_before"],
                    name="Brightness (raw)", line=dict(color="#aed6f1", dash="dot"),
                ))
                fig_tl.add_trace(go.Scatter(
                    x=df["frame_idx"], y=df["brightness_after"],
                    name="Brightness (processed)", line=dict(color="#2980b9"),
                ))
                fig_tl.update_layout(
                    xaxis_title="Frame #", yaxis_title="Brightness",
                    height=300, legend=dict(orientation="h"),
                )
                st.plotly_chart(fig_tl, use_container_width=True)

                fig_bl = go.Figure()
                fig_bl.add_trace(go.Scatter(
                    x=df["frame_idx"], y=df["blur_score_before"],
                    name="Blur Score (raw)", line=dict(color="#f9ca8f", dash="dot"),
                ))
                fig_bl.add_trace(go.Scatter(
                    x=df["frame_idx"], y=df["blur_score_after"],
                    name="Blur Score (processed)", line=dict(color="#e67e22"),
                ))
                fig_bl.update_layout(
                    xaxis_title="Frame #", yaxis_title="Blur Score (Laplacian var)",
                    height=280, legend=dict(orientation="h"),
                )
                st.plotly_chart(fig_bl, use_container_width=True)

                # ── Worst frame highlight ─────────────────────────────
                st.subheader("🔍 Worst Quality Frame")
                # Find frame with most problems
                df["n_problems"] = df["problems_detected"].apply(len)
                worst_idx = int(df["n_problems"].idxmax())
                worst     = reports[worst_idx]
                st.info(
                    f"**Frame #{worst['frame_idx']}** had the most issues: "
                    f"`{'  ·  '.join(worst['problems_detected']) or 'None'}`  →  "
                    f"Fixes: `{'  ·  '.join(worst['fixes_applied']) or 'None'}`"
                )
                wcol1, wcol2, wcol3 = st.columns(3)
                wcol1.metric("Brightness before / after",
                             f"{worst['brightness_before']:.0f}",
                             delta=f"→ {worst['brightness_after']:.0f}")
                wcol2.metric("Blur score before / after",
                             f"{worst['blur_score_before']:.0f}",
                             delta=f"→ {worst['blur_score_after']:.0f}")
                wcol3.metric("Noise Level", f"{worst['noise_level']:.1f}")

                # ── Raw table ─────────────────────────────────────────
                with st.expander("📋 Full Frame Report Table"):
                    show_df = df[["frame_idx", "problems_detected", "fixes_applied",
                                   "brightness_before", "brightness_after",
                                   "blur_score_before", "blur_score_after",
                                   "noise_level"]]
                    st.dataframe(show_df, use_container_width=True)


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
        ["🔍 Violations Review", "📊 Credit Scores", "🖥️ System Status",
         "📤 Upload Video", "🔬 DIP Monitor"],
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
    elif page == "🔬 DIP Monitor":
        _dip_monitor_page()


if __name__ == "__main__":
    main()
