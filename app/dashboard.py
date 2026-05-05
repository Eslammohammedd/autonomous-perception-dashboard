"""
dashboard.py — Streamlit Traffic Intelligence Dashboard.

Run:
    streamlit run app/dashboard.py

Features:
  - Upload a video or use webcam → live YOLO detection + tracking
  - Real-time counter panel (cars, trucks, persons …)
  - Analytics charts (trends, density, class breakdown)
  - Downloadable CSV report + JSON insights
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import tempfile
import json

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.yolo_detector import VehicleDetector, annotate_frame, draw_stats_overlay
from src.tracker       import VehicleTracker, estimate_speed, draw_tracks
from src.analytics     import (
    build_dataframe, compute_insights,
    plot_traffic_trends, print_insights,
    generate_ai_summary, DENSITY_COLORS,
)
from src.download_youtube import download_youtube_video
from src import config

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚗 AV Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS (dark theme polish)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background: #0f0f1a; color: #e0e0e0; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #16213e;
        border-right: 1px solid #1a1a3e;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1a1a2e;
        border: 1px solid #2a2a5e;
        border-radius: 10px;
        padding: 12px !important;
    }
    [data-testid="metric-container"] label { color: #a0a0c0 !important; font-size: 12px; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #00b4d8 !important; font-size: 28px !important; font-weight: 700;
    }

    /* Density badge */
    .density-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
    }

    /* Section headers */
    h2 { color: #00b4d8 !important; }
    h3 { color: #90e0ef !important; }

    /* Upload button */
    .stFileUploader label { color: #90e0ef; }

    /* Slider */
    .stSlider > div > div { background: #00b4d8; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Cached model loader (loads once per session)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_detector(model_name: str, conf: float):
    return VehicleDetector(model_path=model_name, conf_threshold=conf)


# ─────────────────────────────────────────────────────────────
# Sidebar — Configuration
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/traffic-light.png", width=64)
    st.title("⚙️ Configuration")

    st.subheader("Model")
    
    # Custom model path from config
    custom_model = str(config.MODEL_PATH)
    model_options = [custom_model, "yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    
    model_choice = st.selectbox(
        "YOLOv8 Weights",
        model_options,
        index=0,
        help="Custom Trained Model is selected by default.",
    )
    conf_threshold = st.slider("Confidence Threshold", 0.20, 0.90, 0.40, 0.05)
    imgsz_choice = st.select_slider("Inference Size (Resolution)", options=[320, 640, 736, 832, 1088], value=640, help="Higher values detect smaller/distant objects but decrease FPS. Must be a multiple of 32.")

    st.subheader("Tracking & Analysis")
    use_tracker = st.checkbox("Enable DeepSORT Tracking", value=True)
    show_trails = st.checkbox("Show Track Trails", value=True)
    show_distance = st.checkbox("📏 Estimate Distance (Meters)", value=True, help="Uses focal length approximation to estimate distance to vehicles.")
    n_init_val = st.slider("Tracking Stability (n_init)", 1, 15, 5, help="Number of frames a vehicle must be seen before confirming it.")

    st.subheader("Processing")
    max_frames = st.number_input("Max Frames to Process", 50, 5000, 300, 50)
    night_vision = st.checkbox("🌙 Night Vision Enhancement", value=False, help="Uses Gamma Correction to brighten and clarify dark/foggy videos before processing.")
    privacy_mode = st.checkbox("🕵️ Privacy Mode (Blur)", value=False, help="Automatically blurs vehicles and pedestrians to comply with privacy laws (GDPR).")

    st.markdown("---")
    st.caption(f"🖥️  GPU: {config.DEVICE.upper()}")
    st.caption("RTX 4060 — real-time at 45+ FPS")


# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
        color: #e0e0e0;
    }
    
    /* Glassmorphism Cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 15px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Titles & Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #00b4d8, #7209b7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0f0f1a;
    }
    </style>
    """, unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.title("🚗 AV Traffic Intelligence")
    st.markdown("#### Perception Stack v2.5 | Next-Gen Traffic Analysis")
with col_h2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=70)

# Congestion Alert Placeholder
alert_placeholder = st.empty()

tab_upload, tab_analytics, tab_about = st.tabs(
    ["📁 Video Upload", "📊 Analytics Dashboard", "🔧 About"]
)


# ═══════════════════════════════════════════════════════════════
# TAB 1 — Video Upload & Processing
# ═══════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
        <div style='text-align: center; padding: 20px; background: rgba(0, 180, 216, 0.1); border-radius: 15px; border: 1px solid rgba(0, 180, 216, 0.3);'>
            <h2 style='margin-bottom: 5px;'>Upload Your Footage</h2>
            <p style='color: #a8dadc; font-size: 16px;'>Deploy military-grade object tracking and traffic intelligence in one click.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # 3-Step Workflow
    wc1, wc2, wc3 = st.columns(3)
    wc1.markdown("<div style='text-align:center; padding:10px; background: rgba(255,255,255,0.05); border-radius:10px;'><h3>📤 1. Upload</h3><p style='font-size:14px; color:#888;'>Drop your MP4, AVI, or MOV file.</p></div>", unsafe_allow_html=True)
    wc2.markdown("<div style='text-align:center; padding:10px; background: rgba(255,255,255,0.05); border-radius:10px;'><h3>⚙️ 2. Configure</h3><p style='font-size:14px; color:#888;'>Adjust settings in the left sidebar.</p></div>", unsafe_allow_html=True)
    wc3.markdown("<div style='text-align:center; padding:10px; background: rgba(255,255,255,0.05); border-radius:10px;'><h3>📈 3. Analyze</h3><p style='font-size:14px; color:#888;'>Get speeds, heatmaps, and AI reports.</p></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload Area (Centered)
    _, upload_col, _ = st.columns([1, 2, 1])
    with upload_col:
        uploaded = st.file_uploader(
            "📁 Drag and drop video here", type=["mp4", "avi", "mov"],
            help="High-resolution dashcam or drone footage recommended for best results."
        )
        st.markdown("<p style='text-align: center; color:#888; font-weight:bold;'>— OR —</p>", unsafe_allow_html=True)
        youtube_url = st.text_input("🔗 Paste YouTube Link", placeholder="https://youtube.com/...")

    st.markdown("---")

    tmp_path = None
    vid_name = None

    if uploaded:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        vid_name = uploaded.name
    elif youtube_url:
        with st.spinner("⏳ Downloading video from YouTube... This may take a minute."):
            dl_path = download_youtube_video(youtube_url)
        if dl_path:
            tmp_path = dl_path
            vid_name = "YouTube Download"
        else:
            st.error("❌ Failed to download YouTube video. Please verify the link.")

    if tmp_path:
        st.success(f"Video ready ✅  |  {vid_name}")

        col_prev, col_btn = st.columns([3, 1])
        with col_btn:
            run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

        if run_btn:
            detector = load_detector(model_choice, conf_threshold)

            if use_tracker:
                tracker = VehicleTracker(n_init=n_init_val)

            cap     = cv2.VideoCapture(tmp_path)
            fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # UI placeholders
            st.markdown("### 🎬 Live Processing")
            col_vid, col_stats = st.columns([2, 1])

            with col_vid:
                frame_placeholder = st.empty()
            with col_stats:
                st.markdown("#### 📊 Live Counters")
                m_total  = st.metric("Total in Frame", 0)
                m_cars   = st.metric("🚗 Cars", 0)
                m_trucks = st.metric("🚛 Trucks", 0)
                m_people = st.metric("🚶 Persons", 0)
                m_lights = st.metric("🚦 Traffic Lights", 0)
                m_signs  = st.metric("🛣️ Road Signs", 0)
                m_coll   = st.metric("🚨 Collision Warnings", 0)
                m_fps    = st.metric("⚡ FPS", 0)
                m_unique = st.metric("🔢 Unique Vehicles", 0)

            progress = st.progress(0, text="Processing…")

            frame_logs = []
            heatmap_data = [] 
            frame_idx  = 0
            total_collisions = 0
            t_start    = time.perf_counter()

            # Video Export Setup
            frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            temp_output = Path(tempfile.gettempdir()) / "processed_video.mp4"
            out_video = cv2.VideoWriter(str(temp_output), fourcc, fps_src, (frame_width, frame_height))

            while frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # ── Image Enhancement ────────────────────────
                if night_vision:
                    lookUpTable = np.empty((1,256), np.uint8)
                    for i in range(256):
                        lookUpTable[0,i] = np.clip(pow(i / 255.0, 1.0 / 2.2) * 255.0, 0, 255)
                    frame = cv2.LUT(frame, lookUpTable)

                # ── Detect ───────────────────────────────────
                results    = detector.detect(frame, imgsz=imgsz_choice)
                detections = detector.parse_detections(results)
                counts     = detector.count_by_class(detections)

                # ── Privacy Mode (Anonymization) ──────────────
                if privacy_mode:
                    for d in detections:
                        if d['class_name'] in ['car', 'truck', 'bus', 'person']:
                            x1, y1, x2, y2 = map(int, d['bbox'])
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame_width, x2), min(frame_height, y2)
                            if x2 > x1 and y2 > y1:
                                roi = frame[y1:y2, x1:x2]
                                frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 0)

                # ── Collision / Proximity Alert ───────────────
                collision_risk = False
                frame_area = frame_width * frame_height
                for d in detections:
                    if d['class_name'] in ['car', 'truck', 'bus']:
                        x1, y1, x2, y2 = d['bbox']
                        # Lowered threshold to 5% of screen area (Hyper-sensitive)
                        if (x2 - x1) * (y2 - y1) > (0.05 * frame_area):
                            collision_risk = True
                            total_collisions += 1
                            break

                elapsed  = time.perf_counter() - t_start
                live_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
                total_seen = frame_idx   # fallback if no tracker

                # ── Track & Speed & Distance ───────────────────
                if use_tracker:
                    tracks, total_seen = tracker.update(detections, frame)
                    speeds = estimate_speed(tracker.trails, fps_src)
                    
                    if show_trails:
                        annotated = draw_tracks(frame, tracks, tracker.trails, tracker.class_map, speeds)
                        # Collect points for heatmap
                        for track in tracks:
                            ltrb = track.to_ltrb()
                            cx, cy = int((ltrb[0]+ltrb[2])/2), int((ltrb[1]+ltrb[3])/2)
                            heatmap_data.append((cx, cy))
                    else:
                        annotated = detector.annotate(frame, detections)
                        
                    # Draw Distance Estimation
                    if show_distance:
                        for track in tracks:
                            x1, y1, x2, y2 = track.to_ltrb()
                            w = max(x2 - x1, 1)
                            # Approximation: Real Width (2m) * Focal Length (~800) / Pixel Width
                            dist_m = (2.0 * 800) / w
                            # Draw yellow text above the bounding box
                            cv2.putText(annotated, f"~{dist_m:.1f}m", (int(x1), int(y1) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            
                else:
                    annotated = detector.annotate(frame, detections)

                # Draw collision warning directly on the video
                if collision_risk:
                    cv2.putText(annotated, "WARNING: COLLISION RISK", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

                # Stats overlay
                annotated = draw_stats_overlay(
                    annotated, counts, sum(counts.values()), live_fps, frame_idx
                )

                # ── Log ──────────────────────────────────────
                avg_speed = sum(speeds.values()) / len(speeds) if speeds else 0.0
                frame_logs.append({
                    "frame_id":       frame_idx,
                    "timestamp_s":    round(frame_idx / fps_src, 3),
                    "counts":         dict(counts),
                    "total_in_frame": sum(counts.values()),
                    "total_seen":     total_seen, 
                    "avg_speed_kmh":  round(avg_speed, 1),
                    "fps":            round(live_fps, 1),
                    "inference_ms":   detector.get_inference_speed(results),
                })
                
                # Write to export video
                out_video.write(annotated)

                # ── Alert Banner ──────────────────────────────
                current_count = sum(counts.values())
                if collision_risk:
                    alert_placeholder.error("🚨 **COLLISION WARNING:** Vehicle dangerously close to the camera!")
                elif current_count >= 10:
                    alert_placeholder.warning(f"⚠️ **CONGESTION ALERT:** {current_count} vehicles detected.")
                else:
                    alert_placeholder.empty()
                if frame_idx % 5 == 0:
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb, use_container_width=True)
                    
                    # Update live counters
                    m_total.metric("Total in Frame", sum(counts.values()))
                    m_cars.metric("🚗 Cars", counts.get("car", 0))
                    m_trucks.metric("🚛 Trucks", counts.get("truck", 0))
                    m_people.metric("🚶 Persons", counts.get("person", 0))
                    m_lights.metric("🚦 Traffic Lights", counts.get("traffic light", 0))
                    m_signs.metric("🛣️ Road Signs", counts.get("stop sign", 0))
                    m_coll.metric("🚨 Collision Warnings", total_collisions)
                    m_fps.metric("⚡ FPS", round(live_fps, 1))
                    m_unique.metric("🔢 Unique Vehicles", total_seen)
                    
                    progress.progress(
                        min(frame_idx / max_frames, 1.0),
                        text=f"Frame {frame_idx}/{max_frames}  |  {live_fps:.1f} FPS"
                    )

                frame_idx += 1

            cap.release()
            out_video.release()
            progress.progress(1.0, text="✅ Processing complete!")

            # Store results in session state
            st.session_state["frame_logs"]  = frame_logs
            st.session_state["fps_src"]     = fps_src

            # Quick summary
            df       = build_dataframe(frame_logs)
            insights = compute_insights(df)
            st.session_state["df"]       = df
            st.session_state["insights"] = insights
            st.session_state["insights"]["collision_warnings"] = total_collisions
            st.session_state["heatmap_data"] = heatmap_data
            st.session_state["frame_shape"]  = frame.shape

            st.success(
                f"✅ Processed **{frame_idx} frames**  |  "
                f"Total vehicles seen: **{insights.get('total_vehicles_seen', '—')}**  |  "
                f"Peak: **{insights.get('max_vehicles_in_frame', '—')}** vehicles"
            )

            # Download Button for Video
            if temp_output.exists():
                with open(temp_output, "rb") as f:
                    st.download_button(
                        label="📥 Download Processed Video (MP4)",
                        data=f,
                        file_name="autonomous_perception_demo.mp4",
                        mime="video/mp4"
                    )

            st.info("👉 Switch to **Analytics Dashboard** tab to see full charts.")


# ═══════════════════════════════════════════════════════════════
# TAB 2 — Analytics Dashboard
# ═══════════════════════════════════════════════════════════════
with tab_analytics:
    df       = st.session_state.get("df")
    insights = st.session_state.get("insights")

    if df is None or insights is None:
        st.info("📂 Upload and process a video first (Video Upload tab).")
        st.stop()

    st.subheader("📊 Traffic Summary")
    
    # AI Report
    st.markdown(generate_ai_summary(insights))
    st.markdown("---")

    # ── Metrics row ─────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("📹 Total Frames",           insights.get("total_frames", 0))
    c2.metric("🚗 Unique Vehicles",        insights.get("total_vehicles_seen", 0))
    c3.metric("📊 Avg Density",            f"{insights.get('avg_vehicles_per_frame', 0)} veh/f")
    c4.metric("📈 Peak Traffic",           insights.get("max_vehicles_in_frame", 0))
    c5.metric("🚨 Collisions",             insights.get("collision_warnings", 0))
    c6.metric("⚡ Speed",                  f"{insights.get('avg_fps', 0)} FPS")

    st.markdown("---")

    # ── Density distribution ────────────────────────────────
    st.subheader("🚦 Traffic Density Breakdown")
    density_pct = insights.get("density_distribution_%", {})
    dc1, dc2, dc3, dc4, dc5 = st.columns(len(DENSITY_COLORS))
    for col, (label, color) in zip([dc1, dc2, dc3, dc4, dc5], DENSITY_COLORS.items()):
        pct = density_pct.get(label, 0)
        col.markdown(
            f"""<div style='background:{color};border-radius:10px;padding:10px;text-align:center'>
            <b style='color:#111'>{label.upper()}</b><br>
            <span style='font-size:22px;font-weight:700;color:#111'>{pct}%</span>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Charts ──────────────────────────────────────────────
    st.subheader("📈 Vehicle Count Over Time")
    import plotly.graph_objects as go
    import plotly.express as px

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=df["timestamp_s"], y=df["total"],
        fill="tozeroy", mode="lines",
        line=dict(color="#00b4d8", width=1.5),
        fillcolor="rgba(0,180,216,0.2)",
        name="Total Vehicles",
    ))
    # Overlay per-class
    class_colors_px = {
        "car": "#f4a261", "truck": "#e63946", "bus": "#2a9d8f",
        "motorcycle": "#a8dadc", "bicycle": "#457b9d", "person": "#e9c46a"
    }
    for cls, color in class_colors_px.items():
        if cls in df.columns and df[cls].sum() > 0:
            fig_line.add_trace(go.Scatter(
                x=df["timestamp_s"], y=df[cls],
                mode="lines", name=cls,
                line=dict(color=color, width=1),
                opacity=0.7,
            ))

    fig_line.update_layout(
        plot_bgcolor="#1a1a2e", paper_bgcolor="#0f0f1a",
        font=dict(color="#e0e0e0"),
        legend=dict(bgcolor="#1a1a2e"),
        xaxis_title="Time (s)", yaxis_title="Count",
        height=350,
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # ── Speed Analysis Chart ────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Average Speed Trend (km/h)")
    fig_speed = px.line(
        df, x="timestamp_s", y="avg_speed_kmh",
        labels={"avg_speed_kmh": "Avg Speed (km/h)", "timestamp_s": "Time (s)"},
        template="plotly_dark"
    )
    fig_speed.update_traces(line_color="#7209b7", fill="tozeroy")
    fig_speed.update_layout(
        plot_bgcolor="#1a1a2e", paper_bgcolor="#0f0f1a",
        xaxis_title="Time (s)", yaxis_title="Speed (km/h)",
        height=300
    )
    st.plotly_chart(fig_speed, use_container_width=True)

    # ── Class & Density side-by-side ─────────────────────────
    col_cls, col_dens = st.columns(2)

    with col_cls:
        st.subheader("🚗 Class Distribution")
        class_totals = insights.get("class_totals", {})
        fig_bar = px.bar(
            x=list(class_totals.keys()),
            y=list(class_totals.values()),
            color=list(class_totals.keys()),
            color_discrete_map=class_colors_px,
            labels={"x": "Class", "y": "Total Detections"},
        )
        fig_bar.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#0f0f1a",
            font=dict(color="#e0e0e0"), showlegend=False, height=300,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_dens:
        st.subheader("🚦 Density Pie Chart")
        fig_pie = px.pie(
            names=list(density_pct.keys()),
            values=list(density_pct.values()),
            color=list(density_pct.keys()),
            color_discrete_map=DENSITY_COLORS,
            hole=0.4,
        )
        fig_pie.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#0f0f1a",
            font=dict(color="#e0e0e0"), height=300,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    # ── Heatmap Section ─────────────────────────────────────
    st.markdown("---")
    st.subheader("🗺️ Traffic Occupancy Heatmap")
    hdata = st.session_state.get("heatmap_data", [])
    fshape = st.session_state.get("frame_shape", (720, 1280))
    
    if hdata:
        h_df = pd.DataFrame(hdata, columns=["x", "y"])
        fig_heat = px.density_heatmap(
            h_df, x="x", y="y", 
            nbinsx=50, nbinsy=50,
            color_continuous_scale="Viridis",
            labels={"x": "Width (px)", "y": "Height (px)"},
            range_x=[0, fshape[1]], range_y=[fshape[0], 0] # Flip y for image coord
        )
        fig_heat.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#0f0f1a",
            font=dict(color="#e0e0e0"), height=500
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("The brighter areas indicate higher vehicle dwell time or frequency.")

    # ── Congestion events ───────────────────────────────────
    congestion = insights.get("congestion_periods", [])
    if congestion:
        st.subheader(f"⚠️ Congestion Events ({len(congestion)} detected)")
        cdf = pd.DataFrame(congestion)
        cdf["duration_s"] = (cdf["end_s"] - cdf["start_s"]).round(1)
        st.dataframe(
            cdf[["start", "end", "duration_s"]].rename(columns={
                "start": "Start", "end": "End", "duration_s": "Duration (s)"
            }),
            use_container_width=True,
        )

    # ── Raw data + download ─────────────────────────────────
    st.markdown("---")
    st.subheader("📥 Download Results")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "⬇️  Download CSV Log",
            data=df.to_csv(index=False),
            file_name="traffic_log.csv",
            mime="text/csv",
        )
    with col_dl2:
        st.download_button(
            "⬇️  Download JSON Insights",
            data=json.dumps(insights, indent=2),
            file_name="traffic_insights.json",
            mime="application/json",
        )

    with st.expander("🔍 View Raw DataFrame"):
        st.dataframe(df.tail(100), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — About
# ═══════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
    ## 🤖 About This System

    **Autonomous Vehicle Traffic Intelligence Platform**
    Built as part of the *Supervised Learning — Spring 2026* portfolio.

    ### Architecture
    ```
    Video Input
        │
        ▼
    YOLOv8 Detection  (ultralytics)
        │
        ▼
    DeepSORT Tracking (deep-sort-realtime)
        │
        ▼
    Analytics Engine  (pandas + plotly)
        │
        ▼
    Streamlit Dashboard  ←  You are here
        │
        ▼
    FastAPI Backend  (api/main.py)
    ```

    ### Key Technologies
    | Component | Technology |
    |---|---|
    | Object Detection | YOLOv8 (Ultralytics) |
    | Multi-Object Tracking | DeepSORT |
    | GPU | NVIDIA RTX 4060 (CUDA) |
    | Analytics | Pandas + Plotly |
    | UI | Streamlit |
    | API | FastAPI + Uvicorn |
    | Container | Docker |
    """)
