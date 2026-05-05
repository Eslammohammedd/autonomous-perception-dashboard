"""
analytics.py — Traffic Analytics Engine.

Converts raw per-frame detection/tracking logs into a structured
Pandas DataFrame and computes traffic insights:

  - build_dataframe()     : frame_logs → DataFrame
  - compute_insights()    : peak hours, congestion patterns, stats
  - plot_traffic_trends() : visualise counts over time
  - save_report()         : export results/reports/traffic_report.json

The DataFrame schema:
  frame_id | timestamp_s | cars | trucks | buses | motorcycles |
  bicycles | persons | total | fps | inference_ms | density_label
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

from src import config


# ─────────────────────────────────────────────────────────────
# Density thresholds (vehicles per frame)
# ─────────────────────────────────────────────────────────────
DENSITY_LEVELS = {
    "free":       (0,  3),
    "light":      (3,  8),
    "moderate":   (8,  15),
    "heavy":      (15, 25),
    "congested":  (25, float("inf")),
}

DENSITY_COLORS = {
    "free":      "#2ecc71",
    "light":     "#f1c40f",
    "moderate":  "#e67e22",
    "heavy":     "#e74c3c",
    "congested": "#8e44ad",
}

ALL_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "traffic light", "stop sign"]


# ─────────────────────────────────────────────────────────────
# 1. DataFrame Builder
# ─────────────────────────────────────────────────────────────

def build_dataframe(frame_logs: list[dict]) -> pd.DataFrame:
    """
    Convert the raw list of per-frame dicts (from yolo_detector.run_on_video
    or the combined detection + tracking pipeline) into a tidy DataFrame.

    Parameters
    ----------
    frame_logs : list of dicts produced by run_on_video()
        Keys: frame_id, timestamp_s, counts (dict), total_in_frame, fps,
              inference_ms  [total_seen is optional — added by tracker]

    Returns
    -------
    pd.DataFrame with columns:
        frame_id | timestamp_s | car | truck | bus | motorcycle |
        bicycle | person | total | fps | inference_ms | density_label
    """
    rows = []
    for log in frame_logs:
        row = {
            "frame_id":     log["frame_id"],
            "timestamp_s":  log["timestamp_s"],
            "fps":          log.get("fps", 0),
            "inference_ms": log.get("inference_ms", 0),
        }
        counts = log.get("counts", {})
        for cls in ALL_CLASSES:
            row[cls] = counts.get(cls, 0)

        row["total"] = log.get("total_in_frame", sum(counts.values()))
        row["total_seen"] = log.get("total_seen", row["total"])
        row["avg_speed_kmh"] = log.get("avg_speed_kmh", 0.0)
        row["density_label"] = _classify_density(row["total"])
        rows.append(row)

    df = pd.DataFrame(rows)
    # Add a human-readable timestamp column (relative to video start)
    df["time_str"] = df["timestamp_s"].apply(
        lambda s: str(timedelta(seconds=int(s)))
    )
    return df


def _classify_density(total: int) -> str:
    for label, (lo, hi) in DENSITY_LEVELS.items():
        if lo <= total < hi:
            return label
    return "congested"


# ─────────────────────────────────────────────────────────────
# 2. Insights Engine
# ─────────────────────────────────────────────────────────────

def compute_insights(df: pd.DataFrame) -> dict:
    """
    Compute high-level traffic insights from the DataFrame.

    Returns
    -------
    dict with keys:
        total_frames, total_vehicles_seen,
        avg_vehicles_per_frame, max_vehicles_in_frame,
        peak_frame, peak_timestamp,
        density_distribution (%), class_distribution (%),
        avg_fps, avg_inference_ms,
        congestion_periods (list of timestamp ranges)
    """
    if df.empty:
        return {}

    peak_idx = df["total"].idxmax()

    # Density breakdown
    density_counts = df["density_label"].value_counts()
    density_pct = (density_counts / len(df) * 100).round(1).to_dict()

    # Class distribution (total across all frames)
    class_totals = {cls: int(df[cls].sum()) for cls in ALL_CLASSES if cls in df.columns}
    grand_total  = max(sum(class_totals.values()), 1)
    class_pct    = {k: round(v / grand_total * 100, 1) for k, v in class_totals.items()}

    # Congestion periods (sequences of "heavy" or "congested" frames)
    congestion_periods = _find_congestion_windows(df)

    insights = {
        "total_frames":             len(df),
        "total_vehicles_seen":      int(df["total_seen"].max()) if "total_seen" in df.columns else int(df["total"].sum()),
        "avg_vehicles_per_frame":   round(float(df["total"].mean()), 2),
        "max_vehicles_in_frame":    int(df["total"].max()),
        "peak_frame":               int(df.loc[peak_idx, "frame_id"]),
        "peak_timestamp":           df.loc[peak_idx, "time_str"],
        "density_distribution_%":   density_pct,
        "class_distribution_%":     class_pct,
        "class_totals":             class_totals,
        "avg_fps":                  round(float(df["fps"].mean()), 1),
        "avg_inference_ms":         round(float(df["inference_ms"].mean()), 2),
        "congestion_periods":       congestion_periods,
    }
    return insights


def _find_congestion_windows(df: pd.DataFrame) -> list[dict]:
    """Return time windows where density is 'heavy' or 'congested'."""
    is_congested = df["density_label"].isin(["heavy", "congested"])
    windows = []
    in_window = False
    start_ts = None

    for idx, flag in zip(df["timestamp_s"], is_congested):
        if flag and not in_window:
            in_window = True
            start_ts  = idx
        elif not flag and in_window:
            in_window = False
            windows.append({
                "start_s": start_ts,
                "end_s":   idx,
                "start":   str(timedelta(seconds=int(start_ts))),
                "end":     str(timedelta(seconds=int(idx))),
            })
    if in_window:
        windows.append({
            "start_s": start_ts,
            "end_s":   df["timestamp_s"].iloc[-1],
            "start":   str(timedelta(seconds=int(start_ts))),
            "end":     df["time_str"].iloc[-1],
        })
    return windows


# ─────────────────────────────────────────────────────────────
# 3. Visualisation
# ─────────────────────────────────────────────────────────────

def plot_traffic_trends(
    df: pd.DataFrame,
    insights: dict,
    save_dir: Path | None = None,
) -> None:
    """
    Generate a 4-panel traffic analytics figure:
      1. Total vehicles over time (colour-coded by density)
      2. Per-class breakdown over time
      3. Density distribution (pie)
      4. Class distribution (bar)
    """
    save_dir = save_dir or config.PLOTS_DIR
    save_dir = Path(save_dir)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("🚗 Traffic Analytics Dashboard", fontsize=16, fontweight="bold", y=1.01)

    t = df["timestamp_s"].values

    # ── Panel 1: Total count + density background ────────────
    ax = axes[0, 0]
    for label, (lo, hi) in DENSITY_LEVELS.items():
        mask = (df["total"] >= lo) & (df["total"] < hi)
        if mask.any():
            ax.fill_between(t, 0, df["total"].where(mask),
                            color=DENSITY_COLORS[label], alpha=0.6, label=label)
    ax.plot(t, df["total"], color="white", lw=1, alpha=0.8)

    # Mark peak
    peak_t = df.loc[df["total"].idxmax(), "timestamp_s"]
    peak_v = df["total"].max()
    ax.annotate(f"Peak: {peak_v}",
                xy=(peak_t, peak_v), xytext=(peak_t + 1, peak_v + 1),
                fontsize=8, color="white",
                arrowprops=dict(arrowstyle="->", color="white", lw=0.8))

    ax.set_facecolor("#1a1a2e")
    ax.set_title("Vehicles in Frame Over Time", color="white")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Vehicle Count")
    ax.legend(loc="upper right", fontsize=7)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")

    # ── Panel 2: Per-class stacked area ──────────────────────
    ax = axes[0, 1]
    class_colors = ["#00b4d8", "#e63946", "#2a9d8f", "#f4a261", "#a8dadc", "#457b9d"]
    active_classes = [c for c in ALL_CLASSES if c in df.columns and df[c].sum() > 0]
    ax.stackplot(t, [df[c] for c in active_classes],
                 labels=active_classes, colors=class_colors[:len(active_classes)], alpha=0.85)
    ax.set_facecolor("#1a1a2e")
    ax.set_title("Per-Class Vehicle Breakdown", color="white")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Count")
    ax.legend(loc="upper right", fontsize=7)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")

    # ── Panel 3: Density distribution pie ────────────────────
    ax = axes[1, 0]
    density_pct = insights.get("density_distribution_%", {})
    if density_pct:
        labels = list(density_pct.keys())
        sizes  = list(density_pct.values())
        colors = [DENSITY_COLORS.get(l, "#aaa") for l in labels]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=140,
            textprops={"color": "white", "fontsize": 8},
        )
    ax.set_facecolor("#1a1a2e")
    ax.set_title("Density Distribution", color="white")

    # ── Panel 4: Class totals bar chart ──────────────────────
    ax = axes[1, 1]
    class_totals = insights.get("class_totals", {})
    cls_names = list(class_totals.keys())
    cls_vals  = list(class_totals.values())
    bars = ax.bar(cls_names, cls_vals, color=class_colors[:len(cls_names)], edgecolor="none")
    for bar, val in zip(bars, cls_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=8, color="white")
    ax.set_facecolor("#1a1a2e")
    ax.set_title("Total Detections by Class", color="white")
    ax.set_ylabel("Count")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")

    fig.patch.set_facecolor("#0f0f1a")
    plt.tight_layout()
    out = save_dir / "traffic_analytics_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.show()
    print(f"[Analytics] Dashboard saved → {out}")


def plot_fps_profile(df: pd.DataFrame, save_dir: Path | None = None) -> None:
    """Plot inference FPS over time to verify real-time performance."""
    save_dir = save_dir or config.PLOTS_DIR
    fig, ax  = plt.subplots(figsize=(12, 3))
    ax.plot(df["timestamp_s"], df["fps"], color="#00b4d8", lw=1)
    ax.axhline(30, color="#2ecc71", lw=1, ls="--", label="30 FPS target")
    ax.set_facecolor("#1a1a2e"); fig.patch.set_facecolor("#0f0f1a")
    ax.set_title("Inference FPS Profile", color="white")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("FPS")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    ax.legend(fontsize=8)
    out = Path(save_dir) / "fps_profile.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print(f"[Analytics] FPS profile saved → {out}")


# ─────────────────────────────────────────────────────────────
# 4. Report Export
# ─────────────────────────────────────────────────────────────

def save_report(
    df: pd.DataFrame,
    insights: dict,
    csv_path: Path | None = None,
    json_path: Path | None = None,
) -> None:
    """
    Save the full DataFrame as CSV and the insights dict as JSON.
    """
    csv_path  = csv_path  or (config.REPORTS_DIR / "traffic_log.csv")
    json_path = json_path or (config.REPORTS_DIR / "traffic_insights.json")

    df.to_csv(csv_path, index=False)
    print(f"[Analytics] CSV saved → {csv_path}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2, ensure_ascii=False)
    print(f"[Analytics] JSON saved → {json_path}")


def print_insights(insights: dict) -> None:
    """Pretty-print the insights dict to console."""
    print("\n" + "═" * 50)
    print("  🚦 TRAFFIC ANALYTICS SUMMARY")
    print("═" * 50)
    print(f"  Total frames analysed : {insights.get('total_frames', '—')}")
    print(f"  Total vehicles seen   : {insights.get('total_vehicles_seen', '—')}")
    print(f"  Avg vehicles/frame    : {insights.get('avg_vehicles_per_frame', '—')}")
    print(f"  Peak count            : {insights.get('max_vehicles_in_frame', '—')} "
          f"@ {insights.get('peak_timestamp', '—')}")
    print(f"  Avg FPS               : {insights.get('avg_fps', '—')}")
    print(f"  Avg inference         : {insights.get('avg_inference_ms', '—')} ms")
    print("\n  Density distribution:")
    for k, v in insights.get("density_distribution_%", {}).items():
        bar = "█" * int(v / 5)
        print(f"    {k:<12}: {bar:<20} {v}%")
    print("\n  Class totals:")
    for k, v in insights.get("class_totals", {}).items():
        print(f"    {k:<12}: {v}")
    congestion = insights.get("congestion_periods", [])
    if congestion:
        print(f"\n  ⚠️  Congestion windows ({len(congestion)} found):")
        for w in congestion[:5]:
            print(f"    {w['start']} → {w['end']}")
    print("═" * 50 + "\n")

def generate_ai_summary(insights: dict) -> str:
    """
    Generates a natural language summary based on the calculated insights.
    """
    total = insights.get("total_vehicles_seen", 0)
    avg_dens = insights.get("avg_vehicles_per_frame", 0)
    peak = insights.get("max_vehicles_in_frame", 0)
    class_totals = insights.get("class_totals", {})
    density_pct = insights.get("density_distribution_%", {})
    
    # Analyze Heavy Vehicles
    heavy = class_totals.get("truck", 0) + class_totals.get("bus", 0)
    heavy_ratio = (heavy / total) * 100 if total > 0 else 0
    
    # Determine dominant traffic state
    dominant_state = max(density_pct, key=density_pct.get) if density_pct else "Low"
    
    report = f"### 🤖 Automated AI Traffic Report\n\n"
    report += f"The AV perception stack processed the video successfully, identifying a total of **{total} unique vehicles**. "
    report += f"The traffic density was predominantly **{dominant_state}**, with an average of **{avg_dens} vehicles per frame**.\n\n"
    
    if heavy_ratio > 15:
        report += f"- 🚛 **Commercial Activity:** High presence of heavy vehicles detected ({heavy_ratio:.1f}% of total traffic), indicating significant industrial routing.\n"
    elif heavy_ratio > 0:
        report += f"- 🚗 **Light Traffic:** The majority of the traffic consisted of passenger vehicles, with only {heavy_ratio:.1f}% heavy vehicles.\n"
        
    if peak > 8:
        report += f"- 🚨 **Congestion Peaks:** Traffic peaked at **{peak} vehicles** simultaneously on screen, suggesting potential bottleneck events.\n"
    else:
        report += f"- ✅ **Smooth Flow:** Traffic remained smooth throughout the sequence with a peak of only {peak} vehicles.\n"
        
    collisions = insights.get("collision_warnings", 0)
    if collisions > 0:
        report += f"- ⚠️ **Safety Alert:** Detected **{collisions} collision warning(s)** where vehicles approached dangerously close to the camera.\n"
    else:
        report += f"- 🛡️ **Safety:** Zero collision risks detected. Safe distance maintained by all vehicles.\n"
        
    report += f"- ⚡ **System Status:** The perception stack operated efficiently at an average of **{insights.get('avg_fps', 0)} FPS**."
    
    return report

