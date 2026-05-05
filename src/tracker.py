"""
tracker.py — Multi-Object Vehicle Tracking using ByteTrack / DeepSORT.

Provides:
  - VehicleTracker   : wraps deep-sort-realtime for per-ID tracking
  - estimate_speed() : pixel-displacement-based speed approximation
  - draw_tracks()    : render track trails + IDs on frames

Requirements:
    pip install deep-sort-realtime
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
from collections import defaultdict, deque

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_OK = True
except ImportError:
    DEEPSORT_OK = False
    print("[WARNING] deep-sort-realtime not installed.")
    print("          Run: pip install deep-sort-realtime")

from src import config


# ─────────────────────────────────────────────────────────────
# 1. VehicleTracker
# ─────────────────────────────────────────────────────────────

class VehicleTracker:
    """
    Multi-object tracker for vehicles using DeepSORT.

    Wraps deep-sort-realtime to assign persistent IDs across frames,
    count unique vehicles that have ever appeared, and maintain
    trajectory trails for visualization.

    Parameters
    ----------
    max_age     : Frames to keep a track alive with no detection.
    n_init      : Frames needed to confirm a new track.
    nn_budget   : Max size of the appearance descriptor gallery.
    max_cosine_distance : Cosine distance threshold for re-identification.
    trail_len   : Number of past positions stored per track (for drawing).

    Example
    -------
    >>> tracker = VehicleTracker()
    >>> for frame, detections in frames:
    ...     tracks, total_seen = tracker.update(detections, frame)
    ...     annotated = tracker.draw_trails(frame, tracks)
    """

    def __init__(
        self,
        max_age: int = 60,         # Keep track alive longer during occlusions
        n_init: int = 5,           # Wait for more frames to confirm a real vehicle
        nn_budget: int = 100,
        max_cosine_distance: float = 0.4,
        trail_len: int = 40,
    ):
        if not DEEPSORT_OK:
            raise RuntimeError(
                "deep-sort-realtime not installed. "
                "Run: pip install deep-sort-realtime"
            )

        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nn_budget=nn_budget,
            max_cosine_distance=max_cosine_distance,
            embedder="mobilenet",      # lightweight re-ID backbone
            half=True,                 # FP16 on RTX 4060 for speed
            embedder_gpu=True,
        )

        self.all_ids: set[int]            = set()   # all unique IDs ever seen
        self.trails: dict[int, deque]     = defaultdict(lambda: deque(maxlen=trail_len))
        self.class_map: dict[int, str]    = {}      # track_id → class_name
        self._trail_len = trail_len

        print("[VehicleTracker] DeepSORT ready ✅")

    # ----------------------------------------------------------
    def update(
        self,
        detections: list[dict],
        frame: np.ndarray,
    ) -> tuple[list, int]:
        """
        Feed current-frame detections to the tracker.

        Parameters
        ----------
        detections : list from VehicleDetector.parse_detections()
            Each dict must have: bbox [x1,y1,x2,y2], confidence, class_name
        frame      : BGR frame (used by DeepSORT re-ID embedder)

        Returns
        -------
        tracks     : list of confirmed Track objects (deep_sort_realtime)
        total_seen : total unique vehicle IDs seen so far (counter)
        """
        # Convert to DeepSORT input: ([x1,y1,w,h], confidence, class_name)
        ds_input = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w, h = x2 - x1, y2 - y1
            ds_input.append(
                ([x1, y1, w, h], det["confidence"], det["class_name"])
            )

        raw_tracks = self.tracker.update_tracks(ds_input, frame=frame)

        confirmed = []
        for track in raw_tracks:
            if not track.is_confirmed():
                continue

            tid = track.track_id
            self.all_ids.add(tid)

            # Store class label
            if track.det_class and tid not in self.class_map:
                self.class_map[tid] = track.det_class

            # Store centre-point for trail
            ltrb = track.to_ltrb()
            cx = int((ltrb[0] + ltrb[2]) / 2)
            cy = int((ltrb[1] + ltrb[3]) / 2)
            self.trails[tid].append((cx, cy))

            confirmed.append(track)
        
        # Calculate unique count of ONLY vehicles (exclude people)
        vehicle_ids = {
            tid for tid, cls in self.class_map.items() 
            if cls in ["car", "truck", "bus", "motorcycle", "bicycle"]
        }

        return confirmed, len(vehicle_ids)

    # ----------------------------------------------------------
    def draw_trails(
        self,
        frame: np.ndarray,
        tracks: list,
    ) -> np.ndarray:
        """
        Render track ID labels and trajectory trails on a frame copy.
        """
        return draw_tracks(frame, tracks, self.trails, self.class_map)

    # ----------------------------------------------------------
    def get_class_counts(self, tracks: list) -> dict[str, int]:
        """Count active (confirmed) tracks per class this frame."""
        counts: dict[str, int] = defaultdict(int)
        for t in tracks:
            cls = self.class_map.get(t.track_id, "unknown")
            counts[cls] += 1
        return dict(counts)

    # ----------------------------------------------------------
    def reset(self):
        """Reset tracker state (call between unrelated videos)."""
        self.all_ids.clear()
        self.trails.clear()
        self.class_map.clear()
        self.tracker = DeepSort(
            max_age=30, n_init=3, nn_budget=100,
            embedder="mobilenet", half=True, embedder_gpu=True,
        )


# ─────────────────────────────────────────────────────────────
# 2. Speed Estimation (Pixel Displacement)
# ─────────────────────────────────────────────────────────────

def estimate_speed(
    trails: dict[int, deque],
    fps: float,
    px_per_meter: float = 10.0,
) -> dict[int, float]:
    """
    Approximate speed (km/h) for each tracked ID using pixel displacement.

    Note: This is a rough estimate.  Accurate speed requires camera
    calibration (homography) or a reference distance in the scene.

    Parameters
    ----------
    trails       : {track_id: deque of (cx, cy) centre-points}
    fps          : Video frame rate (used to convert frames → seconds)
    px_per_meter : Calibration constant (pixels per real-world meter).
                   Default 10 is a placeholder — calibrate for your camera.

    Returns
    -------
    {track_id: speed_kmh}
    """
    speeds = {}
    for tid, trail in trails.items():
        if len(trail) < 2:
            speeds[tid] = 0.0
            continue
        # Use last two points
        p1, p2 = trail[-2], trail[-1]
        px_dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        meters  = px_dist / px_per_meter
        mps     = meters * fps          # metres per second
        kmh     = mps * 3.6
        speeds[tid] = round(kmh, 1)
    return speeds


# ─────────────────────────────────────────────────────────────
# 3. Visualization
# ─────────────────────────────────────────────────────────────

# Colour palette for track IDs (cycles every 20 IDs)
_TRACK_PALETTE = [
    (255, 128, 0),  (0, 200, 255),  (255, 0, 128),  (0, 255, 128),
    (200, 0, 255),  (255, 200, 0),  (0, 128, 255),  (128, 255, 0),
    (255, 0, 200),  (0, 255, 200),  (128, 0, 255),  (255, 128, 128),
    (0, 128, 128),  (128, 128, 0),  (200, 128, 255),(255, 80,  80),
    (80, 255, 80),  (80, 80, 255),  (200, 200, 0),  (0, 200, 200),
]

def _track_color(tid) -> tuple[int, int, int]:
    """
    Consistently map a track ID (int or string) to a color from the palette.
    """
    try:
        # DeepSORT often returns IDs as strings like "1", "2"...
        itid = int(tid)
    except (ValueError, TypeError):
        # Fallback for non-numeric strings
        itid = hash(tid)
    
    return _TRACK_PALETTE[itid % len(_TRACK_PALETTE)]


def draw_tracks(
    frame: np.ndarray,
    tracks: list,
    trails: dict[int, deque],
    class_map: dict[int, str],
    speeds: dict[int, float] = None,
) -> np.ndarray:
    """
    Draw bounding boxes, IDs, trail lines, and class labels for all
    confirmed tracks on a copy of `frame`.
    """
    out = frame.copy()

    for track in tracks:
        tid   = track.track_id
        color = _track_color(tid)
        ltrb  = track.to_ltrb()
        x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label
        cls_name = class_map.get(tid, "vehicle")
        speed    = speeds.get(tid, 0.0) if speeds else 0.0
        
        label    = f"#{tid} {cls_name}"
        if speed > 0:
            label += f" | {speed} km/h"
            
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Trail lines
    for tid, trail in trails.items():
        pts = list(trail)
        color = _track_color(tid)
        for i in range(1, len(pts)):
            alpha = i / len(pts)          # fades towards tail
            c = tuple(int(v * alpha) for v in color)
            cv2.line(out, pts[i - 1], pts[i], c, 2)

    return out


# ─────────────────────────────────────────────────────────────
# 4. Quick Test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run detection + tracking on a local video.

    Usage:
        python src/tracker.py
    """
    from src.yolo_detector import VehicleDetector, run_on_video

    VIDEO = str(config.DATA_DIR / "demo" / "test_traffic.mp4")
    if not Path(VIDEO).exists():
        print(f"[Test] Video not found at {VIDEO}")
        print("       Run yolo_detector.py first to download sample video.")
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0

    detector = VehicleDetector("yolov8n.pt", conf_threshold=0.40)
    tracker  = VehicleTracker()

    out_path = str(config.RESULTS_DIR / "plots" / "test_tracking_output.mp4")
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))

    frame_idx = 0
    while frame_idx < 200:
        ret, frame = cap.read()
        if not ret:
            break

        results    = detector.detect(frame)
        detections = detector.parse_detections(results)
        tracks, total_seen = tracker.update(detections, frame)

        speeds  = estimate_speed(tracker.trails, fps_src)
        annotated = tracker.draw_trails(frame, tracks)
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[Test] Tracking done — total unique vehicles: {total_seen}")
    print(f"[Test] Output saved → {out_path}")
