"""
yolo_detector.py — YOLOv8 Real-Time Vehicle Detection Engine.

Provides:
  - VehicleDetector : wraps YOLOv8 for vehicle/pedestrian detection
  - annotate_frame() : draws bounding boxes + labels on frames
  - run_on_video()   : processes a video file and returns per-frame analytics

Requirements:
    pip install ultralytics opencv-python
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import cv2
import numpy as np
from collections import defaultdict

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False
    print("[WARNING] ultralytics not installed. Run: pip install ultralytics")

from src import config


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

# Filter class IDs from COCO/BDD100K that map to road-relevant objects
VEHICLE_CLASS_IDS = {
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    0:  "person",
    1:  "bicycle",
    9:  "traffic light",
    11: "stop sign",
}

# Color palette per class (BGR for OpenCV)
CLASS_COLORS = {
    "car":           (0, 200, 255),   # amber
    "truck":         (255, 80,  80),  # red
    "bus":           (80,  80, 255),  # blue
    "motorcycle":    (80, 255, 80),   # green
    "bicycle":       (255, 255, 0),   # cyan
    "person":        (255, 0,  200),  # magenta
    "traffic light": (200, 255, 0),   # yellow-green
    "stop sign":     (0, 0, 255),     # bright red
}


# ─────────────────────────────────────────────────────────────
# 1. VehicleDetector
# ─────────────────────────────────────────────────────────────

class VehicleDetector:
    """
    YOLOv8-based vehicle and pedestrian detector.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        device: str | None = None,
    ):
        if not YOLO_OK:
            raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")

        # Load defaults from config if not provided
        self.model_path = model_path or config.MODEL_PATH
        self.device     = device or str(config.DEVICE)
        self.conf       = conf_threshold or config.CONF_THRESHOLD
        self.iou        = iou_threshold or config.IOU_THRESHOLD

        print(f"[VehicleDetector] Loading {self.model_path} on {self.device} ...")
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        print(f"[VehicleDetector] Ready ✅")

    # ----------------------------------------------------------
    def detect(self, frame: np.ndarray, imgsz: int = 640):
        """
        Run inference on a single BGR frame (OpenCV format).

        Returns
        -------
        results : ultralytics Results object
        """
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=imgsz,
            verbose=False,
            device=self.device,
        )
        return results[0]

    # ----------------------------------------------------------
    def parse_detections(self, results) -> list[dict]:
        """
        Parse ultralytics results into a clean list of dicts.

        Returns
        -------
        list of {class_name, confidence, bbox: [x1,y1,x2,y2]}
        """
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls.item())
            if cls_id not in VEHICLE_CLASS_IDS:
                continue
            detections.append({
                "class_id":   cls_id,
                "class_name": VEHICLE_CLASS_IDS[cls_id],
                "confidence": round(float(box.conf.item()), 3),
                "bbox":       [int(v) for v in box.xyxy[0].tolist()],
            })
        return detections

    # ----------------------------------------------------------
    def count_by_class(self, detections: list[dict]) -> dict[str, int]:
        """
        Count detected objects per class.

        Returns
        -------
        dict e.g. {'car': 5, 'truck': 2, 'person': 1}
        """
        counts: dict[str, int] = defaultdict(int)
        for det in detections:
            counts[det["class_name"]] += 1
        return dict(counts)

    # ----------------------------------------------------------
    def annotate(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on a copy of the frame.
        """
        return annotate_frame(frame, detections)

    # ----------------------------------------------------------
    def get_inference_speed(self, results) -> float:
        """Return inference time in milliseconds for the last frame."""
        try:
            return results.speed.get("inference", 0.0)
        except AttributeError:
            return 0.0


# ─────────────────────────────────────────────────────────────
# 2. Frame Annotation
# ─────────────────────────────────────────────────────────────

def annotate_frame(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Draw bounding boxes + class labels + confidence scores on a frame.

    Parameters
    ----------
    frame      : BGR numpy array (H, W, 3)
    detections : list from VehicleDetector.parse_detections()

    Returns
    -------
    Annotated frame copy.
    """
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["class_name"]
        conf  = det["confidence"]
        color = CLASS_COLORS.get(label, (200, 200, 200))

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label background
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
        )
    return out


def draw_stats_overlay(
    frame: np.ndarray,
    counts: dict[str, int],
    total_vehicles: int,
    fps: float,
    frame_idx: int,
) -> np.ndarray:
    """
    Draw a semi-transparent stats panel in the top-left corner.
    Shows: FPS, total count, per-class counts.
    """
    out = frame.copy()
    panel_h = 30 + 22 * (len(counts) + 2)
    overlay  = out.copy()
    cv2.rectangle(overlay, (0, 0), (220, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    y = 20
    cv2.putText(out, f"Frame: {frame_idx:05d}  FPS: {fps:.1f}",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    y += 22
    cv2.putText(out, f"Total Vehicles: {total_vehicles}",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 200), 1)
    y += 22
    for cls_name, cnt in sorted(counts.items()):
        color = CLASS_COLORS.get(cls_name, (200, 200, 200))
        cv2.putText(out, f"  {cls_name:<12}: {cnt}",
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        y += 20
    return out


# ─────────────────────────────────────────────────────────────
# 3. Video Processing Pipeline
# ─────────────────────────────────────────────────────────────

def run_on_video(
    video_path: str,
    detector: "VehicleDetector",
    output_path: str | None = None,
    max_frames: int | None = None,
    show_window: bool = False,
) -> list[dict]:
    """
    Process a full video file frame-by-frame.

    Parameters
    ----------
    video_path  : Path to the input video (.mp4 / .avi / etc.)
    detector    : Initialised VehicleDetector instance
    output_path : If given, save annotated video here
    max_frames  : Stop after this many frames (None = full video)
    show_window : Display live preview (requires display)

    Returns
    -------
    frame_logs : list of per-frame dicts
        [{ frame_id, timestamp_s, counts, total_in_frame, fps }, ...]

    These logs are directly consumed by analytics.py.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    # Video properties
    fps_src  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Video] {video_path}  |  {w}x{h}  |  {fps_src:.1f} FPS  |  {total_fr} frames")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_src, (w, h))

    frame_logs: list[dict] = []
    frame_idx  = 0
    t_start    = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        # ── Detect ──────────────────────────────────────────
        results    = detector.detect(frame)
        detections = detector.parse_detections(results)
        counts     = detector.count_by_class(detections)

        # ── FPS calculation ──────────────────────────────────
        elapsed = time.perf_counter() - t_start
        live_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0

        # ── Log ─────────────────────────────────────────────
        frame_logs.append({
            "frame_id":       frame_idx,
            "timestamp_s":    round(frame_idx / fps_src, 3),
            "counts":         dict(counts),
            "total_in_frame": sum(counts.values()),
            "fps":            round(live_fps, 1),
            "inference_ms":   detector.get_inference_speed(results),
        })

        # ── Annotate & Write ────────────────────────────────
        annotated = detector.annotate(frame, detections)
        annotated = draw_stats_overlay(annotated, counts, sum(counts.values()), live_fps, frame_idx)

        if writer:
            writer.write(annotated)
        if show_window:
            cv2.imshow("AV Perception — YOLOv8", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

        # Progress every 100 frames
        if frame_idx % 100 == 0:
            print(f"  [{frame_idx}/{total_fr}]  FPS={live_fps:.1f}  "
                  f"Vehicles in frame={sum(counts.values())}")

    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()

    print(f"[Done] Processed {frame_idx} frames. "
          f"Avg FPS={frame_idx/elapsed:.1f}")
    return frame_logs


# ─────────────────────────────────────────────────────────────
# 4. Quick Test (run this file directly)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import urllib.request, os

    # Download a short test clip if no local video is available
    TEST_VIDEO = str(config.DATA_DIR / "demo" / "test_traffic.mp4")
    SAMPLE_URL = (
        "https://commondatastorage.googleapis.com/"
        "gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
    )

    if not Path(TEST_VIDEO).exists():
        print(f"[Test] Downloading sample video → {TEST_VIDEO}")
        urllib.request.urlretrieve(SAMPLE_URL, TEST_VIDEO)

    detector = VehicleDetector(model_path="yolov8n.pt", conf_threshold=0.40)

    output_path = str(config.RESULTS_DIR / "plots" / "test_detection_output.mp4")
    logs = run_on_video(
        video_path=TEST_VIDEO,
        detector=detector,
        output_path=output_path,
        max_frames=200,
        show_window=False,
    )

    print(f"\n[Test] Logged {len(logs)} frames.")
    print(f"[Test] Sample log entry:\n  {logs[50] if len(logs) > 50 else logs[-1]}")
    print(f"[Test] Output saved → {output_path}")
