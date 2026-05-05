"""
multi_perception.py — The Ultimate Autonomous Vehicle Perception Stack.

Combines:
1. Object Detection (Cars, Pedestrians, etc.) -> Project 'phuzv/1'
2. Semantic Segmentation (Drivable Road Area) -> Project '2kvox/2'
3. Real-time Visualization

Powered by Roboflow Inference API.
"""

import cv2
import numpy as np
from roboflow import Roboflow
import sys
from pathlib import Path

# Add root to sys path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
API_KEY = "eTrSU1WAJfSUcSCttQQ4"  # Using the key from your screenshot

# Project IDs
DETECTION_PROJECT    = "autonomous-driving-phuzv"
DETECTION_VERSION    = 1
SEGMENTATION_PROJECT = "autonomous-driving-2kvox"
SEGMENTATION_VERSION = 2

class MultiPerceptionStack:
    def __init__(self):
        print("[System] Initializing Multi-Perception Stack...")
        self.rf = Roboflow(api_key=API_KEY)
        
        # Initialize Detection Model
        print(f"[System] Loading Detection Model: {DETECTION_PROJECT}...")
        self.det_project = self.rf.workspace().project(DETECTION_PROJECT)
        self.det_model   = self.det_project.version(DETECTION_VERSION).model
        
        # Initialize Segmentation Model
        print(f"[System] Loading Segmentation Model: {SEGMENTATION_PROJECT}...")
        self.seg_project = self.rf.workspace().project(SEGMENTATION_PROJECT)
        self.seg_model   = self.seg_project.version(SEGMENTATION_VERSION).model
        
        print("[System] Stack Ready ✅")

    def process_frame(self, frame):
        # 1. Get Predictions from both models
        # We use .json() to get raw data for manual drawing
        det_results = self.det_model.predict(frame, confidence=40).json()
        seg_results = self.seg_model.predict(frame, confidence=40).json()
        
        annotated_frame = frame.copy()
        
        # 2. Draw Segmentation (Road)
        # Segmentation masks in Roboflow JSON often come as polygons or a mask
        if "predictions" in seg_results:
            overlay = annotated_frame.copy()
            for pred in seg_results["predictions"]:
                if "points" in pred:
                    points = np.array([[p['x'], p['y']] for p in pred["points"]], np.int32)
                    cv2.fillPoly(overlay, [points], (0, 255, 0)) # Green road
            # Blend overlay with 30% transparency
            cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)

        # 3. Draw Detection (Boxes)
        if "predictions" in det_results:
            for pred in det_results["predictions"]:
                x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
                label = pred["class"]
                conf  = pred["confidence"]
                
                # Calculate box corners
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                
                # Draw box
                color = (0, 200, 255) if label == "Cars" else (255, 80, 80)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                txt = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1, y1-th-5), (x1+tw, y1), color, -1)
                cv2.putText(annotated_frame, txt, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated_frame

def run_demo(video_path):
    stack = MultiPerceptionStack()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return

    print("🚀 Starting Live Perception... Press 'q' to stop.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize for faster API processing if needed
        # frame = cv2.resize(frame, (640, 480))
        
        processed = stack.process_frame(frame)
        
        cv2.imshow("Multi-Perception: Road + Objects", processed)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Check for youtube_traffic.mp4 first, then test_traffic.mp4
    yt_video = str(config.DATA_DIR / "demo" / "youtube_traffic.mp4")
    test_video = str(config.DATA_DIR / "demo" / "test_traffic.mp4")
    
    if Path(yt_video).exists():
        print(f"🎬 Running on YouTube sample: {yt_video}")
        run_demo(yt_video)
    elif Path(test_video).exists():
        print(f"🎬 Running on local sample: {test_video}")
        run_demo(test_video)
    else:
        print("❌ Error: No sample videos found in data/demo/.")
        print("   Please run download_youtube.py first.")
