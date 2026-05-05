"""
train_yolo.py — Advanced Training Script for BDD100K using YOLOv8.

Optimized for NVIDIA RTX 4060 (Laptop) using CUDA.
This script performs Fine-tuning on the BDD100K dataset.
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

# Add root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    # 1. Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Training on CPU will be extremely slow.")
        return
    
    device_name = torch.cuda.get_device_name(0)
    print(f"🚀 Training on: {device_name}")
    print(f"📦 CUDA Version: {torch.version.cuda}")

    # 2. Path to our configuration
    yaml_path = Path(__file__).resolve().parent.parent / "bdd100k.yaml"
    if not yaml_path.exists():
        print(f"❌ Configuration file not found at: {yaml_path}")
        return

    # 3. Load Model
    # We start with 'yolov8n.pt' (nano) for high speed, 
    # or 'yolov8s.pt' (small) for better accuracy.
    model = YOLO("yolov8n.pt") 

    # 4. Start Training
    print("\n⚡ Starting Training Phase...")
    results = model.train(
        data="coco8.yaml",  # Auto-downloads a tiny 8-image dataset for testing
        epochs=50,          # Start with 50 epochs
        imgsz=640,          # Standard resolution for BDD100K
        batch=16,           # Optimal for RTX 4060 (8GB VRAM)
        device=0,           # Use GPU 0
        project="AV_Perception",
        name="bdd100k_finetune",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",  # Better convergence for fine-tuning
        lr0=0.01,           # Initial learning rate
        patience=10,        # Early stopping if no improvement
        workers=4,          # DataLoader workers
        amp=True            # Automatic Mixed Precision (saves VRAM & time)
    )

    print("\n✅ Training Complete!")
    print(f"🏆 Best model saved at: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    main()
