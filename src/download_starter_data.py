"""
download_starter_data.py — Instant Vehicle Detection Dataset Downloader.

Downloads a small, reliable dataset (COCO128 subset) so you can train 
immediately without waiting hours for 1.5GB files.
"""

import os
import zipfile
import urllib.request
import shutil
from pathlib import Path

def main():
    root_dir = Path(__file__).resolve().parent.parent
    dataset_dir = root_dir / "datasets" / "av_starter"
    zip_path = root_dir / "datasets" / "av_starter.zip"
    
    os.makedirs(root_dir / "datasets", exist_ok=True)
    
    print("🚀 Downloading 'Starter Vehicle Dataset' (22 MB) - Takes 3 seconds...")
    url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("✅ Download complete! Extracting...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root_dir / "datasets")
            
        # Rename coco128 to av_starter
        coco_dir = root_dir / "datasets" / "coco128"
        if coco_dir.exists():
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            coco_dir.rename(dataset_dir)
            
        os.remove(zip_path)
        
        # Create a custom yaml for Autonomous Vehicles
        yaml_content = f"""
path: {dataset_dir.absolute()} # dataset root dir
train: images/train2017 # train images
val: images/train2017 # val images

# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  9: traffic light
  10: fire hydrant
  11: stop sign
"""
        yaml_path = root_dir / "av_starter.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)
            
        print("✨ Dataset is READY!")
        print(f"📁 Location: {dataset_dir}")
        print("✅ You can now start training instantly.")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
