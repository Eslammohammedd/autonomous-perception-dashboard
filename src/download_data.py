"""
download_data.py — Downloader for BDD100K Sample (YOLOv8 Formatted).

This script downloads a curated subset of BDD100K to get you started 
immediately without downloading the full 100GB dataset.
"""

import os
import zipfile
import urllib.request
from pathlib import Path

def download_progress(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        print(f"\r📥 Downloading: {percent:5.1f}% [{read_so_far / 1e6:.1f}MB / {total_size / 1e6:.1f}MB]", end="")
    else:
        print(f"\r📥 Downloading: {read_so_far / 1e6:.1f}MB", end="")

def main():
    # 1. Setup Paths
    # We want to place datasets in the root folder alongside the project
    root_dir = Path(__file__).resolve().parent.parent.parent
    dataset_dir = root_dir / "datasets" / "bdd100k"
    zip_path = root_dir / "datasets" / "bdd_sample.zip"

    dataset_dir.mkdir(parents=True, exist_ok=True)

    # 2. URL for a YOLO-ready BDD100K subset (Hosted on a reliable server)
    # Using a sample from a public repo or a direct drive link
    # Here we use a 1000-image curated subset
    url = "https://github.com/alex-p-93/bdd100k-yolov8-subset/releases/download/v1.0/bdd100k_subset_yolo.zip"

    if not (dataset_dir / "images").exists():
        print(f"🌍 Target: {dataset_dir}")
        print("🔗 Starting download from GitHub Release (Approx 180MB)...")
        
        try:
            urllib.request.urlretrieve(url, zip_path, download_progress)
            print("\n✅ Download complete!")

            print("📦 Extracting files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir.parent) # Extract into datasets/ (contains bdd100k/)
            
            print("🧹 Cleaning up...")
            os.remove(zip_path)
            print("✨ BDD100K Sample is ready in 'datasets/bdd100k/'")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\n💡 Alternative: Please download a YOLO-formatted BDD100K dataset manually from Kaggle")
            print("   and place it in: " + str(dataset_dir))
    else:
        print("✅ Dataset already exists. You are ready to train!")

if __name__ == "__main__":
    main()
