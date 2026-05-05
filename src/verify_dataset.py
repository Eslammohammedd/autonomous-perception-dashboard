"""
verify_dataset.py — Verify and Arrange BDD100K Dataset for Training.

Run this after downloading the dataset from Kaggle and extracting it 
into 'datasets/bdd100k'.
"""

import os
from pathlib import Path

def main():
    root_dir = Path(__file__).resolve().parent.parent.parent
    dataset_dir = root_dir / "datasets" / "bdd100k"
    
    print(f"🔍 Checking dataset at: {dataset_dir}")
    
    expected_structure = [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val"
    ]
    
    missing = []
    for rel_path in expected_structure:
        full_path = dataset_dir / rel_path
        if not full_path.exists():
            missing.append(rel_path)
    
    if not missing:
        print("✅ Dataset structure is CORRECT!")
        print("🚀 You can now run: python 'Autonomous Vehicle Perception Module/src/train_yolo.py'")
    else:
        print("❌ Missing directories:")
        for m in missing:
            print(f"   - {m}")
        print("\n💡 Tip: If you downloaded from Kaggle, the folders might be named differently.")
        print("   Make sure to rename or move them so the path is:")
        print(f"   {dataset_dir}/images/train, etc.")
        
        # Check if they are inside a subfolder (common with Kaggle zips)
        subfolders = [f for f in dataset_dir.iterdir() if f.is_dir()]
        if subfolders:
            print("\n📂 Found subfolders that might contain the data:")
            for s in subfolders:
                print(f"   - {s.name}")

if __name__ == "__main__":
    main()
