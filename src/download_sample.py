import requests
import os
from pathlib import Path

def download_sample_video():
    # URL to a high-quality, free-to-use traffic video from Pexels/Pixabay direct link
    # This is a sample dashcam video for testing autonomous driving perception
    video_url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4"
    
    target_dir = Path("Autonomous Vehicle Perception Module/data/demo")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = target_dir / "test_traffic.mp4"
    
    print(f"Downloading sample video to: {target_path}...")
    
    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Download complete! You can now run the multi_perception.py script.")
    except Exception as e:
        print(f"Error downloading video: {e}")

if __name__ == "__main__":
    download_sample_video()
