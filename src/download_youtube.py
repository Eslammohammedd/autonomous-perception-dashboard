import yt_dlp
import os
from pathlib import Path

def download_youtube_video(url=None):
    # Default high-quality 4K Dashcam video if none provided
    if url is None:
        url = "https://youtube.com/shorts/yKjXmCQF-iI?si=QG3rpOQJAzavl1h7"
    
    target_dir = Path("Autonomous Vehicle Perception Module/data/demo")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    output_template = str(target_dir / "%(title)s_%(id)s.%(ext)s")
    
    ydl_opts = {
        # Use 'best' to get a single file that doesn't require ffmpeg merging
        'format': 'best[ext=mp4]/best',
        'outtmpl': output_template,
        'quiet': False,
        'no_warnings': True,
    }
    
    print(f"🌍 Downloading YouTube video from: {url}...")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filepath = ydl.prepare_filename(info)
        print(f"✅ Download complete! Video saved in: {filepath}")
        return str(filepath)
    except Exception as e:
        print(f"❌ Error downloading from YouTube: {e}")
        return None

if __name__ == "__main__":
    # You can change the URL here
    download_youtube_video()
