import time
import os
import ctypes
from pathlib import Path

def notify_completion():
    # Windows Message Box
    title = "YOLOv8 Training Complete! 🚀"
    message = (
        "Congratulations! The full VisDrone model training has finished.\n\n"
        "Weights are saved in: runs/detect/AV_Perception/Full_VisDrone_Model/weights/best.pt\n\n"
        "You can now run the Dashboard to see your professional AI in action."
    )
    # MB_OK | MB_ICONINFORMATION
    ctypes.windll.user32.MessageBoxW(0, message, title, 0x40 | 0x0)

def monitor_training():
    # The final results.csv is a better indicator of full completion
    results_file = Path("runs/detect/AV_Perception/Full_VisDrone_Model/results.csv")
    target_file = Path("runs/detect/AV_Perception/Full_VisDrone_Model/weights/best.pt")
    
    print(f"Monitoring for REAL completion... Waiting for final epoch in results.csv")
    
    while True:
        if results_file.exists() and target_file.exists():
            # Check if we have reached the expected number of epochs (50)
            try:
                with open(results_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 51: # Header + 50 epochs
                        break
            except:
                pass
        time.sleep(60) # Check every minute
        
    print("Full training detected! Sending real notification...")
    notify_completion()
    print("📢 Notification sent.")

if __name__ == "__main__":
    monitor_training()
