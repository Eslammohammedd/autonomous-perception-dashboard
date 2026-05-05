<div align="center">
  <img src="https://img.icons8.com/fluency/96/artificial-intelligence.png" alt="AI Icon" width="80"/>
  <h1>🚗 Autonomous Vehicle Perception Module</h1>
  <p><strong>Next-Generation Traffic Intelligence & Object Tracking Stack</strong></p>

  <p>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version"/></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-Optimized-EE4C2C.svg" alt="PyTorch"/></a>
    <a href="https://github.com/ultralytics/ultralytics"><img src="https://img.shields.io/badge/YOLO-v8-blueviolet.svg" alt="YOLOv8"/></a>
    <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg" alt="Streamlit"/></a>
  </p>
</div>

> **Note to Recruiters & Reviewers:** This project demonstrates advanced capabilities in Computer Vision, Deep Learning inference optimization, and Full-Stack AI dashboard development. It is built to mimic the perception backend of autonomous driving systems.

---

## 📸 Project Showcase
*(💡 Tip for GitHub/LinkedIn: Replace the text below with a GIF or Screenshot of your dashboard running)*
`![Dashboard Screenshot](app/assets/screenshot.png)`
`![Radar GIF](app/assets/demo.gif)`

---

## 🌟 Overview
The **Autonomous Vehicle Perception Module** is a production-grade Computer Vision pipeline built for autonomous driving and traffic analytics. Powered by state-of-the-art deep learning architectures (**YOLOv8** & **DeepSORT**), this system processes raw dashcam or drone footage to extract rich, actionable intelligence in real-time.

Designed with a premium **SaaS-style Glassmorphism UI**, it acts as a central command hub for traffic analysis, offering features typically found in proprietary enterprise systems.

---

## 🔥 Key Technical Features

- **🎯 Real-Time Multi-Object Tracking:** Flawlessly detects and tracks cars, trucks, buses, pedestrians, and critical road infrastructure using `deep-sort-realtime` and MobileNet Re-ID.
- **📏 Monocular Distance Estimation (3D Depth):** Geometrically approximates the distance (in meters) to moving vehicles using a single camera lens—eliminating the need for LiDAR.
- **🚨 Hyper-Sensitive Proximity Alerts:** Computes bounding-box mass to trigger massive UI/video warnings when a vehicle breaches safety distance thresholds (customizable down to 5% screen area).
- **🕵️ GDPR Privacy Mode (Anonymization):** Instantly applies military-grade Gaussian blurring to license plates, vehicles, and pedestrians to comply with international privacy laws.
- **🌙 Night Vision Enhancement:** Uses dynamic Gamma Correction algorithms to artificially illuminate dark, foggy, or low-visibility footage via OpenCV.
- **🔗 Direct YouTube Integration:** Built-in `yt-dlp` integration allows users to paste any YouTube link directly into the UI for instant download and analysis.
- **🤖 Automated AI Reporting:** Generates a natural language textual summary analyzing traffic density, peak congestion periods, and safety metrics based on raw Pandas dataframes.
- **📥 Processed Video Export:** Downloads the fully rendered, annotated video (with bounding boxes, trails, and speed labels) directly from the browser.

---

## 🛠️ Technology Stack

| Domain | Technology |
| :--- | :--- |
| **Object Detection** | Ultralytics YOLOv8 (Custom Trained Weights + Pre-trained) |
| **Object Tracking** | DeepSORT (`deep-sort-realtime`) |
| **Computer Vision** | OpenCV (cv2) |
| **Data & Analytics** | Pandas, Plotly Express |
| **Frontend UI** | Streamlit (Custom CSS Glassmorphism) |
| **Video Ingestion** | `yt-dlp` |

---

## 📂 Project Structure

```text
Autonomous Vehicle Perception Module/
├── app/
│   └── dashboard.py           # Streamlit Web Application & UI
├── data/
│   └── demo/                  # Auto-downloaded YouTube footage
├── src/
│   ├── analytics.py           # Pandas dataframe building and AI text generation
│   ├── config.py              # Centralized hyperparameters & thresholds
│   ├── download_youtube.py    # yt-dlp integration script
│   ├── tracker.py             # DeepSORT class and Speed/Distance Logic
│   └── yolo_detector.py       # YOLOv8 Inference engine
└── README.md                  # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Clone & Environment Setup
Ensure you have a Python environment with CUDA support for maximum FPS.

```bash
# Clone the repository
git clone <your-repo-url>
cd "Autonomous Vehicle Perception Module"

# Install dependencies 
pip install -r requirements.txt
```

### 2. Launch the Dashboard
Start the Streamlit application:
```bash
streamlit run app/dashboard.py
```

### 3. Usage Instructions
1. **Upload or Paste:** Drag and drop an `.mp4` file OR simply paste a YouTube URL into the provided text box.
2. **Configure (Sidebar):** Adjust `Inference Size`, turn on `Privacy Mode`, or enable `Estimate Distance` depth perception.
3. **Analyze:** Click **🚀 Run Analysis** to watch the live tracking, proximity alerts, and speed estimations.
4. **Export & Reports:** Head over to the **📊 Analytics Dashboard** tab to read the AI Summary, view the Density Heatmap, and download the annotated video.

---

## 👨‍💻 Author & Contact

**Eslam**  
AI & Computer Vision Engineer  
- 💼 LinkedIn: [Your LinkedIn Profile URL]
- 🐙 GitHub: [Your GitHub Profile URL]
- 📧 Email: [Your Email Address]

*Feel free to reach out for collaborations, discussions on Autonomous Systems, or Computer Vision opportunities!*
