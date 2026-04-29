# 🚦 Smart Traffic Enforcement & Vehicle Behavior Analysis System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green.svg)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-red.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-orange.svg)](https://streamlit.io/)

A state-of-the-art, modular, and config-driven traffic enforcement pipeline designed for real-time and batch video processing. This system integrates advanced computer vision, deep learning, and automated rule enforcement to monitor and improve road safety.

---

## ✨ Core Features

### 🛠️ Advanced Computer Vision Pipeline
- **Digital Image Processing (DIP):** Automated white balance, defogging, CLAHE, gamma correction, and video stabilization.
- **Low-Light Enhancement (LLIE):** Hybrid classical (CLAHE+Gamma) and deep-learning-ready enhancement for night-time monitoring.
- **Precision Detection & Tracking:** Powered by **YOLOv8** for multi-class vehicle detection and **ByteTrack** for robust temporal tracking.

### 🧠 Intelligent Behavior Detection (13+ Types)
Detects a wide range of violations based on trajectory analysis and visual cues:
- 🏎️ **Speeding:** Dynamic fine calculation based on MV Act §183.
- 〰️ **Zigzag/Rash Driving:** Trajectory-based weaving detection (§184).
- 📏 **Tailgating:** Time-headway sustained duration monitoring.
- ⛔ **Wrong Way & Lane Violations:** Zone-based and lane-fill analysis.
- 🛣️ **Highway Restrictions:** Detection of restricted vehicles (2W/3W) in high-speed zones.
- 👷 **Safety Gear:** Helmet and Seatbelt classification swap points.
- 📱 **Distracted Driving:** Phone use and triple riding detection.

### 📋 Enforcement & Governance
- **ANPR (Automatic Number Plate Recognition):** 3-tier confidence gate using YOLO plate detection and EasyOCR.
- **Rule Engine:** Automated fine issuance and a **Credit Score System** (80-100: Safe, 50-79: Moderate, 0-49: Risky).
- **Evidence Management:** Automatic generation of annotated JPEG images and ±2s MP4 video clips for every violation.
- **Blockchain Audit Log:** SHA-256 chain for immutable violation records.

### 🖥️ Modern Management Interface
- **FastAPI Backend:** Secure REST API with JWT authentication and RBAC (Role-Based Access Control).
- **Streamlit Dashboard:** Real-time system monitoring, human-in-the-loop violation review, and rich data visualization with Plotly.

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+** installed ([download](https://www.python.org/downloads/))
- A **webcam** (for real-time mode) or an **MP4/AVI video file** (for batch mode)
- **~2 GB disk space** (for YOLO model weights and dependencies)

### 1. Clone & Install
```bash
git clone https://github.com/Vk18dh/behaviour_analysis_vehicals.git
cd behaviour_analysis_vehicals

# (Recommended) Create a virtual environment
python -m venv .venv

# Activate it
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows CMD:
.\.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

> **Note:** On first run, YOLOv8 will automatically download the `yolov8n.pt` model weights (~6.5 MB). An internet connection is required for this one-time download.

### 2. Configuration
The system is entirely **config-driven**. All thresholds, fines, camera settings, and detection parameters live in `config/settings.yaml`.

```bash
# (Optional) Copy the environment template and set your secrets
copy .env.example .env
```

Key settings you may want to adjust in `config/settings.yaml`:
| Setting | Location | Default | Purpose |
|---------|----------|---------|---------|
| `camera.process_every_n_frames` | Line 29 | `3` | Frame skip for real-time (higher = faster, less accurate) |
| `camera.batch_process_every_n_frames` | Line 33 | `5` | Frame skip for uploaded videos |
| `behavior.wrong_direction.road_dir` | Line 177 | `[0.0, -1.0]` | Traffic direction vector (`[0,-1]`=up, `[0,1]`=down, `[1,0]`=right) |
| `preprocessing.defog` | Line 65 | `false` | Enable Dark Channel Prior defogging (heavy, only for hazy conditions) |

### 3. How to Run

#### Option A: Run Everything Together (Recommended)
Starts the **API server**, **Streamlit dashboard**, and **live webcam pipeline** simultaneously:
```bash
python main.py all --webcam 0
```
This launches three services:

| Service | URL | Description |
|---------|-----|-------------|
| **FastAPI Backend** | http://localhost:8000/docs | REST API with Swagger docs |
| **Streamlit Dashboard** | http://localhost:8501 | Human review UI |
| **Live Pipeline** | *(runs in terminal)* | Real-time detection from webcam |

#### Option B: Run Individual Services
```bash
# Real-time pipeline only (webcam)
python main.py live --webcam 0

# Real-time pipeline only (RTSP camera)
python main.py live --camera_id cam_01 --rtsp rtsp://192.168.1.100/stream

# Batch process a video file
python main.py batch --video path/to/video.mp4

# API server only
python main.py api --host 0.0.0.0 --port 8000

# Dashboard only
python main.py dash

# Clear all violations from database
python main.py clear          # violations + logs only
python main.py clear --full   # also removes vehicles and credit scores
```

### 4. Login Credentials
| Username | Password | Role |
|----------|----------|------|
| `admin` | `Admin@PBL2026` | Full admin access |

### 5. Uploading a Video for Analysis
1. Open the dashboard at **http://localhost:8501**
2. Log in with the credentials above
3. Navigate to the **"Upload Video"** section
4. Select an MP4 or AVI file and click **Upload**
5. The system will process the video in the background and violations will appear in the **"Human Review Queue"**

### 6. Stopping the System
Press **`Ctrl + C`** in the terminal where `main.py` is running. All services will shut down gracefully.

---

## 📂 Project Structure
```text
behaviorpbl/
├── config/             # YAML settings for thresholds, fines, and cameras
├── src/
│   ├── pipeline/       # Real-time & Batch processing logic
│   ├── behavior/       # Core behavior detection algorithms
│   ├── api/            # FastAPI backend endpoints
│   ├── dashboard/      # Streamlit UI implementation
│   ├── anpr/           # License plate detection & OCR
│   ├── database/       # SQLAlchemy models & encryption
│   └── ...             # Modular vision & utility components
├── tests/              # Comprehensive Pytest suite
└── main.py             # Unified entry point
```

---

## 🛠️ Recent Hardening & Optimization
We recently implemented several critical updates to ensure production-grade reliability:
- **Aspect-Ratio Invariance:** Seamlessly supports both **Landscape** and **Portrait** (vertical) video feeds through dynamic homography scaling.
- **Timing Synchronization:** All behavior detectors now use **Video Timestamps** instead of system wall-clock time, eliminating "random" false positives during high CPU load.
- **Administrative Control:** Added a "Danger Zone" in the dashboard for irreversible database purging and violation clearing.
- **Security Hardening:** Updated default encryption keys and administrative passwords to resolve critical startup warnings.
- **Refined Detectors:** Enhanced Wrong Direction and Lane Violation logic for better accuracy on complex road layouts.

---

## ⚙️ Performance & Optimization
- **Frame Skipping:** Configurable `process_every_n_frames` (in `settings.yaml`) to optimize throughput for high-resolution streams.
- **Dynamic Toggles:** Enable/Disable specific behavior detectors directly from the configuration file.
- **Longest-Edge Resizing:** Optimized batch processing to resize based on the longest dimension, preventing memory overflows for vertical videos.
- **Async I/O:** Multi-threaded frame buffering to prevent stale frames and minimize latency.

---


## 📜 License & Usage
This project is intended for educational and research purposes. All MV Act fine values are aligned with the official Indian Motor Vehicles Act 2019 schedules.

---
Developed with ❤️ for Smart City Traffic Management.