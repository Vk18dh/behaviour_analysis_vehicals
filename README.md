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

### 1. Installation
```bash
git clone https://github.com/Vk18dh/behaviour_analysis_vehicals.git
cd behaviour_analysis_vehicals
pip install -r requirements.txt
```

### 2. Configuration
The system is entirely **config-driven**. Adjust thresholds, fines, and camera parameters in `config/settings.yaml`.

```bash
copy .env.example .env  # Update secrets like JWT_SECRET
```

### 3. Execution
Run all services (API, Dashboard, and Pipeline) simultaneously:
```bash
python main.py all --webcam 0
```
- **API:** `http://localhost:8000/docs`
- **Dashboard:** `http://localhost:8501` (Default login: `admin` / `admin123`)

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