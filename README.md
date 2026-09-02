# Real-Time Hand Gesture Recognition for Interactive Systems (HGR V1)

A Python-based classical computer vision application that detects and recognizes hand gestures in real-time using user-adaptive HSV skin-color segmentation, contour scoring, and geometric feature analysis with a **0–100% heuristic confidence score**. No deep learning required—fully explainable and suitable for college-level coursework and B.Tech viva demonstrations.

---

## 🎯 Project Overview

This application captures video from a webcam, isolates the user's hand using user-adaptive HSV color-space thresholding (supporting normal and circular hue wrap-around ranges), and recognizes four distinct hand gestures in real time. Each gesture is mapped to an interactive action (`START`, `STOP`, `SELECT`, `NEXT`), accompanied by an explainable 0–100% heuristic confidence score and temporal smoothing.

### Key Features
- ✅ **User-adaptive HSV calibration** with relative path resolution (`config/hsv_calibration.json`).
- ✅ **Recalibration safety**: Failed recalibration preserves valid previous calibration; invalid initial calibration exits cleanly.
- ✅ **Normal & circular hue wrap-around** range handling (e.g. $[162, 15, 40] \to [16, 170, 255]$ for $162^\circ \text{--} 180^\circ$ or $0^\circ \text{--} 16^\circ$).
- ✅ **Skin detection** using configurable HSV thresholds with morphological cleanup (`open`, `close`).
- ✅ **Hand contour filtering & min-score rejection**: Rejects background blobs, strips, and non-hand shapes.
- ✅ **4-gesture recognition** using measurable geometric features (solidity, extent, aspect ratio, convexity defects).
- ✅ **0–100% Heuristic Detection Confidence**: Calculated from contour quality score, geometric plausibility, gesture feature margins, and temporal history.
- ✅ **Temporal smoothing** with consensus voting to eliminate flicker.
- ✅ **"No Hand Detected" state** with low confidence reporting when no hand is present.
- ✅ **Real-time FPS & diagnostic overlay** in `main_debug.py`.
- ✅ **Type-safe result structures** (`results.py`).
- ✅ **Pure HSV V1 focus**: Purged unused Gaussian/Mahalanobis statistical skin models.
- ✅ **Compact, reproducible test suite**: 67 automated tests passing in virtual environment.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.7+ |
| Computer Vision | OpenCV (`opencv-python-headless` / `opencv-python`) |
| Numerical Computing | NumPy |
| Testing | pytest |
| Color Space | HSV (Hue: 0–180, Saturation: 0–255, Value: 0–255) |
| Feature Logic | Classical CV (Convex hull, convexity defects, solidity, extent) |

### ⚠️ Scope Locks (No Heavy ML)
- ❌ No MediaPipe, TensorFlow, or PyTorch
- ❌ No deep learning or pretrained models
- ❌ No statistical skin models (purged from V1)
- ❌ No cloud APIs or network calls

---

## 📊 Pipeline Architecture

```text
Webcam
  ↓
CameraCapture (Resize factor validation & resource cleanup)
  ↓
User HSV Calibration (Normal & circular wrap-around validation)
  ↓
HSV Skin Segmentation (Dual-mask bitwise OR for hue wrap-around)
  ↓
Morphological Cleaning (Opening & Closing)
  ↓
Contour Detection
  ↓
Hand Contour Selection (Area, strip, background & min-score filter)
  ↓
Gesture Recognition (Geometric feature extraction & classification)
  ↓
Gesture History (Temporal consensus voting)
  ↓
Gesture Label + 0–100% Confidence Score
```

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.7+
- USB webcam (or mock synthetic frames for automated tests)

### Setup Virtual Environment

```bash
# Navigate to repository root
cd /home/Prajesh/hgr

# Activate project virtual environment
source venv/bin/activate

# Install reproducible dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Main Gesture Application
```bash
./venv/bin/python src/main.py
```

### Real-Time Diagnostic Mode
```bash
./venv/bin/python src/main_debug.py
```

### Keyboard Controls
- `SPACE`: Capture sample during calibration mode
- `ESC`: Cancel calibration
- `c`: Trigger interactive recalibration during runtime
- `q`: Quit application

---

## 🖐️ Supported Gestures & Action Mapping

| Gesture | Action | Features / Classification Rules |
|---------|--------|----------------------------------|
| **Fist** | `STOP` | High solidity ($\ge 0.88$), high extent ($\ge 0.60$), 0 convexity defects |
| **Open Palm** | `START` | Deep convexity defects ($\ge 3$), lower solidity ($\le 0.88$) |
| **One Finger** | `SELECT` | Elongated contour ($\text{elongation} \ge 2.0$), $\le 1$ defect |
| **Two Fingers** | `NEXT` | Moderately elongated ($\text{elongation} \ge 1.25$), $1 \text{--} 2$ defects |

Example output:
```text
Gesture: Open Palm (91%)
Action: START
FPS: 24.5
```

If no candidate meets the minimum contour score or shape criteria:
```text
No Hand Detected | Confidence: 28%
```

---

## 📂 Project Structure

```text
/home/Prajesh/hgr/
├── src/
│   ├── main.py                 # Main entry point & event loop
│   ├── main_debug.py           # Real-time diagnostic overlay runner
│   ├── camera.py               # CameraCapture (resize validation & resource release)
│   ├── skin_detection.py       # SkinDetector (HSV segmentation & wrap-around handling)
│   ├── hand_detection.py       # HandDetector (contour analysis & min-score filtering)
│   ├── gesture_recognition.py  # GestureRecognizer & heuristic confidence calculation
│   ├── gesture_history.py      # GestureHistory (temporal smoothing)
│   ├── calibration.py          # CalibrationManager (project-relative persistence)
│   └── results.py              # Type-safe result dataclasses
├── tests/
│   ├── test_calibration.py
│   ├── test_skin_detection.py
│   ├── test_hand_detection.py
│   ├── test_camera.py
│   ├── test_gesture_recognition.py
│   ├── test_gesture_history.py
│   └── test_integration.py
├── config/
│   └── hsv_calibration.json    # Calibrated HSV thresholds (relative path, git-ignored)
├── requirements.txt            # Reproducible dependencies
├── README.md                   # Project documentation
├── AGENTS.md                   # Guidance for AI coding assistants
├── SKILLS.md                   # Computer vision skills cheatsheet
└── ARCHITECTURE.md             # Technical architecture & data models
```

---

## 🧪 Testing

Run the complete test suite inside the project virtual environment:

```bash
./venv/bin/pytest -v
```

All 67 tests pass with 0 failures, 0 errors, and 0 warnings:
```text
67 passed in 1.18s
```

Path independence test:
```bash
cd /tmp && /home/Prajesh/hgr/venv/bin/pytest /home/Prajesh/hgr/tests/ -v
```
