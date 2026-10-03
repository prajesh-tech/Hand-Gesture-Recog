# Real-Time Hand Gesture Recognition for Interactive Systems (HGR Strategy 1)

A real-time computer vision application that detects and recognizes hand gestures using **MediaPipe 3D Hand Landmark Tracking** paired with **explainable, rule-based geometric logic** (Euclidean joint distance ratios, straightness dot products, and normalized thumb spread) with a **0–100% deterministic heuristic confidence score**. 

No black-box classifiers or custom neural networks—100% transparent, mathematically provable, and designed for rigorous B.Tech / college-level viva defense.

---

## 🎯 Project Overview

This application captures video from a webcam, extracts 21 spatial hand keypoints $(x, y, z)$ via a decoupled detector, and analyzes relative finger geometry to recognize four distinct hand gestures in real time. Each gesture maps directly to an interactive action (`START`, `STOP`, `SELECT`, `NEXT`), backed by temporal consensus voting to eliminate flicker.

### Key Features
- ✅ **MediaPipe 3D Landmark Tracking**: Extracts 21 anatomical keypoints in real time.
- ✅ **Strict Architectural Decoupling**: Detector handles landmark extraction and skeleton overlay; `LandmarkGestureRecognizer` operates purely on plain Python coordinate dictionaries with zero MediaPipe dependencies.
- ✅ **Orientation-Aware Geometry**: Anatomically adaptive finger extension via 3D Euclidean distance ratios and vector dot product straightness ($\cos\theta > 0.85$).
- ✅ **Spatial Invariance**: Mathematics is strictly invariant under 2D/3D rotations, translation, and uniform scaling.
- ✅ **4-Gesture Recognition**:
  - **Fist** (`STOP`): All 4 fingers closed & thumb tucked.
  - **Open Palm** (`START`): All 4 fingers extended & thumb extended.
  - **One Finger** (`SELECT`): Only Index extended; Middle, Ring, Pinky closed.
  - **Two Fingers** (`NEXT`): Index & Middle extended; Ring, Pinky closed.
- ✅ **0–100% Deterministic Heuristic Confidence**: Computed from Landmark Quality (0–30), Rule Match Margin (0–50), and Temporal Stability (0–20).
- ✅ **Temporal Smoothing Engine**: Sliding window consensus voting in `GestureHistory`.
- ✅ **Diagnostic HUD & Telemetry**: Real-time FPS, finger ratios, angles, and confidence sub-scores in `main_debug.py`.
- ✅ **Type-Safe Result Contracts**: Dataclasses defined in `results.py`.
- ✅ **Legacy HSV Preservation**: Classical HSV skin-color segmentation code preserved in `src/legacy_hsv/` for side-by-side comparative viva evaluations.
- ✅ **100% Offline Pytest Suite**: 71 automated unit and integration tests passing completely offline without hardware webcam dependencies.

---

## 🛠️ Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| Language | Python 3.8+ / 3.13 | Core runtime environment |
| Keypoint Extraction | MediaPipe (`mediapipe` >= 0.10.0) | 21 3D spatial hand landmarks $(x, y, z)$ |
| Image Processing | OpenCV (`opencv-python` >= 4.8.0) | Camera stream acquisition, BGR/RGB conversion, overlays |
| Numerical Computing | NumPy | Vector mathematics and transformation matrices |
| Test Framework | pytest | Offline invariant unit and integration testing |
| Feature Logic | Custom Spatial Geometry | Euclidean distance ratios, vector dot product straightness |

### 🚫 Scope Locks (Viva & Academic Defense)
- ❌ No custom neural networks, TensorFlow, PyTorch, or Scikit-Learn classifiers.
- ❌ No black-box gesture classification APIs (e.g. MediaPipe Tasks GestureRecognizer API is prohibited).
- ❌ No databases, cloud APIs, Ollama, or external network requests.
- ❌ No statistical skin models.

The technical story is strictly: **MediaPipe extracts 21 3D hand keypoints $\longrightarrow$ custom geometric mathematics determines finger states $\longrightarrow$ explicit rules classify four gestures $\longrightarrow$ temporal consensus stabilizes the result $\longrightarrow$ confidence and actions are displayed.**

---

## 📊 Pipeline Architecture

The processing pipeline strictly executes in this deterministic sequential order:

```text
Webcam / Video Device
  ↓
CameraCapture (Resize factor validation & resource safety)
  ↓
MediaPipeDetector (Extracts 21 (x, y, z) landmarks & renders skeleton)
  ↓
LandmarkGestureRecognizer (Euclidean distance ratios & vector straightness)
  ↓
Raw Gesture Classification (Fist, Open Palm, One Finger, Two Fingers, Unknown)
  ↓
GestureHistory (Sliding window temporal consensus voting)
  ↓
Stable Gesture & Interactive Action (START, STOP, SELECT, NEXT)
  ↓
0–100% Deterministic Confidence HUD (Quality + Rule Match + Temporal Stability)
```

---

## 🖐️ Supported Gestures & Mathematical Rules

Finger extension is evaluated independently of hand orientation using relative spatial geometry:

### 1. 4-Finger Extension Logic (Index, Middle, Ring, Pinky)
A finger is **extended** if BOTH conditions are met:
1. **Euclidean Extension Ratio**:
   $$\frac{\text{dist}(\text{Wrist}, \text{Tip})}{\text{dist}(\text{Wrist}, \text{PIP})} > 1.35$$
2. **Alignment / Straightness**:
   $$\cos\theta = \frac{\vec{v}_1 \cdot \vec{v}_2}{\|\vec{v}_1\| \|\vec{v}_2\|} > 0.85 \quad \text{where } \vec{v}_1 = \text{PIP} - \text{MCP}, \;\; \vec{v}_2 = \text{Tip} - \text{PIP}$$

### 2. Normalized Thumb Spread Logic
$$\text{thumb\_ratio} = \frac{\text{dist}(\text{Thumb\_Tip}, \text{Pinky\_MCP})}{\text{dist}(\text{Wrist}, \text{Pinky\_MCP})} > 0.85$$

### Rule Matrix

| Gesture | Action | Mathematical Rule |
|---------|--------|-------------------|
| **Fist** | `STOP` | All 4 fingers closed ($\text{ratio} \le 1.35$) & thumb tucked ($\text{thumb\_ratio} \le 0.85$) |
| **Open Palm** | `START` | All 4 fingers extended ($\text{ratio} > 1.35 \land \cos\theta > 0.85$) & thumb extended ($> 0.85$) |
| **One Finger** | `SELECT` | Index extended; Middle, Ring, Pinky closed |
| **Two Fingers** | `NEXT` | Index & Middle extended; Ring, Pinky closed |
| **Unknown** | None | Hand detected, but joint layout does not match target rules |
| **No Hand** | None | Landmarks is `None` (no hand detected in frame) |

---

## 📈 Deterministic Heuristic Confidence Score (0–100%)

The confidence score is computed from physical and temporal evidence, clamped to $[0, 100]$:

$$\text{Confidence} = \text{clamp}\Big(S_{\text{quality}} + S_{\text{rule\_match}} + S_{\text{stability}}, \, 0, \, 100\Big)$$

- **Landmark Quality ($0 \le S_{\text{quality}} \le 30$)**: Structural plausibility of keypoints (adequate coordinate variance and physiological palm-to-hand proportions).
- **Rule Match Margin ($0 \le S_{\text{rule\_match}} \le 50$)**: Measures how decisively joint ratios and straightness angles exceed or fall below decision boundaries.
- **Temporal Stability ($0 \le S_{\text{stability}} \le 20$)**: Proportion of matching predictions within the sliding history buffer.

---

## 🎓 Viva / College Discussion: Keypoints vs Classical HSV

| Dimension | Classical HSV Segmentation (Legacy) | MediaPipe 3D Landmark Tracking (Strategy 1) |
|-----------|-------------------------------------|---------------------------------------------|
| **Skin-Tone Invariance** | Fragile: requires calibration per individual melanin level | **Immune**: landmark model is trained across diverse demographic datasets |
| **Illumination Sensitivity** | High: shadows, uneven lighting, and yellow/blue bulbs break thresholds | **Robust**: spatial geometry remains intact under varying ambient lighting |
| **Orientation Invariance** | Limited: relies on bounding box aspect ratio and vertical orientation | **Complete**: Euclidean distance ratios and vector dot products are invariant under 2D/3D rotation |
| **Self-Occlusion** | Poor: hand overlaps create merged blobs | **Moderate**: handles partial occlusion, but side-view fingers can jitter |
| **Computational Footprint** | Extremely low (simple color thresholding and binary morphology) | Moderate: requires keypoint inference (optimized for real-time CPU) |

> **Comparative Demonstration**: Classical HSV modules are isolated in `src/legacy_hsv/` and can be demonstrated side-by-side during viva defense to prove why modern keypoint analysis is superior for practical interactive systems.

---

## 📦 Installation & Setup

```bash
# Navigate to repository
cd /home/Prajesh/hgr

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Main Gesture Application
```bash
./venv/bin/python src/main.py
```

### Real-Time Diagnostic Mode (HUD Telemetry)
```bash
./venv/bin/python src/main_debug.py
```

### Controls
- `q`: Quit application
- `Ctrl+C`: Terminal interrupt

---

## 📂 Project Structure

```text
/home/Prajesh/hgr/
├── src/
│   ├── main.py                 # Main entry point & event loop
│   ├── main_debug.py           # Real-time diagnostic telemetry overlay
│   ├── mediapipe_detector.py   # Decoupled MediaPipe 3D Landmark detector
│   ├── gesture_recognition.py  # LandmarkGestureRecognizer & vector geometry
│   ├── gesture_history.py      # GestureHistory (temporal consensus voting)
│   ├── camera.py               # CameraCapture (resize validation & resource safety)
│   ├── results.py              # Type-safe dataclasses (Landmark, Gesture, History)
│   └── legacy_hsv/             # Preserved classical HSV modules for viva comparison
│       ├── skin_detection.py
│       ├── hand_detection.py
│       └── calibration.py
├── tests/
│   ├── test_mediapipe_detector.py   # Detector unit tests & input validation
│   ├── test_gesture_recognition.py  # Spatial invariance, rotation & boundary tests
│   ├── test_integration.py          # End-to-end pipeline & consensus tests
│   ├── test_gesture_history.py      # Temporal consensus engine tests
│   ├── test_camera.py               # Camera acquisition & cleanup tests
│   ├── test_skin_detection.py       # Legacy HSV tests
│   ├── test_hand_detection.py       # Legacy contour tests
│   └── test_calibration.py          # Legacy calibration tests
├── requirements.txt            # Project dependencies (mediapipe, opencv, numpy, pytest)
├── README.md                   # Project documentation & viva guide
├── ARCHITECTURE.md             # Technical architecture & mathematical specifications
├── AGENTS.md                   # Coding agent constraints and rules
└── SKILLS.md                   # Technical skills cheatsheet
```

---

## 🧪 Testing

Run the complete test suite inside the project virtual environment:

```bash
./venv/bin/pytest -v
```

All 71 tests pass with 0 failures, 0 errors, and 0 warnings completely offline without hardware webcam dependencies:
```text
============================== 71 passed in 2.12s ==============================
```
