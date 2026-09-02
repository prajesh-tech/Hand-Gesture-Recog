# AGENTS.md — Guidance for AI Coding Agents

This document defines guidelines, design constraints, and rules for AI coding assistants working on the **Real-Time Hand Gesture Recognition (HGR)** repository.

---

## 🎯 Repository Purpose & Core Constraints

1. **Classical Computer Vision V1**:
   - The project is a lightweight, explainable, classical computer vision system.
   - Core technology: user-adaptive HSV color segmentation, morphological cleaning, contour analysis, geometric feature extraction, and temporal history.

2. **Strict Scope Locks (DO NOT ADD)**:
   - ❌ **No MediaPipe, TensorFlow, PyTorch, or deep learning models**.
   - ❌ **No pretrained weights or neural networks**.
   - ❌ **No cloud APIs, Ollama, or remote services**.
   - ❌ **No statistical skin models** (Gaussian/Mahalanobis code was intentionally purged from V1).

3. **Gesture Threshold Lock**:
   - Do **NOT** modify existing gesture classification thresholds in `src/gesture_recognition.py` (`THRESHOLDS` dictionary) unless explicitly instructed by the user.

4. **Heuristic Confidence Score**:
   - Confidence is computed from physical/geometric evidence (contour score, shape metrics, feature margins, temporal history) and clamped to `0–100%`.
   - Do NOT describe or implement confidence as a machine learning probability.

5. **Calibration Safety**:
   - A failed or cancelled recalibration must **NEVER** corrupt or overwrite a valid existing calibration.
   - If recalibration fails and a valid calibration exists, retain the previous calibration.
   - HSV ranges must support both normal (`lower_h <= upper_h`) and circular hue wrap-around (`lower_h > upper_h`, e.g., $[162, 15, 40] \to [16, 170, 255]$).

6. **Test Environment & Execution**:
   - Always run unit and integration tests using the virtual environment:
     ```bash
     ./venv/bin/pytest -v
     ```
   - Maintain 100% passing test status (0 failures, 0 errors, 0 warnings).
   - Automated tests must run completely offline without depending on a physical webcam or network access.

7. **Relative File Paths**:
   - All configuration files (`config/hsv_calibration.json`) must be resolved dynamically relative to the project root directory, not the current working directory (`CWD`).

8. **Public Input Validation**:
   - Public processing methods in `SkinDetector`, `HandDetector`, and `CameraCapture` must explicitly validate input arguments (raising clear `TypeError` or `ValueError` exceptions for `None`, empty arrays, incorrect dimensions, or invalid numbers) before passing them to OpenCV.
