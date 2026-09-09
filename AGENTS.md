# AGENTS.md — Guidance for AI Coding Agents

This document defines guidelines, design constraints, and rules for AI coding assistants working on the **Real-Time Hand Gesture Recognition (HGR Strategy 1)** repository.

---

## 🎯 Repository Purpose & Core Constraints

1. **MediaPipe 3D Landmark Analysis + Rule Geometry (Strategy 1)**:
   - The project is an explainable, real-time hand gesture recognition system.
   - Core technology: MediaPipe 3D Hand Landmark Tracking (extracting 21 spatial keypoints) paired with deterministic, orientation-aware geometric rules (Euclidean joint distance ratios, vector dot product straightness, and normalized thumb spread).

2. **Strict Scope Locks (DO NOT ADD)**:
   - ❌ **No custom neural networks, TensorFlow, PyTorch, or Scikit-Learn classifiers**.
   - ❌ **No black-box gesture classification models** (e.g., do NOT use `MediaPipe Tasks GestureRecognizer API`).
   - ❌ **No databases, cloud APIs, Ollama, or remote services**.
   - ❌ **No statistical skin models** (classical HSV code is isolated in `src/legacy_hsv/` for comparative viva demos).

3. **Strict Decoupling Lock**:
   - `MediaPipeDetector` (`src/mediapipe_detector.py`) extracts landmarks and renders skeleton overlays; it must **NEVER** contain gesture classification logic.
   - `LandmarkGestureRecognizer` (`src/gesture_recognition.py`) must **NOT** import or depend directly on MediaPipe APIs or classes (operates purely on plain Python coordinate dictionaries).

4. **Orientation-Aware Gesture Mathematics**:
   - Finger extension must use relative Euclidean distance ratios:
     $$\frac{\text{dist}(\text{Wrist}, \text{Tip})}{\text{dist}(\text{Wrist}, \text{PIP})} > 1.35$$
   - Straightness must use vector dot products:
     $$\cos\theta = \frac{\vec{v}_1 \cdot \vec{v}_2}{\|\vec{v}_1\| \|\vec{v}_2\|} > 0.85$$
   - Thumb spread must use normalized distance ratio:
     $$\text{thumb\_ratio} = \frac{\text{dist}(\text{Thumb\_Tip}, \text{Pinky\_MCP})}{\text{dist}(\text{Wrist}, \text{Pinky\_MCP})} > 0.85$$

5. **Explainable 0–100% Heuristic Confidence Score**:
   - Clamped to $[0, 100]$:
     $$\text{Confidence} = \text{clamp}\Big(\text{Quality (0–30)} + \text{Rule Match (0–50)} + \text{Stability (0–20)}, \, 0, \, 100\Big)$$
   - Quality (0–30): Structural plausibility (coordinate variance and palm proportion bounds).
   - Rule Match Margin (0–50): Distance and straightness decision margins.
   - Temporal Stability (0–20): Fraction of agreement within the sliding history buffer.
   - Do NOT describe or implement confidence as a machine learning probability.

6. **Temporal Consensus Rules**:
   - `Unknown` and `No Hand Detected` (or `None`) frames must NOT compete as valid gesture candidates in consensus voting.
   - Compute final confidence score AFTER adding current frame's raw gesture to the history buffer.

7. **Test Environment & Execution**:
   - Always run unit and integration tests using the virtual environment:
     ```bash
     ./venv/bin/pytest -v
     ```
   - Maintain 100% passing test status (0 failures, 0 errors, 0 warnings).
   - Automated tests must run completely offline without depending on a physical webcam or network access.

8. **Public Input Validation**:
   - Public methods in `MediaPipeDetector` and `CameraCapture` must explicitly validate input arguments (raising clear `TypeError` or `ValueError` exceptions for `None`, empty arrays, or invalid shapes) before processing.
