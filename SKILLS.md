# SKILLS.md — Technical Skills Reference (Strategy 1)

This document outlines the computer vision, spatial mathematics, and software engineering skills implemented in the Real-Time Hand Gesture Recognition (HGR Strategy 1) project.

---

## 🛠️ Core Technical Skills

### 1. 3D Spatial Keypoint Tracking
- **MediaPipe Hands Pipeline**: Extracting 21 anatomical landmarks $(x, y, z)$ normalized to image coordinates with relative depth estimates.
- **Decoupled Architecture**: Encapsulating keypoint extraction and skeleton visualization (`MediaPipeDetector`) strictly separate from classification logic.
- **Skeleton Rendering**: Visualizing hand topology via anatomical connections using OpenCV and drawing utilities.

### 2. Computational Geometry & Vector Mathematics
- **3D Euclidean Distance Ratios**: Computing scale-invariant radial extension ratios:
  $$\frac{\text{dist}(\text{Wrist}, \text{Tip})}{\text{dist}(\text{Wrist}, \text{PIP})}$$
- **Vector Dot Product Straightness**: Evaluating joint collinearity and straightness via vector dot product angles:
  $$\cos\theta = \frac{\vec{v}_1 \cdot \vec{v}_2}{\|\vec{v}_1\| \|\vec{v}_2\|}$$
- **Normalized Thumb Spread Geometry**: Measuring thumb abduction/adduction relative to lateral palm anchors (`Wrist` and `Pinky_MCP`).
- **Rigid-Body Invariance**: Mathematical formulation ensuring gesture classifications remain strictly invariant to 2D/3D rotation, uniform scaling, and spatial translation.

### 3. Heuristic Confidence Evaluation & Decision Logic
- **Structural Plausibility Checks**: Evaluating landmark coordinate variance and anatomical proportions to detect degenerate or collapsed predictions.
- **Decision Boundary Margin Evaluation**: Quantifying how decisively finger ratios exceed or fall below classification thresholds.
- **Temporal Stability Weighting**: Integrating history buffer consensus agreement into real-time confidence scores.

### 4. Temporal Smoothing & State Machines
- **Sliding-Window Consensus Voting**: Requiring consecutive agreement across a rolling history buffer (`GestureHistory`) to eliminate frame-to-frame label flicker.
- **Ambiguous State Rejection**: Preventing `Unknown` or empty (`None`) frames from competing as candidate votes during consensus formation.

### 5. Defensive Software Engineering & Testing
- **Type-Safe Result Structures**: Passing explicit, typed dataclasses (`results.py`) between processing stages.
- **Strict Decoupling & Dependency Isolation**: Ensuring `LandmarkGestureRecognizer` operates on pure Python dictionaries with zero imports from MediaPipe.
- **Synthetic Test Generation**: Constructing canonical 3D landmark arrays for automated testing.
- **Transformation Invariance Verification**: Programmatically applying 2D and 3D rotation matrices, translations, and scaling to verify gesture invariance.
- **Offline Pytest Execution**: Achieving 100% test coverage without hardware camera or external network dependencies.
