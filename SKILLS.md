# SKILLS.md — Technical Skills Reference

This document outlines the computer vision, software engineering, and mathematical skills implemented in the Real-Time Hand Gesture Recognition (HGR) project.

---

## 🛠️ Core Technical Skills

### 1. Classical Computer Vision & Image Processing
- **HSV Color-Space Segmentation**: Converting BGR video frames to HSV ($H \in [0, 180], S \in [0, 255], V \in [0, 255]$) to decouple chromaticity from illumination. Handling hue wrap-around ($H_{\text{lower}} > H_{\text{upper}}$) via dual-mask bitwise OR operations.
- **Morphological Cleaning**: Applying structuring elements (`cv2.MORPH_ELLIPSE`) for `MORPH_OPEN` (noise removal) and `MORPH_CLOSE` (hole filling).
- **Contour Analysis**: Boundary tracing (`cv2.findContours`), arc length calculation, area integration, and bounding box computation.

### 2. Computational Geometry & Feature Engineering
- **Convex Hull & Convexity Defects**: Calculating `cv2.convexHull` and extracting finger valleys using `cv2.convexityDefects`.
- **Geometric Metrics**: Computing compactness metrics including Solidity ($\text{Area} / \text{Hull Area}$), Extent ($\text{Area} / \text{Bounding Box Area}$), Circularity ($4\pi \times \text{Area} / \text{Perimeter}^2$), and Elongation ($\max(\text{Aspect Ratio}, 1 / \text{Aspect Ratio})$).

### 3. Software Architecture & Design Patterns
- **Pipeline Architecture**: Decoupled, modular component structure (`CameraCapture` $\to$ `SkinDetector` $\to$ `HandDetector` $\to$ `GestureRecognizer` $\to$ `GestureHistory`).
- **Type-Safe Data Structures**: Utilizing dataclasses (`results.py`) for explicit communication contracts between processing stages.
- **State Machine Calibration**: Safe state management ensuring failed recalibration retains valid saved state without corrupting persistent configuration.

### 4. Robust Testing & Verification
- **Pytest Suite**: Complete unit, integration, and robustness testing without physical hardware dependencies.
- **Mocking & Synthetic Data**: Mocking `cv2.VideoCapture` hardware boundaries and generating synthetic geometric contours for deterministic automated testing.

### 5. Input Validation & Resource Management
- **Defensive API Guardrails**: Explicit type and shape checking on public entry points to prevent unhandled C++ exceptions in OpenCV bindings.
- **Safe Resource Cleanup**: Guaranteeing resource release (`VideoCapture.release()`) on error paths, exceptions, and process exit.
