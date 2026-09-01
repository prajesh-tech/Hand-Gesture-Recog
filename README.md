# Real-Time Hand Gesture Recognition for Interactive Systems

A Python-based computer vision application that detects and recognizes hand gestures in real-time using classical HSV skin-color segmentation and geometric feature analysis. No deep learning required—fully explainable and suitable for college-level coursework.

## 🎯 Project Overview

This application captures video from a webcam, detects the user's hand using HSV color-space thresholding, and recognizes four distinct hand gestures in real time. Each gesture is mapped to an interactive action (START, STOP, SELECT, NEXT), and the system uses temporal smoothing to provide stable, flicker-free gesture recognition.

### Key Features
- ✅ **User-adaptive HSV calibration** on first run (calibration saved locally, not in repository)
- ✅ **Skin detection** using configurable HSV thresholds with morphological cleanup
- ✅ **Hand contour detection** from binary masks with noise filtering
- ✅ **4-gesture recognition** using measurable geometric features (solidity, extent, aspect ratio, defects)
- ✅ **Temporal smoothing** with consensus voting to eliminate flicker
- ✅ **"No Hand Detected" state** instead of forced classification
- ✅ **Real-time FPS measurement and display**
- ✅ **Modular, explainable architecture** suitable for viva discussion
- ✅ **Comprehensive unit and integration tests** (77 tests, all passing)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.x |
| Computer Vision | OpenCV 5.0 |
| Numerical Computing | NumPy |
| Testing | pytest |
| Color Space | HSV (hue, saturation, value) |
| Gesture Logic | Classical CV features (convex hull, defects, solidity, extent) |

### ⚠️ What We DON'T Use
- ❌ Deep learning or neural networks
- ❌ Pre-trained gesture models
- ❌ MediaPipe or other ML toolkits
- ❌ YOLO or object detection models
- ❌ TensorFlow, PyTorch, or similar

---

## 📊 System Workflow

```
┌─────────────────┐
│    Webcam       │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Preprocessing: Resize + Blur + BGR→HSV Conversion   │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Skin-Tone Calibration (First Run)                   │
│ • User places palm in center, presses SPACE (10x)   │
│ • Compute HSV min/max from samples                  │
│ • Save to config/hsv_calibration.json (Git-ignored) │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Skin Color Segmentation (HSV Thresholding)          │
│ • Apply saved/calibrated HSV range                  │
│ • Output: Binary mask (white=skin, black=background)│
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Morphological Cleanup                               │
│ • Opening (remove noise)                            │
│ • Closing (fill holes)                              │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Contour Detection                                   │
│ • Find all contours in mask                         │
│ • Select largest (assumed to be hand)               │
│ • Filter by minimum area (>500 pixels)              │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Feature Extraction (Geometric Analysis)             │
│ • Area (contour pixels)                             │
│ • Solidity (area / convex_hull_area)                │
│ • Extent (area / bounding_rect_area)                │
│ • Aspect Ratio (width / height)                     │
│ • Convexity Defects (indentations/fingers)          │
│ • Circularity (perimeter² / area)                   │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Gesture Classification (Rule-Based)                 │
│ • Compare features to tunable thresholds            │
│ • Output: "Fist" | "Open Palm" | "One Finger"      │
│           "Two Fingers" | None                      │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Temporal Smoothing (Consensus Voting)               │
│ • Track last N predictions (e.g., 10 frames)        │
│ • Output gesture only if M frames agree (e.g., 5)   │
│ • Reduces flicker and noise                         │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Action Mapping & Display                            │
│ • Fist → STOP                                       │
│ • Open Palm → START                                 │
│ • One Finger → SELECT                               │
│ • Two Fingers → NEXT                                │
│ • Show: contour, box, gesture, action, FPS         │
└─────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites
- Python 3.7+
- USB webcam (standard, no special requirements)

### Setup

```bash
# Clone or download the project
cd hand-gesture-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Basic Usage
```bash
source venv/bin/activate
python src/main.py
```

The application will:
1. **Initialize the camera**
2. **Run calibration mode** (if first time):
   - Show camera feed with center box
   - Prompt: "Place your PALM in the center. Press SPACE to sample."
   - You press SPACE 10 times while keeping hand steady
   - System computes HSV range from samples
   - Saves to `config/hsv_calibration.json`
3. **Start main loop**:
   - Display live webcam with gesture/action labels and FPS counter
   - Press `q` to quit
   - Press `c` to recalibrate

### Calibration and Diagnostic Workflow

Place your palm so that it fills most of the centre ROI but does not touch its
edges. Use an evenly lit area and avoid skin-coloured objects behind the ROI.
Calibration discards very dark and desaturated pixels, then uses percentile
bounds with a tolerance margin; this is less sensitive to a few background
pixels than a raw min/max range. Recalibrate with `C` whenever lighting changes.

For pipeline diagnostics after calibration, run:

```bash
python src/main_debug.py
```

It opens the camera frame with its selected contour and feature values, plus
separate raw and cleaned HSV-mask windows. It displays raw and smoothed gesture
labels, current HSV bounds, and measured FPS. Press `Q` to exit. Use these
values to tune `GestureRecognizer.THRESHOLDS`; synthetic tests only validate
pipeline behaviour, not real-world recognition accuracy.

---

## 🖐️ Supported Gestures

### 1. **Fist** → **STOP**
- **Appearance**: Closed hand, compact shape
- **Features**: Compact contour, high solidity and extent, with no significant defects
- **Use Case**: Stop or deactivate current action

### 2. **Open Palm** → **START**
- **Appearance**: Fingers spread, low compactness
- **Features**: Several deep finger valleys (convexity defects) and lower solidity
- **Use Case**: Start or activate action

### 3. **One Finger** → **SELECT**
- **Appearance**: Single finger pointing, elongated
- **Features**: Elongated contour with at most one significant defect
- **Use Case**: Select item or navigate

### 4. **Two Fingers** → **NEXT**
- **Appearance**: Two fingers (peace sign or index+middle)
- **Features**: Moderately elongated contour with one or two significant defects
- **Use Case**: Move to next item or advance

---

## 🧮 Algorithm Explanation

### HSV Color Space
- **H (Hue)**: Color (0-180 in OpenCV, representing 0-360°)
- **S (Saturation)**: Color intensity (0-255, more saturated = purer color)
- **V (Value)**: Brightness (0-255, darker to lighter)

**Why HSV?** More robust to lighting changes than RGB. Skin tones occupy a specific HSV region.

### Skin Segmentation Process
1. Convert BGR frame to HSV
2. Apply thresholding: keep only pixels where H, S, V fall within calibrated range
3. Result: binary mask (white where skin detected, black elsewhere)
4. Apply morphological operations:
   - **Opening** (erode then dilate): removes small noise spots
   - **Closing** (dilate then erode): fills small holes in detected regions

Hue ranges crossing OpenCV's 0/180 boundary are supported. Blur-kernel size,
morphology kernel size, and opening/closing iterations are constructor settings
on `SkinDetector`; use smaller kernels when fine finger gaps are being removed.

### Contour Detection
- Find all connected white regions (contours) in binary mask
- Filter by minimum area (ignore tiny noise)
- Select the largest contour (assumed to be the hand)

### Feature Extraction & Gesture Classification

| Gesture | Solidity | Defects | Extent | Aspect Ratio | Area |
|---------|----------|---------|--------|--------------|------|
| **Fist** | High | 0 | High | Compact | Valid hand |
| **Open Palm** | Lower | 3+ | Variable | Variable | Valid hand |
| **One Finger** | Variable | 0-1 | Variable | Highly elongated | Valid hand |
| **Two Fingers** | Variable | 1-2 | Variable | Moderately elongated | Valid hand |

A valid contour that does not match a rule is displayed as **Unknown Gesture**.
`No Hand Detected` is reserved for empty or undersized contours.

**These thresholds are starting points and should be tuned based on your environment**, camera, and hand size. See "Threshold Tuning" section below.

### Temporal Smoothing
Without smoothing, classifications would flicker frame-to-frame due to slight contour variations. Solution:
1. Store the last 8 gesture predictions in a buffer
2. Use majority voting: output a gesture only when 4 frames agree
3. Example:
   - Frames 1-3: [Fist, Fist, Open Palm] → no consensus yet, output: None
   - Frames 1-5: [Fist, Fist, Fist, Fist, Open Palm] → Fist has 4/5, still not 5 → output: None
   - Frames 1-6: [Fist, Fist, Fist, Fist, Fist, Open Palm] → Fist has 5/6 → output: **Fist**

This reduces flicker and provides stable output for interactive applications.

---

## ⚙️ Threshold Tuning Guide

**The gesture classification thresholds are NOT magic numbers**—they depend on:
- Camera distance from hand
- Hand size (varies by person)
- Lighting conditions
- Background complexity
- Camera resolution

### How to Tune

1. **Run the application and observe**:
   ```bash
   python src/main.py
   ```
   Go through calibration, then make each gesture and watch the console output (if you modify main.py to print features).

2. **Extract feature values**:
   - Modify `src/main.py` temporarily to print features:
     ```python
     features = gesture_recognizer.gesture_recognizer.extract_features(hand_contour)
     print(f"Fist attempt - Solidity: {features['solidity']:.3f}, Defects: {features['convexity_defects_count']}")
     ```

3. **Identify patterns**:
   - Collect 10-20 samples per gesture
   - Note min/max for each feature
   - Adjust thresholds in `GestureRecognizer.THRESHOLDS` dict

4. **Example adjustment**:
   ```python
   # If "Fist" is being misclassified as "Open Palm"
   # Adjust threshold to be more strict:
   GestureRecognizer.THRESHOLDS['fist_solidity_min'] = 0.8  # was 0.75
   GestureRecognizer.THRESHOLDS['fist_defects_max'] = 2      # was 3
   ```

5. **Re-calibrate if lighting changes dramatically** (press 'c' in main loop)

---

## 📂 Project Structure

```
hand-gesture-recognition/
├── src/
│   ├── main.py                 # Application entry point & event loop
│   ├── camera.py               # CameraCapture class
│   ├── skin_detection.py       # SkinDetector class (HSV thresholding)
│   ├── hand_detection.py       # HandDetector class (contour analysis)
│   ├── gesture_recognition.py  # GestureRecognizer class (feature extraction & classification)
│   ├── gesture_history.py      # GestureHistory class (temporal smoothing)
│   └── calibration.py          # CalibrationManager (save/load HSV thresholds)
├── tests/
│   ├── test_skin_detection.py
│   ├── test_hand_detection.py
│   ├── test_gesture_recognition.py
│   ├── test_gesture_history.py
│   ├── test_calibration.py
│   ├── test_improved_segmentation.py
│   ├── test_improvements.py
│   ├── test_main_console.py
│   └── test_integration.py
├── config/
│   └── hsv_calibration.json    # User's calibrated HSV thresholds (created at runtime, Git-ignored)
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── .gitignore                  # Python + config/ ignores
```

---

## 🧪 Testing

### Run All Tests
```bash
./venv/bin/pytest -v
```

### Test Coverage
- **Unit tests**: Skin detection, hand detection, gesture recognition, gesture history, calibration (62 tests)
- **Integration & Regression tests**: Synthetic hands, contour scoring, pipeline robustness, console safety (15 tests)
- **All tests pass**: ✅ 77/77 passing

### Running Specific Tests
```bash
pytest tests/test_gesture_recognition.py -v
pytest tests/test_integration.py::TestIntegrationSyntheticHands::test_full_pipeline_with_circle -v
```

---

## ⚡ Performance Considerations

### Real-Time Performance
- **Target**: Smooth interactive response (ideally 20+ FPS on CPU)
- **Measured**: Actual FPS displayed on screen
- **Bottlenecks**: Skin detection (HSV conversion) typically fastest; morphology can be slower on large frames

### Optimization Tips
1. **Reduce frame size**: Modify `frame_width=640, frame_height=480` in `main.py`
2. **Use resize factor**: Pass `resize_factor=0.5` to halve input dimensions (4x pixel reduction)
3. **Adjust morphology kernel**: Smaller kernels in `SkinDetector(morphology_kernel_size=5)` = faster but less cleanup
4. **Skip frames**: Modify main loop to process every Nth frame for gesture recognition only

---

## 🔧 Troubleshooting

### Issue: "Failed to open camera"
- **Cause**: Camera not available or permission denied
- **Solution**: Check camera is connected, ensure no other app is using it

### Issue: Gesture recognition not working or very inaccurate
- **Cause**: HSV calibration not suitable for your lighting/skin tone
- **Solution**: Press 'c' to recalibrate; ensure good lighting; place hand clearly in center

### Issue: Gestures flicker rapidly
- **Cause**: Temporal smoothing threshold too high for your environment
- **Solution**: Tune `GestureHistory(buffer_size=8, consensus_threshold=4)` in `main.py`

### Issue: Performance is slow (FPS < 15)
- **Cause**: Frame size too large or CPU limited
- **Solution**: Reduce `frame_width`/`frame_height`, or use `resize_factor=0.5`

---

## 🚀 Future Improvements (Not in V1)

- [ ] Support for multiple simultaneous hands
- [ ] Gesture recording and custom gesture training
- [ ] Hand pose estimation (finger angles, wrist position)
- [ ] Kalman filtering for smoother contour tracking
- [ ] Background subtraction (for complex backgrounds)
- [ ] Confidence scores alongside gesture predictions
- [ ] Gesture sequence recognition ("draw a triangle")
- [ ] Mobile/embedded deployment (TensorFlow Lite)
- [ ] Voice feedback or haptic confirmation
- [ ] GUI beyond OpenCV window

---

## 📚 References & Concepts

### Classical CV Techniques Used
- **Color space conversion**: BGR ↔ HSV using OpenCV `cvtColor()`
- **Thresholding**: Binary segmentation with `inRange()`
- **Morphological operations**: Opening/closing with `morphologyEx()`
- **Contour detection**: `findContours()` with external hierarchy
- **Convex hull**: `convexHull()` for gesture features
- **Convexity defects**: `convexityDefects()` to count finger indentations

### Related Topics (For Extended Learning)
- HSV color space properties and advantages over RGB
- Morphological image processing (dilation, erosion, opening, closing)
- Contour moments and shape descriptors
- Gesture recognition taxonomies in HCI (Human-Computer Interaction)
- Real-time video processing pipelines

---

## 📋 Limitations

1. **Single hand only**: Focuses on largest contour; doesn't support dual-hand gestures
2. **Skin tone dependent**: HSV range calibration needed per user/lighting
3. **Artificial lighting sensitivity**: Performs best with consistent, natural lighting
4. **No finger tracking**: Detects overall hand shape, not individual finger positions
5. **Gesture set is fixed**: Only 4 gestures in V1
6. **No depth information**: 2D recognition only; camera distance affects accuracy
7. **Synthetic test data**: Synthetic shapes in tests don't perfectly match real hand contours
8. **No hand pose**: Cannot determine hand orientation beyond gross gesture

---

## 🎓 For College Assignments & Viva

This project is designed to be **explainable and defensible** in an academic setting:

### Key Points to Discuss
1. **Why HSV, not RGB?** HSV separates color from intensity; more robust to lighting changes
2. **Why no deep learning?** Classical CV is interpretable; you can explain every decision
3. **How does calibration work?** Captures user's actual skin tone; adapts to environment
4. **What are convexity defects?** Indentations in convex hull; indicate finger presence
5. **Why temporal smoothing?** Removes noise from frame-to-frame classification jitter
6. **How to tune thresholds?** Collect feature samples; analyze distributions; adjust cutoffs

### Viva Questions to Prepare For
- "Explain the workflow from webcam to gesture output"
- "Why does solidity help distinguish Fist from Open Palm?"
- "How would you extend this to 6 gestures? What new features would you use?"
- "What happens if lighting changes? How does calibration help?"
- "How does temporal smoothing prevent flicker?"
- "Can this work on mobile? What optimizations would you apply?"

---

## 📝 License & Usage

This project is provided as-is for educational and research purposes. Adapt and extend as needed for your coursework.

---

## ✅ Implementation Checklist

- [x] Webcam capture with error handling
- [x] HSV calibration mode (mandatory first run)
- [x] Skin detection with morphological cleanup
- [x] Hand contour detection and filtering
- [x] Geometric feature extraction (6+ features)
- [x] Rule-based gesture classification (4 gestures)
- [x] Temporal smoothing (consensus voting)
- [x] Action mapping and display
- [x] Real-time FPS measurement
- [x] 77 unit + integration tests (all passing)
- [x] Comprehensive documentation
- [x] Code is modular and explainable
- [x] Git-ignored calibration file (not in repository)

---

**Ready to run!** Start with:
```bash
python src/main.py
```

Good luck with your gesture recognition system! 🎉
