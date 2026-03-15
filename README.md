# MediaPipe Real-Time Shoplifting Heuristic Detector

This project detects human pose from:

- Webcam stream (default)
- Recorded video file

It overlays:

- Multi-person MediaPipe pose skeletons
- A person bounding box
- A behavior score and label: `NORMAL` or `SUSPICIOUS`

Detection pipeline:

- YOLO person detection (`ultralytics` / `yolov8n.pt`) provides person bounding boxes.
- MediaPipe Pose Landmarker runs inside each detected person crop.
- Landmarks are mapped back to full-frame coordinates for tracking and behavior scoring.

The suspicious label is based on lightweight hardcoded heuristics for concealment risk:

- Wrist near torso or bag region
- Item touch then disappearance near torso
- Center-crossing concealment gesture
- Look-around head turns, rapid hand retract, and long loitering

The logic is implemented in local modules:

- `interaction_detector.py`
- `person_tracker.py`
- `pose_behavior.py`

## 1) Setup

Use your existing virtual environment and install dependencies:

```powershell
cd D:\projects\mediapipe
.\med_venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Run with webcam

```powershell
python main.py --source 0
```

## 3) Run with a recorded video

```powershell
python main.py --source "D:\path\to\video.mp4"
```

Press `q` or `Esc` to quit.

## Notes

- This is a rule-based baseline optimized for low overhead.
- Runtime supports multiple simultaneous person tracks.
- You can tune thresholds in `pose_behavior.py`.
