# MediaPipe Real-Time Shoplifting Heuristic Detector

This project detects human pose from:

- Webcam stream (default)
- Recorded video file

It overlays:

- Multi-person MediaPipe pose skeletons
- A person bounding box
- A behavior score and label: `NORMAL` or `SUSPICIOUS`

Detection pipeline:

- Single YOLOv8 pass (`ultralytics` / `yolov8n.pt`) detects both people and target objects.
- ByteTrack (`supervision`) assigns stable track IDs for multi-person scenes.
- MediaPipe Pose Landmarker runs inside each detected person crop.
- Landmarks are mapped back to full-frame coordinates for tracking and behavior scoring.

The suspicious label is based on lightweight hardcoded heuristics for concealment risk:

- Wrist near torso or bag region
- Item touch then disappearance near torso
- Center-crossing concealment gesture
- Look-around head turns, rapid hand retract, and long loitering

The logic is implemented in local modules:

- `interaction_detector.py`
- `item_memory.py`
- `person_tracker.py`
- `pose_behavior.py`

## Final Architecture

Camera / Video
	|
	v
YOLOv8 Detection
(person + items + bags)
	|
	v
ByteTrack Tracker
(stable person IDs)
	|
	v
MediaPipe Pose
(hand + body landmarks)
	|
	v
Interaction Detector
(hand <-> object)
(hand <-> torso)
(hand <-> bag)
	|
	v
Item Memory
(track item disappearance)
	|
	v
Behavior Engine
(head turns)
(hand speed)
(loitering)
(item concealment)
	|
	v
Suspicion Score
	|
	v
Label Output
NORMAL / SUSPICIOUS

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

## 4) Run Web Interface

Start the web server:

```powershell
python web_app.py
```

Open your browser at:

```text
http://localhost:5000
```

Web features:

- Select live webcam or upload a recorded video.
- Start and stop the detection stream from the UI.
- Run terminal commands in parallel from the web terminal panel.

This web mode runs independently from your shell, so you can also execute commands in your regular terminal while the stream is running.

## 5) Docker Deployment

Build image:

```powershell
docker build -t pose-monitor-web .
```

Run container:

```powershell
docker run --rm -p 5000:5000 pose-monitor-web
```

Then open:

```text
http://localhost:5000
```

Notes:

- Uploaded videos are stored in `uploads/` inside the container runtime.
- Webcam passthrough in Docker is OS-dependent; recorded video mode is the most portable in containers.

## Notes

- This is a rule-based baseline optimized for low overhead.
- Runtime supports multiple simultaneous person tracks.
- You can tune thresholds in `pose_behavior.py`.
