from __future__ import annotations

import os
import platform
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
from flask import Flask, Response, jsonify, render_template, request

from bytetrack_tracker import ByteTrackerWrapper
from interaction_detector import InteractionDetector
from item_memory import ItemMemory
from main import draw_pose_skeleton, to_pixel_keypoints
from person_tracker import PersonTracker
from pose_behavior import classify_behavior
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from unified_detector import UnifiedDetector


WORKSPACE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = WORKSPACE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


class VideoEngine:
    def __init__(self, model_path: str = "pose_landmarker.task") -> None:
        self.model_path = model_path
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._status_lock = threading.Lock()

        self._latest_jpeg: bytes | None = None
        self._status: dict[str, Any] = {
            "running": False,
            "mode": "idle",
            "source": None,
            "message": "idle",
            "backend": None,
        }

    def _set_status(self, **kwargs: Any) -> None:
        with self._status_lock:
            self._status.update(kwargs)

    def get_status(self) -> dict[str, Any]:
        with self._status_lock:
            return dict(self._status)

    def get_latest_jpeg(self) -> bytes | None:
        with self._frame_lock:
            return self._latest_jpeg

    def start(self, source: str, frame_skip: int = 1) -> None:
        self.stop()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_pipeline,
            args=(source, max(1, frame_skip)),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._set_status(running=False, mode="idle", message="stopped")

    def _run_pipeline(self, source_text: str, frame_skip: int) -> None:
        source: int | str
        source = int(source_text) if source_text.isdigit() else source_text
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            self._set_status(
                running=False,
                mode="error",
                source=source_text,
                message="Cannot open source",
            )
            return

        pose_detector = None
        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
            )
            pose_detector = vision.PoseLandmarker.create_from_options(options)

            detector = UnifiedDetector()
            tracker = ByteTrackerWrapper()
            item_memory = ItemMemory()
            interaction_detector = InteractionDetector()
            person_tracker = PersonTracker(retention_seconds=3.0)

            source_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval_ms = int(1000 / source_fps) if source_fps and source_fps > 1 else 33
            target_frame_ms = float(frame_interval_ms)
            is_live_source = isinstance(source, int)
            frame_index = 0

            backend_name = getattr(detector, "_backend", "unknown")
            self._set_status(
                running=True,
                mode="running",
                source=source_text,
                message="running",
                backend=backend_name,
            )

            while not self._stop_event.is_set():
                frame_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    self._set_status(running=False, mode="ended", message="Stream ended")
                    break

                frame_index += 1
                if frame_skip > 1 and (frame_index % frame_skip) != 0:
                    self._publish_frame(frame)
                    continue

                output = frame.copy()
                frame_h, frame_w = frame.shape[:2]
                now = time.monotonic()

                person_boxes, objects = detector.detect(frame)
                item_memory.update(objects)
                tracks = tracker.update(person_boxes)

                for obj in objects:
                    x1, y1, x2, y2 = obj["bbox"]
                    cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(
                        output,
                        obj["label"],
                        (x1, max(18, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                    )

                detections = []
                for track in tracks:
                    x1, y1, x2, y2 = track["xyxy"]
                    track_id = track["tracker_id"]

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                    result = pose_detector.detect(mp_image)

                    if not result.pose_landmarks:
                        continue

                    landmarks = result.pose_landmarks[0]
                    keypoints = to_pixel_keypoints(landmarks, x2 - x1, y2 - y1, x1, y1)
                    detections.append((landmarks, (x1, y1, x2, y2), keypoints, track_id))

                if detections:
                    for landmarks, bbox, keypoints, track_id in detections:
                        x1, y1, x2, y2 = bbox

                        interaction = interaction_detector.analyze(
                            bbox=bbox,
                            keypoints=keypoints,
                            items=objects,
                            frame_shape=frame.shape,
                        )

                        state = person_tracker.update(
                            track_id=track_id,
                            bbox=bbox,
                            keypoints=keypoints,
                            interaction=interaction,
                            timestamp=now,
                        )

                        behavior = classify_behavior(state, now, item_memory=item_memory)
                        color = (0, 0, 255) if behavior.label == "SUSPICIOUS" else (0, 200, 0)

                        draw_pose_skeleton(
                            output,
                            landmarks,
                            x2 - x1,
                            y2 - y1,
                            offset_x=x1,
                            offset_y=y1,
                            color=(0, 255, 255),
                            thickness=2,
                        )
                        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

                        label = f"id={track_id} {behavior.label} score={behavior.score:.1f}"
                        cv2.putText(
                            output,
                            label,
                            (x1, max(18, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                        )

                        reasons_text = ", ".join(behavior.reasons[:2]) if behavior.reasons else "none"
                        cv2.putText(
                            output,
                            f"reasons: {reasons_text}",
                            (x1, min(frame_h - 8, y2 + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            color,
                            1,
                        )

                    person_tracker.prune(now)
                else:
                    person_tracker.prune(time.monotonic())
                    cv2.putText(
                        output,
                        "No pose detected",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (200, 200, 200),
                        2,
                    )

                self._publish_frame(output)

                if not is_live_source:
                    elapsed_ms = (time.perf_counter() - frame_start) * 1000.0
                    sleep_ms = target_frame_ms - elapsed_ms
                    if sleep_ms > 1:
                        time.sleep(sleep_ms / 1000.0)

        except Exception as exc:
            self._set_status(running=False, mode="error", message=str(exc))
        finally:
            if pose_detector is not None:
                pose_detector.close()
            cap.release()
            if self.get_status().get("mode") == "running":
                self._set_status(running=False, mode="idle", message="stopped")

    def _publish_frame(self, frame) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        with self._frame_lock:
            self._latest_jpeg = encoded.tobytes()


app = Flask(__name__, template_folder="templates", static_folder="static")
engine = VideoEngine()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    return jsonify(engine.get_status())


@app.post("/api/upload")
def upload_video():
    file = request.files.get("video")
    if file is None or file.filename is None or file.filename.strip() == "":
        return jsonify({"ok": False, "error": "No video uploaded"}), 400

    suffix = Path(file.filename).suffix or ".mp4"
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    path = UPLOAD_DIR / safe_name
    file.save(path)

    return jsonify({"ok": True, "path": str(path)})


@app.post("/api/start")
def start_stream():
    payload = request.get_json(silent=True) or {}
    source = str(payload.get("source", "0"))
    frame_skip = int(payload.get("frame_skip", 1))

    engine.start(source=source, frame_skip=frame_skip)
    return jsonify({"ok": True, "status": engine.get_status()})


@app.post("/api/stop")
def stop_stream():
    engine.stop()
    return jsonify({"ok": True, "status": engine.get_status()})


@app.post("/api/terminal/run")
def run_terminal_command():
    payload = request.get_json(silent=True) or {}
    command = str(payload.get("command", "")).strip()
    timeout = int(payload.get("timeout", 30))

    if not command:
        return jsonify({"ok": False, "error": "Command is empty"}), 400

    if platform.system().lower().startswith("win"):
        shell_cmd = ["powershell", "-NoProfile", "-Command", command]
    else:
        shell_cmd = ["/bin/sh", "-c", command]

    completed = subprocess.run(
        shell_cmd,
        cwd=str(WORKSPACE_DIR),
        capture_output=True,
        text=True,
        timeout=max(1, min(timeout, 120)),
    )

    return jsonify(
        {
            "ok": True,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    )


@app.get("/video_feed")
def video_feed():
    def generate():
        try:
            while True:
                frame = engine.get_latest_jpeg()
                if frame is None:
                    time.sleep(0.08)
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
        except GeneratorExit:
            return

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
