import argparse
import time

import cv2
import mediapipe as mp

# MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from bytetrack_tracker import ByteTrackerWrapper
from item_memory import ItemMemory
from interaction_detector import InteractionDetector
from person_tracker import PersonTracker
from pose_behavior import classify_behavior
from unified_detector import UnifiedDetector


# BlazePose landmark edges used by Pose Landmarker (33 keypoints).
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
)


def to_pixel_keypoints(landmarks, w, h, offset_x=0, offset_y=0):
    points = []
    for lm in landmarks:
        x = int(lm.x * w) + offset_x
        y = int(lm.y * h) + offset_y
        if 0 <= x < w + offset_x and 0 <= y < h + offset_y:
            points.append((float(x), float(y)))
        else:
            points.append(None)
    return points


def draw_pose_skeleton(
    image,
    landmarks,
    crop_w,
    crop_h,
    offset_x=0,
    offset_y=0,
    color=(0, 255, 255),
    thickness=2,
):
    pixel_points: list[tuple[int, int] | None] = []
    for lm in landmarks:
        x = int(lm.x * crop_w) + offset_x
        y = int(lm.y * crop_h) + offset_y
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            pixel_points.append((x, y))
        else:
            pixel_points.append(None)

    for start_index, end_index in POSE_CONNECTIONS:
        if start_index >= len(pixel_points) or end_index >= len(pixel_points):
            continue
        start_point = pixel_points[start_index]
        end_point = pixel_points[end_index]
        if start_point is None or end_point is None:
            continue
        cv2.line(image, start_point, end_point, color, thickness)

    for point in pixel_points:
        if point is not None:
            cv2.circle(image, point, 3, color, -1)


# =============================
# Argument parsing
# =============================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default="pose_landmarker.task")
    parser.add_argument("--frame-skip", type=int, default=1)
    return parser.parse_args()


# =============================
# Main pipeline
# =============================

def main():
    args = parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    frame_skip = max(1, int(args.frame_skip))
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video source")

    # MediaPipe model
    base_options = python.BaseOptions(model_asset_path=args.model)

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

    while True:
        frame_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        if frame_skip > 1 and (frame_index % frame_skip) != 0:
            cv2.imshow("Pose Behavior Monitor", frame)
            if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
                break
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
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_crop,
            )

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

                if interaction["item_touch_labels"]:
                    touch_text = "touching: " + ",".join(interaction["item_touch_labels"])
                    cv2.putText(
                        output,
                        touch_text,
                        (x1, min(frame_h - 8, y2 + 35)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                    )

                if "item_disappearance" in behavior.reasons:
                    cv2.putText(
                        output,
                        "ITEM DISAPPEARED",
                        (x1, max(18, y1 - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
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

        cv2.imshow("Pose Behavior Monitor", output)

        if is_live_source:
            wait_ms = 1
        else:
            elapsed_ms = (time.perf_counter() - frame_start) * 1000.0
            wait_ms = int(target_frame_ms - elapsed_ms)
            if wait_ms < 1:
                wait_ms = 1

        if cv2.waitKey(wait_ms) & 0xFF in [27, ord("q")]:
            break

    pose_detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
