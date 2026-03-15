from __future__ import annotations

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - environment-specific import failure
    YOLO = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class PersonDetector:
    def __init__(self, model: str = "yolov8n.pt", conf: float = 0.4) -> None:
        if YOLO is None:
            raise RuntimeError(
                "Ultralytics/Torch is unavailable in this environment. "
                "On Windows, install Microsoft Visual C++ Redistributable and a compatible torch build. "
                f"Original error: {_IMPORT_ERROR}"
            )
        self.model = YOLO(model)
        self.conf = conf

    def detect(self, frame) -> list[tuple[int, int, int, int]]:
        frame_h, frame_w = frame.shape[:2]
        results = self.model(frame, conf=self.conf, verbose=False)

        boxes: list[tuple[int, int, int, int]] = []
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                # COCO class 0 = person
                if class_id != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, min(frame_w - 1, x1))
                y1 = max(0, min(frame_h - 1, y1))
                x2 = max(0, min(frame_w, x2))
                y2 = max(0, min(frame_h, y2))

                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue

                boxes.append((x1, y1, x2, y2))

        return boxes
