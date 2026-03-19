from __future__ import annotations

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - environment-specific import failure
    YOLO = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class ObjectDetector:
    def __init__(self, model: str = "yolov8n.pt", conf: float = 0.4) -> None:
        if YOLO is None:
            raise RuntimeError(
                "Ultralytics/Torch is unavailable in this environment. "
                "On Windows, install Microsoft Visual C++ Redistributable and a compatible torch build. "
                f"Original error: {_IMPORT_ERROR}"
            )

        self.model = YOLO(model)
        self.conf = conf

        # COCO class IDs
        self.targets = {
            24: "backpack",
            26: "handbag",
            39: "bottle",
            41: "cup",
            67: "cell phone",
        }

    def detect(self, frame) -> list[dict]:
        frame_h, frame_w = frame.shape[:2]
        results = self.model(frame, conf=self.conf, verbose=False)

        objects: list[dict] = []
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id not in self.targets:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, min(frame_w - 1, x1))
                y1 = max(0, min(frame_h - 1, y1))
                x2 = max(0, min(frame_w, x2))
                y2 = max(0, min(frame_h, y2))

                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue

                objects.append({
                    "label": self.targets[class_id],
                    "bbox": (x1, y1, x2, y2),
                })

        return objects
