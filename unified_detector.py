from __future__ import annotations

import cv2

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - environment-specific import failure
    YOLO = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class UnifiedDetector:
    def __init__(self, model: str = "yolov8n.pt", conf: float = 0.4) -> None:
        self.model = None
        self._backend = "opencv-hog"
        self._fallback_reason = ""

        if YOLO is not None:
            try:
                self.model = YOLO(model)
                self._backend = "ultralytics"
            except Exception as exc:  # pragma: no cover - environment-specific init failure
                self._fallback_reason = str(exc)
        else:
            self._fallback_reason = str(_IMPORT_ERROR)

        self.conf = conf

        # COCO class IDs
        self.object_targets = {
            24: "backpack",
            26: "handbag",
            39: "bottle",
            41: "cup",
            67: "cell phone",
        }

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Keep fallback responsive on CPU-only machines.
        self._hog_every_n = 3
        self._hog_scale = 0.5
        self._hog_frame_index = 0
        self._cached_people: list[tuple[int, int, int, int, float]] = []

        if self._backend != "ultralytics":
            print(
                "[WARN] Ultralytics/Torch unavailable; using OpenCV HOG fallback "
                "(person-only, no item classes)."
            )
            if self._fallback_reason:
                print(f"[WARN] Detector fallback reason: {self._fallback_reason}")

    def detect(self, frame) -> tuple[list[tuple[int, int, int, int, float]], list[dict]]:
        if self._backend != "ultralytics":
            return self._detect_with_hog(frame)

        frame_h, frame_w = frame.shape[:2]
        results = self.model(frame, conf=self.conf, verbose=False)

        person_boxes: list[tuple[int, int, int, int, float]] = []
        objects: list[dict] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, min(frame_w - 1, x1))
                y1 = max(0, min(frame_h - 1, y1))
                x2 = max(0, min(frame_w, x2))
                y2 = max(0, min(frame_h, y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                if class_id == 0:
                    if x2 - x1 >= 8 and y2 - y1 >= 8:
                        person_boxes.append((x1, y1, x2, y2, confidence))
                    continue

                if class_id in self.object_targets and x2 - x1 >= 4 and y2 - y1 >= 4:
                    objects.append(
                        {
                            "label": self.object_targets[class_id],
                            "bbox": (x1, y1, x2, y2),
                        }
                    )

        return person_boxes, objects

    def _detect_with_hog(self, frame) -> tuple[list[tuple[int, int, int, int, float]], list[dict]]:
        self._hog_frame_index += 1
        if self._hog_frame_index % self._hog_every_n != 0:
            return list(self._cached_people), []

        frame_h, frame_w = frame.shape[:2]
        scaled = cv2.resize(
            frame,
            None,
            fx=self._hog_scale,
            fy=self._hog_scale,
            interpolation=cv2.INTER_LINEAR,
        )

        boxes, weights = self.hog.detectMultiScale(
            scaled,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.08,
        )

        person_boxes: list[tuple[int, int, int, int, float]] = []
        for (x, y, w, h), weight in zip(boxes, weights):
            x1 = max(0, int(x / self._hog_scale))
            y1 = max(0, int(y / self._hog_scale))
            x2 = min(frame_w, int((x + w) / self._hog_scale))
            y2 = min(frame_h, int((y + h) / self._hog_scale))

            if x2 - x1 < 8 or y2 - y1 < 8:
                continue

            confidence = max(0.01, min(0.99, float(weight)))
            person_boxes.append((x1, y1, x2, y2, confidence))

        if len(person_boxes) > 6:
            person_boxes.sort(key=lambda b: b[4], reverse=True)
            person_boxes = person_boxes[:6]

        self._cached_people = person_boxes

        # HOG fallback does not provide object classes (bag/phone/bottle/cup).
        return person_boxes, []
