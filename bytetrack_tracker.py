from __future__ import annotations

import numpy as np
import supervision as sv


class ByteTrackerWrapper:
    def __init__(self) -> None:
        self.tracker = sv.ByteTrack()

    def update(self, detections: list[tuple[int, int, int, int, float]]) -> list[dict]:
        # detections format: [(x1, y1, x2, y2, conf), ...]
        if not detections:
            empty = sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=np.int32),
            )
            self.tracker.update_with_detections(empty)
            return []

        sv_detections = sv.Detections(
            xyxy=np.array([d[:4] for d in detections], dtype=np.float32),
            confidence=np.array([d[4] for d in detections], dtype=np.float32),
            class_id=np.zeros((len(detections),), dtype=np.int32),
        )

        tracks = self.tracker.update_with_detections(sv_detections)

        tracker_ids = tracks.tracker_id if tracks.tracker_id is not None else []
        if len(tracker_ids) == 0:
            return []

        results: list[dict] = []
        for xyxy, confidence, tracker_id in zip(tracks.xyxy, tracks.confidence, tracker_ids):
            if tracker_id is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            results.append(
                {
                    "xyxy": (x1, y1, x2, y2),
                    "tracker_id": int(tracker_id),
                    "confidence": float(confidence),
                }
            )

        return results
