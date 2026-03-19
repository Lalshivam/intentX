from __future__ import annotations

import time


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class ItemMemory:
    def __init__(self) -> None:
        self.last_seen_time: dict[str, float] = {}
        self.last_seen_center: dict[str, tuple[float, float]] = {}
        self.current_labels: set[str] = set()

    def update(self, objects: list[dict]) -> None:
        now = time.monotonic()
        self.current_labels = set()

        for obj in objects:
            label = obj.get("label")
            bbox = obj.get("bbox")
            if label is None or bbox is None:
                continue

            self.current_labels.add(label)
            self.last_seen_time[label] = now
            self.last_seen_center[label] = _bbox_center(tuple(bbox))

    def disappeared_recently(self, label: str, now: float, window: float = 1.2) -> bool:
        if label not in self.last_seen_time:
            return False
        if label in self.current_labels:
            return False
        return now - self.last_seen_time[label] <= window
