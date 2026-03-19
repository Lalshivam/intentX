from __future__ import annotations

import math
from typing import Iterable

# MediaPipe pose landmark indexes.
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

BAG_LABELS = {"backpack", "handbag", "suitcase"}


def _point_from_keypoint(
    keypoints: list[tuple[float, float] | None] | None,
    index: int,
) -> tuple[float, float] | None:
    if keypoints is None or index >= len(keypoints):
        return None
    return keypoints[index]


def _distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    if a is None or b is None:
        return float("inf")
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt((dx * dx) + (dy * dy))


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class InteractionDetector:
    def __init__(self, zones: Iterable | None = None) -> None:
        # Kept for API compatibility; zones are intentionally ignored.
        _ = zones
        self.zones = []

    def analyze(
        self,
        bbox: tuple[int, int, int, int],
        keypoints: list[tuple[float, float] | None] | None,
        items: list[dict],
        frame_shape: tuple[int, int, int],
    ) -> dict:
        _ = frame_shape

        result = {
            "left_wrist": None,
            "right_wrist": None,
            "torso_center": None,
            "near_torso": False,
            "near_bag": False,
            "active_zones": [],
            "is_near_shelf": False,
            "head_offset": 0.0,
            "item_touch_labels": [],
            "bag_regions": [],
        }

        if not keypoints:
            return result

        left_wrist = _point_from_keypoint(keypoints, LEFT_WRIST)
        right_wrist = _point_from_keypoint(keypoints, RIGHT_WRIST)
        left_shoulder = _point_from_keypoint(keypoints, LEFT_SHOULDER)
        right_shoulder = _point_from_keypoint(keypoints, RIGHT_SHOULDER)
        left_hip = _point_from_keypoint(keypoints, LEFT_HIP)
        right_hip = _point_from_keypoint(keypoints, RIGHT_HIP)
        nose = _point_from_keypoint(keypoints, NOSE)

        result["left_wrist"] = left_wrist
        result["right_wrist"] = right_wrist

        torso_center = self._torso_center(bbox, left_shoulder, right_shoulder, left_hip, right_hip)
        result["torso_center"] = torso_center

        item_touch_labels: list[str] = []
        bag_regions: list[tuple[int, int, int, int]] = []

        # HAND <-> ITEM proximity
        for item in items:
            item_bbox = item.get("bbox")
            item_label = item.get("label")
            if item_bbox is None or item_label is None:
                continue

            item_box = tuple(item_bbox)
            item_center = _bbox_center(item_box)
            if _distance(left_wrist, item_center) < 80 or _distance(right_wrist, item_center) < 80:
                item_touch_labels.append(item_label)

            if item_label in BAG_LABELS:
                bag_regions.append(item_box)

        # HAND <-> TORSO proximity
        if _distance(left_wrist, torso_center) < 100 or _distance(right_wrist, torso_center) < 100:
            result["near_torso"] = True

        # HAND <-> BAG proximity
        for bag_bbox in bag_regions:
            bag_center = _bbox_center(bag_bbox)
            if _distance(left_wrist, bag_center) < 90 or _distance(right_wrist, bag_center) < 90:
                result["near_bag"] = True
                break

        shoulder_center_x = None
        shoulder_width = 0.0
        if left_shoulder is not None and right_shoulder is not None:
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
            shoulder_width = max(1.0, abs(right_shoulder[0] - left_shoulder[0]))

        head_offset = 0.0
        if nose is not None and shoulder_center_x is not None:
            head_offset = (nose[0] - shoulder_center_x) / shoulder_width

        result["head_offset"] = head_offset
        result["item_touch_labels"] = item_touch_labels
        result["bag_regions"] = bag_regions
        result["is_near_shelf"] = result["near_torso"] or result["near_bag"]

        return result

    @staticmethod
    def _torso_center(
        bbox: tuple[int, int, int, int],
        left_shoulder: tuple[float, float] | None,
        right_shoulder: tuple[float, float] | None,
        left_hip: tuple[float, float] | None,
        right_hip: tuple[float, float] | None,
    ) -> tuple[float, float]:
        points = [point for point in (left_shoulder, right_shoulder, left_hip, right_hip) if point is not None]
        if points:
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            return (sum(xs) / len(xs), sum(ys) / len(ys))

        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, y1 + (y2 - y1) * 0.55)
