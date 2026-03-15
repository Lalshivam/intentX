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


def _point_in_rect(point: tuple[float, float] | None, rect: tuple[int, int, int, int]) -> bool:
    if point is None:
        return False
    x, y = point
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def _distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    if a is None or b is None:
        return float("inf")
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt((dx * dx) + (dy * dy))


def _bbox_overlap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(min(area_a, area_b))


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
        x1, y1, x2, y2 = bbox
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)

        left_wrist = _point_from_keypoint(keypoints, LEFT_WRIST)
        right_wrist = _point_from_keypoint(keypoints, RIGHT_WRIST)
        left_shoulder = _point_from_keypoint(keypoints, LEFT_SHOULDER)
        right_shoulder = _point_from_keypoint(keypoints, RIGHT_SHOULDER)
        left_hip = _point_from_keypoint(keypoints, LEFT_HIP)
        right_hip = _point_from_keypoint(keypoints, RIGHT_HIP)
        nose = _point_from_keypoint(keypoints, NOSE)

        torso_center = self._torso_center(bbox, left_shoulder, right_shoulder, left_hip, right_hip)
        torso_radius = max(32.0, box_width * 0.22)

        bag_regions: list[tuple[int, int, int, int]] = []
        for item in items:
            item_bbox = item.get("bbox")
            item_label = item.get("label")
            if item_bbox is None or item_label is None:
                continue
            item_box = tuple(item_bbox)
            if item_label in BAG_LABELS and _bbox_overlap(bbox, item_box) >= 0.1:
                bag_regions.append(item_box)

        if not bag_regions:
            hip_band_top = int(y1 + box_height * 0.45)
            hip_band_bottom = int(y1 + box_height * 0.92)
            hip_band_left = int(x1 - box_width * 0.05)
            hip_band_right = int(x2 + box_width * 0.05)
            bag_regions.append((hip_band_left, hip_band_top, hip_band_right, hip_band_bottom))

        near_torso = min(_distance(left_wrist, torso_center), _distance(right_wrist, torso_center)) <= torso_radius
        near_bag = any(
            _point_in_rect(left_wrist, region) or _point_in_rect(right_wrist, region)
            for region in bag_regions
        )

        item_touch_labels: list[str] = []
        for item in items:
            item_bbox = item.get("bbox")
            item_label = item.get("label")
            if item_bbox is None or item_label is None:
                continue
            item_center = ((item_bbox[0] + item_bbox[2]) / 2.0, (item_bbox[1] + item_bbox[3]) / 2.0)
            item_radius = max(24.0, min(item_bbox[2] - item_bbox[0], item_bbox[3] - item_bbox[1]) * 0.6)
            if _distance(left_wrist, item_center) <= item_radius or _distance(right_wrist, item_center) <= item_radius:
                item_touch_labels.append(item_label)

        shoulder_center_x = None
        shoulder_width = 0.0
        if left_shoulder is not None and right_shoulder is not None:
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
            shoulder_width = max(1.0, abs(right_shoulder[0] - left_shoulder[0]))

        head_offset = 0.0
        if nose is not None and shoulder_center_x is not None:
            head_offset = (nose[0] - shoulder_center_x) / shoulder_width

        return {
            "left_wrist": left_wrist,
            "right_wrist": right_wrist,
            "torso_center": torso_center,
            "near_torso": near_torso,
            "near_bag": near_bag,
            "active_zones": [],
            "is_near_shelf": near_torso or near_bag,
            "head_offset": head_offset,
            "item_touch_labels": item_touch_labels,
            "bag_regions": bag_regions,
        }

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
