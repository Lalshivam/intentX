from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _center_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    acx, acy = _bbox_center(a)
    bcx, bcy = _bbox_center(b)
    dx = acx - bcx
    dy = acy - bcy
    return ((dx * dx) + (dy * dy)) ** 0.5


def _trim(history: deque, timestamp: float, window_seconds: float) -> None:
    while history and history[0][0] < timestamp - window_seconds:
        history.popleft()


def _point_speed(history: deque) -> float:
    if len(history) < 2:
        return 0.0
    previous_time, previous_point = history[-2]
    current_time, current_point = history[-1]
    dt = max(1e-3, current_time - previous_time)
    dx = current_point[0] - previous_point[0]
    dy = current_point[1] - previous_point[1]
    return ((dx * dx) + (dy * dy)) ** 0.5 / dt


def _point_direction(history: deque) -> tuple[float, float]:
    if len(history) < 2:
        return (0.0, 0.0)
    _, previous_point = history[-2]
    _, current_point = history[-1]
    dx = current_point[0] - previous_point[0]
    dy = current_point[1] - previous_point[1]
    norm = max(1.0, (dx * dx + dy * dy) ** 0.5)
    return (dx / norm, dy / norm)


@dataclass
class PersonState:
    track_id: int
    last_seen: float = 0.0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    keypoints: object | None = None
    label: str = "NORMAL"
    suspicion_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    left_wrist_history: deque = field(default_factory=lambda: deque(maxlen=90))
    right_wrist_history: deque = field(default_factory=lambda: deque(maxlen=90))
    bbox_centers: deque = field(default_factory=lambda: deque(maxlen=90))
    head_offsets: deque = field(default_factory=lambda: deque(maxlen=90))
    shelf_entries: deque = field(default_factory=lambda: deque(maxlen=90))
    torso_contacts: deque = field(default_factory=lambda: deque(maxlen=90))
    bag_contacts: deque = field(default_factory=lambda: deque(maxlen=90))
    center_crossings: deque = field(default_factory=lambda: deque(maxlen=90))
    item_contacts: deque = field(default_factory=lambda: deque(maxlen=90))
    last_zone_names: set[str] = field(default_factory=set)
    loiter_start: float | None = None
    last_shelf_timestamp: float | None = None
    last_item_touch_timestamp: float | None = None
    just_lost_item_near_torso: bool = False

    def loiter_duration(self, now: float) -> float:
        if self.loiter_start is None:
            return 0.0
        return max(0.0, now - self.loiter_start)

    def hand_metrics(self) -> dict:
        return {
            "left_speed": _point_speed(self.left_wrist_history),
            "right_speed": _point_speed(self.right_wrist_history),
            "left_direction": _point_direction(self.left_wrist_history),
            "right_direction": _point_direction(self.right_wrist_history),
        }


class PersonTracker:
    def __init__(self, retention_seconds: float = 2.5) -> None:
        self.retention_seconds = retention_seconds
        self.people: dict[int, PersonState] = {}
        self.next_track_id = 1

    def assign_tracks(self, bboxes: list[tuple[int, int, int, int]], timestamp: float) -> list[int]:
        self.prune(timestamp)
        if not bboxes:
            return []

        # Match detections to recently seen tracks by nearest bbox center.
        recent_tracks = [
            (track_id, state)
            for track_id, state in self.people.items()
            if state.last_seen >= timestamp - self.retention_seconds
        ]

        matches: list[int | None] = [None] * len(bboxes)
        used_tracks: set[int] = set()
        candidates: list[tuple[float, int, int]] = []

        for detection_index, bbox in enumerate(bboxes):
            for track_id, state in recent_tracks:
                distance = _center_distance(bbox, state.bbox)
                candidates.append((distance, detection_index, track_id))

        candidates.sort(key=lambda item: item[0])

        for distance, detection_index, track_id in candidates:
            if matches[detection_index] is not None or track_id in used_tracks:
                continue

            # Distance gate prevents accidental cross-matching when people are far apart.
            bbox = bboxes[detection_index]
            scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1], 80)
            if distance > scale * 1.5:
                continue

            matches[detection_index] = track_id
            used_tracks.add(track_id)

        for index, matched in enumerate(matches):
            if matched is not None:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            self.people[track_id] = PersonState(track_id=track_id)
            matches[index] = track_id

        return [track_id for track_id in matches if track_id is not None]

    def update(
        self,
        track_id: int,
        bbox: tuple[int, int, int, int],
        keypoints,
        interaction: dict,
        timestamp: float,
    ) -> PersonState:
        state = self.people.setdefault(track_id, PersonState(track_id=track_id))
        state.last_seen = timestamp
        state.bbox = bbox
        state.keypoints = keypoints

        center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        state.bbox_centers.append((timestamp, center))

        for name, history in (
            ("left_wrist", state.left_wrist_history),
            ("right_wrist", state.right_wrist_history),
        ):
            point = interaction.get(name)
            if point is not None:
                history.append((timestamp, point))

        state.head_offsets.append((timestamp, interaction.get("head_offset", 0.0)))

        current_zones = set(interaction.get("active_zones", []))
        new_zones = current_zones - state.last_zone_names
        for zone_name in new_zones:
            state.shelf_entries.append((timestamp, zone_name))
            state.last_shelf_timestamp = timestamp
        state.last_zone_names = current_zones

        if interaction.get("near_torso"):
            state.torso_contacts.append((timestamp, 1))
        if interaction.get("near_bag"):
            state.bag_contacts.append((timestamp, 1))

        torso_center = interaction.get("torso_center")
        if torso_center is not None:
            for history in (state.left_wrist_history, state.right_wrist_history):
                if len(history) >= 2:
                    _, previous_point = history[-2]
                    _, current_point = history[-1]
                    previous_side = previous_point[0] - torso_center[0]
                    current_side = current_point[0] - torso_center[0]
                    if previous_side == 0:
                        previous_side = current_side
                    if previous_side * current_side < 0:
                        state.center_crossings.append((timestamp, current_side))

        item_touch_labels = interaction.get("item_touch_labels", [])
        if item_touch_labels:
            state.item_contacts.append((timestamp, list(item_touch_labels)))
            state.last_item_touch_timestamp = timestamp
            state.just_lost_item_near_torso = False
        else:
            recently_had_item = (
                state.last_item_touch_timestamp is not None
                and timestamp - state.last_item_touch_timestamp <= 1.0
            )
            state.just_lost_item_near_torso = recently_had_item and interaction.get("near_torso", False)

        if interaction.get("is_near_shelf"):
            if state.loiter_start is None:
                state.loiter_start = timestamp
        else:
            state.loiter_start = None

        for history in (
            state.left_wrist_history,
            state.right_wrist_history,
            state.bbox_centers,
            state.head_offsets,
            state.shelf_entries,
            state.torso_contacts,
            state.bag_contacts,
            state.center_crossings,
            state.item_contacts,
        ):
            _trim(history, timestamp, self.retention_seconds)

        self.prune(timestamp)
        return state

    def prune(self, timestamp: float) -> None:
        stale_track_ids = [
            track_id
            for track_id, state in self.people.items()
            if state.last_seen < timestamp - self.retention_seconds
        ]
        for track_id in stale_track_ids:
            del self.people[track_id]
