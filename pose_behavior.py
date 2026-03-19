from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from person_tracker import PersonState


@dataclass
class BehaviorDecision:
    label: str
    score: float
    reasons: list[str]
    metrics: dict


def _recent_count(history, now: float, window_seconds: float) -> int:
    return sum(1 for timestamp, _ in history if timestamp >= now - window_seconds)


def _count_head_turns(state: PersonState, now: float, window_seconds: float = 1.5) -> int:
    recent = [offset for timestamp, offset in state.head_offsets if timestamp >= now - window_seconds]
    if len(recent) < 3:
        return 0
    turns = 0
    previous_sign = 0
    for offset in recent:
        sign = 1 if offset > 0.18 else -1 if offset < -0.18 else 0
        if sign == 0:
            continue
        if previous_sign and sign != previous_sign:
            turns += 1
        previous_sign = sign
    return turns


def _recent_item_labels(state: PersonState, now: float, window_seconds: float = 1.2) -> set[str]:
    labels: set[str] = set()
    for timestamp, touched_labels in state.item_contacts:
        if timestamp < now - window_seconds:
            continue
        for label in touched_labels:
            labels.add(label)
    return labels


def classify_behavior(state: PersonState, now: float, item_memory: Any = None) -> BehaviorDecision:
    score = 0.0
    reasons: list[str] = []
    recent_torso = _recent_count(state.torso_contacts, now, 1.2) > 0
    recent_bag = _recent_count(state.bag_contacts, now, 1.2) > 0
    recent_interaction = recent_torso or recent_bag

    if recent_torso:
        score += 1.8
        reasons.append("concealment_near_torso")

    if recent_bag:
        score += 2.0
        reasons.append("bag_interaction")

    if state.just_lost_item_near_torso:
        score += 2.3
        reasons.append("item_disappearance")

    if item_memory is not None and recent_interaction:
        for label in _recent_item_labels(state, now):
            if item_memory.disappeared_recently(label, now):
                score += 2.5
                if "item_disappearance" not in reasons:
                    reasons.append("item_disappearance")
                break

    recent_crossings = _recent_count(state.center_crossings, now, 1.5)
    if recent_interaction and recent_crossings >= 1:
        score += 1.5
        reasons.append("concealment_gesture")

    head_turns = _count_head_turns(state, now)
    if recent_interaction and head_turns >= 2:
        score += 1.2
        reasons.append("look_around_and_grab")

    loiter_duration = state.loiter_duration(now)
    if loiter_duration >= 5.0:
        score += 1.0
        reasons.append("long_loitering")

    hand_metrics = state.hand_metrics()
    fast_hand = max(hand_metrics["left_speed"], hand_metrics["right_speed"])
    if recent_interaction and fast_hand >= 180.0:
        score += 0.7
        reasons.append("rapid_hand_retract")

    label = "SUSPICIOUS" if score >= 5.0 else "NORMAL"
    return BehaviorDecision(
        label=label,
        score=score,
        reasons=reasons,
        metrics={
            "head_turns": head_turns,
            "loiter_duration": round(loiter_duration, 2),
            "left_hand_speed": round(hand_metrics["left_speed"], 2),
            "right_hand_speed": round(hand_metrics["right_speed"], 2),
        },
    )
